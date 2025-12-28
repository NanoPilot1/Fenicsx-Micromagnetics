import dolfinx
from dolfinx import fem, io
from dolfinx.fem import  Constant, functionspace, form
from dolfinx.fem.petsc import (assemble_vector,)
from mpi4py import MPI

from .Exchange import ExchangeField
from .Demag import DemagField
from .Anisotropy import AnisotropyField
from .DMI_Bulk import DMIBULK
from .DMI_Interfacial import DMIInterfacial
from .Cubic_Anisotropy import CubicAnisotropyField


import adios4dolfinx as ad
import ufl
import numpy as np
from petsc4py import PETSc
from time import perf_counter
from pathlib import Path
import sys


# ---------------------------------------------------------
#  Effective Field Class
# ---------------------------------------------------------
class EffectiveField:
    def __init__(
        self,
        mesh,
        Ms,
        Aex,
        Ku,
        n_ani_vec,
        D_bulk,
        D_int,
        n0_int_vec,
        Kc1=0.0, 
        u1_cub=None, 
        u2_cub=None,
        gamma=2.211e5,
        alpha=0.5,
        do_precess=1,
        use_demag=True,
        H0_static=None,       
    ):

        self.mesh = mesh
        self.V = fem.functionspace(
            self.mesh, ("Lagrange", 1, (self.mesh.geometry.dim,))
        )
        self.V1 = fem.functionspace(self.mesh, ("Lagrange", 1))

        self.Ms = Ms
        self.gamma = gamma
        self.alpha = alpha
        self.A = Aex
        self.Ku = Ku
        self.D_bulk = D_bulk
        self.D_int = D_int
        self.do_precess = do_precess
        self.use_demag = use_demag

        # Anisotropy axis
        self.n_ani = fem.Function(self.V)
        self.n_ani.x.array[:] = n_ani_vec

        self.Kc1 = float(Kc1)
        self.cubic_field = None

        self.n0_int = fem.Function(self.V)
        self.n0_int.x.array[:] = n0_int_vec

        self.comm = self.mesh.comm
        self.m = fem.Function(self.V)

        self.H_eff = fem.Function(self.V)

        self.prefactor = -self.gamma / ((1 + self.alpha**2))


        # Following Sci. Rep. 15, 15775 (2025), a small penalty term Stab*(1-|m|^2)m  is added to weakly enforce the normalization condition.
        

        self.Stab = self.Ms * self.gamma / (1 + self.alpha**2)*0.5

        self.n_nodes_local = len(self.mesh.geometry.x)
        self.coords = self.mesh.geometry.x

        self.mx = np.zeros(self.n_nodes_local)
        self.my = np.zeros(self.n_nodes_local)
        self.mz = np.zeros(self.n_nodes_local)

        self.Hx = np.zeros(self.n_nodes_local)
        self.Hy = np.zeros(self.n_nodes_local)
        self.Hz = np.zeros(self.n_nodes_local)

        self.mcx = np.zeros(self.n_nodes_local)
        self.mcy = np.zeros(self.n_nodes_local)
        self.mcz = np.zeros(self.n_nodes_local)

        self.mcmx = np.zeros(self.n_nodes_local)
        self.mcmy = np.zeros(self.n_nodes_local)
        self.mcmz = np.zeros(self.n_nodes_local)
        self.norma = np.zeros(self.n_nodes_local)
        self.Hfield = np.zeros(3 * self.n_nodes_local)

    
        self.start, self.end = self.V.dofmap.index_map.local_range
        owned_dofs = self.end - self.start
        self.local_dofs = self.end - self.start
        self.local_size = 3 * self.local_dofs

        v = ufl.TestFunction(self.V)
        tmp_0 = ufl.dot( v, Constant(self.mesh, PETSc.ScalarType((1.0, 1.0, 1.0)))) * ufl.dx

        volN_f = fem.Function(self.V)
        volN_f.x.petsc_vec.set(0.0)
        assemble_vector(volN_f.x.petsc_vec, form(tmp_0))
        volN_f.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,mode=PETSc.ScatterMode.REVERSE,)
        volN_f.x.scatter_forward()
        volN = volN_f.x.array

        self.Hvec = np.zeros(3 * self.n_nodes_local)
        self.dmdt = fem.Function(self.V)

        # ---------------- Effective fields contributions ----------------
        self.demag_field = None
        if self.use_demag:

            if self.mesh.comm.rank == 0:
                print("[Demag] Precomputing the demagnetizing field...", flush=True)
            t0 = perf_counter()

            self.demag_field = DemagField(self.mesh, self.V, self.V1, self.Ms)

            t1 = perf_counter()
            if self.mesh.comm.rank == 0:
                print(f"[Demag] Precomputation finished in {t1 - t0:.2f} s", flush=True)

        self.exchange_field = ExchangeField(
            self.mesh, self.V, self.A, self.Ms, volN 
        )

        self.anisotropy_field = AnisotropyField(
            self.mesh, self.V, self.Ku, self.Ms, n_ani_vec, volN 
        )

        self.DMIBULK = DMIBULK(
            self.mesh, self.V, self.V1, self.D_bulk, self.Ms, volN 
        )

        # Interfacial DMI: can be None if D_int = 0
        self.DMI_int = None
        if abs(self.D_int) > 0.0:
            self.DMI_int = DMIInterfacial(
                self.mesh, self.V, self.V1, self.D_int, n0_int_vec, self.Ms, volN 
            )


        if abs(self.Kc1) > 0.0:
            if (u1_cub is None) or (u2_cub is None):
                raise ValueError("Kc1 != 0 but u1 or u2 is None")

            self.cubic_field = CubicAnisotropyField(self.mesh, self.V, self.Kc1, self.Ms,u1=u1_cub, u2=u2_cub)

            # buffer para Hv_cubic en jac_times_vec (solo owned)
            self.Hv_cubic = np.zeros((self.local_dofs, 3), dtype=np.float64)


        self.JacSteps = 0
        self.LLGSteps = 0

        # ---------------- K_total: sum of all K ----------------
        # Assuming each field has a K array of the same size.
        self.K_total = self.exchange_field.K + self.anisotropy_field.K + self.DMIBULK.K
        if self.DMI_int is not None:
            self.K_total = self.K_total + self.DMI_int.K

        self.m_jac = fem.Function(self.V)
        self.v_jac = fem.Function(self.V)



        self.H_m = fem.Function(self.V)
        self.H_v = fem.Function(self.V)



        self.Jv_buffer = np.zeros(3 * owned_dofs, dtype=np.float64)

        self.M_cached = np.zeros((owned_dofs, 3))
        self.Hm_cached = np.zeros((owned_dofs, 3))



        # H0_static must be 3*N (flattened as m)
        if H0_static is not None:
            self.H0_static = np.array(H0_static, copy=True)
        else:
            self.H0_static = np.zeros(3 * self.n_nodes_local)

        self.H0_static = np.ascontiguousarray(self.H0_static)
        self.H0_owned = self.H0_static[:self.local_size].reshape((-1, 3))

        self.current_time = 0.0  


    def compute_H_eff(self, m):
        """
        Compute the total effective field H_eff(m).

        Contributions:
        - exchange (always)
        - demagnetizing field (optional)
        - uniaxial anisotropy (optional)
        - bulk DMI (optional)
        - interfacial DMI (optional)
        - cubic Anisotropy
        - external field: static + time-dependent (optional)
        """

        He = self.H_eff.x.array


        He[:] = self.exchange_field.compute(m).x.array
        
        if self.demag_field is not None:
            He += self.demag_field.compute(m).x.array

        if abs(self.Ku) > 0.0:
            He += self.anisotropy_field.compute(m).x.array

        if abs(self.D_bulk) > 0.0:
            He += self.DMIBULK.compute(m).x.array

        if self.DMI_int is not None and abs(self.D_int) > 0.0:
            He += self.DMI_int.compute(m).x.array

        if self.cubic_field is not None:
            He += self.cubic_field.compute(m).x.array

        He += self.H0_static


        self.H_eff.x.scatter_forward()

        return He

    # ---------- Compute total energy ----------
    def compute_Energy(self, m):
        E_exch = self.exchange_field.Energy(m)
        E_demag = 0.0
        if self.demag_field is not None:
            E_demag = self.demag_field.Energy(m)

        E_ani = 0.0
        if abs(self.Ku) > 0.0:
            E_ani = self.anisotropy_field.Energy(m)

        E_dmi_bulk = 0.0
        if abs(self.D_bulk) > 0.0:
            E_dmi_bulk = self.DMIBULK.Energy(m)

        E_dmi_int = 0.0
        if self.DMI_int is not None and abs(self.D_int) > 0.0:
            E_dmi_int = self.DMI_int.Energy(m)

        E_cub = 0.0
        if self.cubic_field is not None:
            E_cub = self.cubic_field.Energy(m)

        return E_exch + E_demag + E_ani + E_dmi_bulk + E_dmi_int+ E_cub

    # ---------- Jacobian times vector ----------
    def update_jac_state(self, m_vec):
        """
        m_vec: local NumPy view of m.x.array (length = 3*local_dofs)
        """
        self.m_jac.x.array[:self.local_size] = m_vec

        self.m_jac.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES,mode=PETSc.ScatterMode.FORWARD,)

        self.H_m.x.petsc_vec.set(0.0)
        self.K_total.mult(self.m_jac.x.petsc_vec, self.H_m.x.petsc_vec)
        self.H_m.x.scatter_forward()

        M_loc = m_vec.reshape(-1, 3)
        Hm_loc = self.H_m.x.array[:self.local_size].reshape(-1, 3)

        if self.cubic_field is not None:
            Hc = self.cubic_field.compute(self.m_jac).x.array[:self.local_size].reshape(-1, 3)
            Hm_loc = Hm_loc + Hc

        self.M_cached[:, :] = M_loc
        self.Hm_cached[:, :] = Hm_loc + self.H0_owned

    def jac_vec_times(self, m_unused, v, out):


        self.JacSteps += 1

        self.H_v.x.petsc_vec.set(0.0)


        self.v_jac.x.array[:self.local_size] = v
        self.v_jac.x.petsc_vec.ghostUpdate( addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD,)

        self.K_total.mult(self.v_jac.x.petsc_vec, self.H_v.x.petsc_vec)
        self.H_v.x.scatter_forward()


        

        M = self.M_cached
        Hm = self.Hm_cached  
 
        #
        V = v.reshape(-1, 3)
        Hv_lin = self.H_v.x.array[:self.local_size].reshape(-1, 3)

        Hv = Hv_lin 

            
        if self.cubic_field is not None:
            self.cubic_field.jac_times_vec_owned(M, V, self.Hv_cubic)
            Hv = Hv_lin + self.Hv_cubic
        else:
            Hv = Hv_lin


        cross_vHm = np.cross(V, Hm)
        cross_mHv = np.cross(M, Hv)
        prec = self.do_precess * (cross_vHm + cross_mHv)

        mdHm = np.sum(M * Hm, axis=1)[:, None]
        mdHv = np.sum(M * Hv, axis=1)[:, None]
        vdHm = np.sum(V * Hm, axis=1)[:, None]
        mdv = np.sum(M * V, axis=1)[:, None]
        mdmm = np.sum(M * M, axis=1)[:, None]

        term1 = V * mdHm
        term2 = -2.0 * Hm * mdv
        term3 = M * (vdHm + mdHv)
        term4 = -Hv * mdmm
        damp = term1 + term2 + term3 + term4

        coef = -self.gamma / (1.0 + self.alpha**2)


        Jv = coef * (prec + self.alpha * damp)

        extra = self.Stab * (V * (1.0 - mdmm) - 2.0 * M * mdv)
        Jv += extra

        out[:] = Jv.reshape(-1)

    # ---------- RHS LLG ----------
    def llg_rhs(self, m):
        self.LLGSteps += 1

        m_numeric = m.x.array
        self.Hfield[:] = self.compute_H_eff(m)

        self.mx[:] = m_numeric[0::3]
        self.my[:] = m_numeric[1::3]
        self.mz[:] = m_numeric[2::3]

        self.norma[:] = self.mx[:] * self.mx[:]+ self.my[:] * self.my[:]+ self.mz[:] * self.mz[:]

        self.Hx[:] = self.Hfield[0::3]
        self.Hy[:] = self.Hfield[1::3]
        self.Hz[:] = self.Hfield[2::3]

        self.mcx[:] = self.my * self.Hz - self.mz * self.Hy
        self.mcy[:] = self.mz * self.Hx - self.mx * self.Hz
        self.mcz[:] = self.mx * self.Hy - self.my * self.Hx

        self.mcmx[:] = self.my * self.mcz - self.mz * self.mcy
        self.mcmy[:] = self.mz * self.mcx - self.mx * self.mcz
        self.mcmz[:] = self.mx * self.mcy - self.my * self.mcx

        self.dmdt.x.array[0::3] = self.prefactor * (self.do_precess * self.mcx + self.alpha * self.mcmx) + self.Stab*(1.0 - self.norma) * self.mx[:]
        self.dmdt.x.array[1::3] = self.prefactor * (self.do_precess * self.mcy + self.alpha * self.mcmy) + self.Stab*(1.0 - self.norma) * self.my[:]
        self.dmdt.x.array[2::3] = self.prefactor * (self.do_precess * self.mcz + self.alpha * self.mcmz) + self.Stab*(1.0 - self.norma) * self.mz[:] 

        return self.dmdt

    # ---------- IFunction  ----------
    def ifunction(self, ts, t, y, ydot, f):

        #self.LLGSteps += 1

        self.current_time = t

        y.copy(self.m.x.petsc_vec)
        self.m.x.scatter_forward()

        dmdt = self.llg_rhs(self.m)
        dmdt.x.scatter_forward()

        f.waxpy(-1.0, dmdt.x.petsc_vec, ydot)
        return 0



    def set_uniform_field(self, Hx, Hy, Hz):
        self.H0_static[0::3] = Hx
        self.H0_static[1::3] = Hy
        self.H0_static[2::3] = Hz







class StopByMaxDmdtFD:


    def __init__(self, comm, stopping_dm_dt_deg_ns, vec_template,
                check_every=10, print_every_hit=True):
        self.comm = comm
        self.thresh_deg_ns = float(stopping_dm_dt_deg_ns)
        self.check_every = int(check_every)
        self.print_every_hit = bool(print_every_hit)


        self.u_prev = vec_template.duplicate()
        self.du = vec_template.duplicate()


        r0, r1 = vec_template.getOwnershipRange()
        self.n_owned = int(r1 - r0)

        self.n_owned3 = (self.n_owned // 3) * 3

        self.t_prev = None
        self.last_max_dmdt_deg_ns = float("nan")


        vec_template.copy(self.u_prev)

    def __call__(self, ts):
        if self.thresh_deg_ns <= 0.0:
            return 0

        step = ts.getStepNumber()
        if self.check_every > 1 and (step % self.check_every) != 0:
            return 0

        u = ts.getSolution()
        t = ts.getTime()

        if self.t_prev is None:
            u.copy(self.u_prev)
            self.t_prev = t
            return 0

        dt = t - self.t_prev
        if dt <= 0.0:
            u.copy(self.u_prev)
            self.t_prev = t
            return 0

        # du = u - u_prev
        u.copy(self.du)
        self.du.axpy(-1.0, self.u_prev)

        arr = self.du.getArray(readonly=True)
        a = np.asarray(arr[:self.n_owned3]).reshape((-1, 3))
        if a.size:
            d = np.sqrt((a * a).sum(axis=1))
            max_local = float(d.max()) / dt          # 1/s
        else:
            max_local = 0.0

        max_global = self.comm.allreduce(max_local, op=MPI.MAX)
        max_deg_ns = max_global * (180.0 / np.pi) * 1e-9
        self.last_max_dmdt_deg_ns = max_deg_ns

        if self.comm.rank == 0 and self.print_every_hit:
            print(f"[dmdt] step={step} t={t*1e9:.6f} ns  max|dm/dt|={max_deg_ns:.6e} deg/ns", flush=True)

        if max_deg_ns < self.thresh_deg_ns:
            #if self.comm.rank == 0:
                #print(f"[STOP] step={step} t={t*1e9:.6f} ns  max|dm/dt|={max_deg_ns:.6e} < {self.thresh_deg_ns:.6e} deg/ns",flush=True)
            ts.setConvergedReason(PETSc.TS.ConvergedReason.CONVERGED_USER)


        u.copy(self.u_prev)
        self.t_prev = t
        return 0
        
    def reset(self, vec_current):
        vec_current.copy(self.u_prev)
        self.t_prev = None
        self.last_max_dmdt_deg_ns = float("nan")


def skew_from_vec(a):

    ax = a[:, 0]; ay = a[:, 1]; az = a[:, 2]
    S = np.zeros((a.shape[0], 3, 3), dtype=a.dtype)
    S[:, 0, 1] = -az
    S[:, 0, 2] =  ay
    S[:, 1, 0] =  az
    S[:, 1, 2] = -ax
    S[:, 2, 0] = -ay
    S[:, 2, 1] =  ax
    return S


class JvContext:
    def __init__(self, hef_):
        self.hef = hef_
        self.shift = 0.0
        self.calls = 0
        self.callsPre = 0

        diag = self.hef.K_total.getDiagonal()
        self.diagK = diag.getArray(readonly=True).copy()

        self.gamma = float(self.hef.gamma)
        self.do_precess = float(self.hef.do_precess)

        self.enable_pc = True


        self.w = None
        self.denom = None
        self.s_eff = None

        diagK = diag.getArray(readonly=True).copy()
        d = diagK.reshape(-1, 3)
        self.kappa_abs = np.mean(np.abs(d), axis=1).astype(np.float64)  
        self.kappa_sgn = np.mean(d, axis=1).astype(np.float64)         


        g  = float(self.hef.gamma)
        a  = float(self.hef.alpha)
        dp = float(self.hef.do_precess)

        self.c1 = (g / (1.0 + a*a)) * dp
        self.c2 = (g * a / (1.0 + a*a))
        self.Stab = float(self.hef.Stab)

        self.enable_pc = True

        n = self.hef.M_cached.shape[0]
        self.A00 = np.empty(n); self.A01 = np.empty(n); self.A02 = np.empty(n)
        self.A10 = np.empty(n); self.A11 = np.empty(n); self.A12 = np.empty(n)
        self.A20 = np.empty(n); self.A21 = np.empty(n); self.A22 = np.empty(n)

        self.i00 = np.empty(n); self.i01 = np.empty(n); self.i02 = np.empty(n)
        self.i10 = np.empty(n); self.i11 = np.empty(n); self.i12 = np.empty(n)
        self.i20 = np.empty(n); self.i21 = np.empty(n); self.i22 = np.empty(n)

        self._pc_ready = False

        self.base_kappa_abs = self.kappa_abs.copy()
        self.base_kappa_sgn = self.kappa_sgn.copy()
        self.kappa_work = np.empty_like(self.base_kappa_abs)

        self.include_cubic_pc = True
        self._kappa_cub = np.empty_like(self.base_kappa_abs)

    def _compute_kappa_cubic(self, M, out_kappa):

        cub = getattr(self.hef, "cubic_field", None)
        if cub is None:
            out_kappa[:] = 0.0
            return

        u1 = cub.u1A[:M.shape[0]]
        u2 = cub.u2A[:M.shape[0]]
        u3 = cub.u3A[:M.shape[0]]
        pref = float(cub.pref)

        a1 = np.einsum("ij,ij->i", M, u1)
        a2 = np.einsum("ij,ij->i", M, u2)
        a3 = np.einsum("ij,ij->i", M, u3)

        a1_2 = a1*a1; a2_2 = a2*a2; a3_2 = a3*a3

        g1 = u1*(a2_2 + a3_2)[:,None] + (2.0*a1*a2)[:,None]*u2 + (2.0*a1*a3)[:,None]*u3
        g2 = u2*(a3_2 + a1_2)[:,None] + (2.0*a2*a3)[:,None]*u3 + (2.0*a2*a1)[:,None]*u1
        g3 = u3*(a1_2 + a2_2)[:,None] + (2.0*a3*a1)[:,None]*u1 + (2.0*a3*a2)[:,None]*u2


        diag = pref*(u1*g1 + u2*g2 + u3*g3)    

        out_kappa[:] = (np.abs(diag[:,0]) + np.abs(diag[:,1]) + np.abs(diag[:,2])) / 3.0


    def mult(self, A, x, y):
        self.calls += 1
        xv = x.getArray(readonly=True)
        yv = y.getArray()

        m_vec = self.hef.m.x.petsc_vec.getArray(readonly=True)
        self.hef.jac_vec_times(None, xv, out=self.hef.Jv_buffer)

        yv[:] = self.shift * xv - self.hef.Jv_buffer



    def update_pc_full_fast(self, shift, include_stab=True, use_abs_kappa=True,
                            eps_reg=1e-14, det_eps=1e-30):
        self.shift = float(shift)

        M = self.hef.M_cached   
        H = self.hef.Hm_cached  
        mx, my, mz = M[:,0], M[:,1], M[:,2]
        hx, hy, hz = H[:,0], H[:,1], H[:,2]

        kappa_base = self.base_kappa_abs if use_abs_kappa else self.base_kappa_sgn
        kappa = self.kappa_work
        kappa[:] = kappa_base

        #cub = getattr(self.hef, "cubic_field", None)
        #if self.include_cubic_pc and (cub is not None):
        #    self._compute_kappa_cubic(M, self._kappa_cub)
        #    kappa += self._kappa_cub


        c1 = self.c1
        c2 = self.c2
        Stab = self.Stab

        mdH = mx*hx + my*hy + mz*hz
        mdm = mx*mx + my*my + mz*mz

        Jp00 = 0.0
        Jp01 = -c1*(-hz - kappa*(-mz))   # -c1*(S_H01 - kappa*S_m01)
        Jp02 = -c1*( hy - kappa*( my))
        Jp10 = -c1*( hz - kappa*( mz))
        Jp11 = 0.0
        Jp12 = -c1*(-hx - kappa*(-mx))
        Jp20 = -c1*(-hy - kappa*(-my))
        Jp21 = -c1*( hx - kappa*( mx))
        Jp22 = 0.0


        # Diagonal:
        B00 = mdH + mx*hx - 2*hx*mx  # = mdH - hx*mx
        B11 = mdH - hy*my
        B22 = mdH - hz*mz
        C00 = mx*mx - mdm
        C11 = my*my - mdm
        C22 = mz*mz - mdm
        #
        # Off-diagonal:
        B01 = mx*hy - 2*hx*my
        B02 = mx*hz - 2*hx*mz
        B10 = my*hx - 2*hy*mx
        B12 = my*hz - 2*hy*mz
        B20 = mz*hx - 2*hz*mx
        B21 = mz*hy - 2*hz*my

        C01 = mx*my; C02 = mx*mz
        C10 = my*mx; C12 = my*mz
        C20 = mz*mx; C21 = mz*my

        Jd00 = c2*(B00 + kappa*C00)
        Jd11 = c2*(B11 + kappa*C11)
        Jd22 = c2*(B22 + kappa*C22)

        Jd01 = c2*(B01 + kappa*C01)
        Jd02 = c2*(B02 + kappa*C02)
        Jd10 = c2*(B10 + kappa*C10)
        Jd12 = c2*(B12 + kappa*C12)
        Jd20 = c2*(B20 + kappa*C20)
        Jd21 = c2*(B21 + kappa*C21)


        if include_stab:
            s0 = (1.0 - mdm)
            # diag: s0 - 2 m_i^2 ; offdiag: -2 m_i m_j
            Js00 = Stab*(s0 - 2*mx*mx)
            Js11 = Stab*(s0 - 2*my*my)
            Js22 = Stab*(s0 - 2*mz*mz)
            Js01 = Stab*(-2*mx*my); Js02 = Stab*(-2*mx*mz)
            Js10 = Stab*(-2*my*mx); Js12 = Stab*(-2*my*mz)
            Js20 = Stab*(-2*mz*mx); Js21 = Stab*(-2*mz*my)
        else:
            Js00=Js11=Js22=0.0
            Js01=Js02=Js10=Js12=Js20=Js21=0.0

        # J = Jp + Jd + Js
        J00 = (0.0)     + Jd00 + Js00
        J11 = (0.0)     + Jd11 + Js11
        J22 = (0.0)     + Jd22 + Js22

        J01 = Jp01 + Jd01 + Js01
        J02 = Jp02 + Jd02 + Js02
        J10 = Jp10 + Jd10 + Js10
        J12 = Jp12 + Jd12 + Js12
        J20 = Jp20 + Jd20 + Js20
        J21 = Jp21 + Jd21 + Js21

        # A = shift*I - J + eps_reg*I
        s = self.shift + eps_reg
        A00 = self.A00; A01 = self.A01; A02 = self.A02
        A10 = self.A10; A11 = self.A11; A12 = self.A12
        A20 = self.A20; A21 = self.A21; A22 = self.A22

        A00[:] = s - J00;  A01[:] =   - J01;  A02[:] =   - J02
        A10[:] =   - J10;  A11[:] = s - J11;  A12[:] =   - J12
        A20[:] =   - J20;  A21[:] =   - J21;  A22[:] = s - J22


        # det = a00*(a11*a22-a12*a21) - a01*(a10*a22-a12*a20) + a02*(a10*a21-a11*a20)
        m00 = A11*A22 - A12*A21
        m01 = A10*A22 - A12*A20
        m02 = A10*A21 - A11*A20
        det = A00*m00 - A01*m01 + A02*m02


        det_abs = np.abs(det)
        det = np.where(det_abs < det_eps, det + np.sign(det + det_eps)*det_eps, det)
        invdet = 1.0/det

        i00=i00_ = self.i00; i01=self.i01; i02=self.i02
        i10=self.i10; i11=self.i11; i12=self.i12
        i20=self.i20; i21=self.i21; i22=self.i22

        i00[:] =  (A11*A22 - A12*A21) * invdet
        i01[:] = -(A01*A22 - A02*A21) * invdet
        i02[:] =  (A01*A12 - A02*A11) * invdet

        i10[:] = -(A10*A22 - A12*A20) * invdet
        i11[:] =  (A00*A22 - A02*A20) * invdet
        i12[:] = -(A00*A12 - A02*A10) * invdet

        i20[:] =  (A10*A21 - A11*A20) * invdet
        i21[:] = -(A00*A21 - A01*A20) * invdet
        i22[:] =  (A00*A11 - A01*A10) * invdet

        self._pc_ready = True

    def apply(self, pc, x, y):
        self.callsPre += 1
        if (not self.enable_pc) or (not self._pc_ready):
            x.copy(y)
            return

        xv = x.getArray(readonly=True)
        yv = y.getArray()

        x0 = xv[0::3]; x1 = xv[1::3]; x2 = xv[2::3]

        y0 = self.i00*x0 + self.i01*x1 + self.i02*x2
        y1 = self.i10*x0 + self.i11*x1 + self.i12*x2
        y2 = self.i20*x0 + self.i21*x1 + self.i22*x2

        yv[0::3] = y0
        yv[1::3] = y1
        yv[2::3] = y2


# ---------------------------------------------------------
#  LLG main class
# ---------------------------------------------------------
class LLG:
    def __init__(self, mesh, Ms, gamma=2.211e5, alpha=0.5, do_precess=1):
        self.mesh = mesh
        self.Ms = Ms
        self.gamma = gamma
        self.alpha = alpha
        self.do_precess = do_precess

        
        self._Aex = 0.0
        self._Ku = 0.0
        self._n_ani = None

        self._D_bulk = 0.0
        self._D_int = 0.0
        self._n0_int = None
        self._H0_vec = None    


        self._Kc1 = 0.0
        self._u1_cub = None
        self._u2_cub = None      

        self._has_exchange = False
        self._has_demag = False
        self._has_anisotropy = False
        self._has_dmi_bulk = False
        self._has_dmi_int = False
        self._has_cubic = False

        self.hef: EffectiveField | None = None


        self.ts = None
        self.ctx = None
        self.J = None
        self.y = None
        self.stopper = None
        self._solver_ready = False


    def add_exchange(self, Aex):
        self._Aex = Aex
        self._has_exchange = True

    def add_demag(self):
        self._has_demag = True 

    def add_anisotropy(self, Ku, n_vec):
        self._Ku = Ku
        self._n_ani = n_vec
        self._has_anisotropy = True

    def add_dmi_bulk(self, D_bulk):
        self._D_bulk = D_bulk
        self._has_dmi_bulk = True

    def add_dmi_interfacial(self, D_int, n0_vec):
        self._D_int = D_int
        self._n0_int = n0_vec
        self._has_dmi_int = True

    def add_external_field(self, H0_vec=None, H_time_func=None):

        self._H0_vec = H0_vec


    def add_cubic_anisotropy(self, Kc1, u1_vec, u2_vec):

        self._Kc1 = float(Kc1)
        self._u1_cub = u1_vec
        self._u2_cub = u2_vec
        self._has_cubic = True

    # build effective field

    def _build_effective_field(self):
        Aex = self._Aex if self._has_exchange else 0.0

        if self._has_anisotropy and self._n_ani is not None:
            Ku = self._Ku
            n_ani_vec = self._n_ani
        else:
            Ku = 0.0
            n_ani_vec = np.zeros(3 * len(self.mesh.geometry.x), dtype=np.float64)

        if self._has_dmi_bulk:
            D_bulk = self._D_bulk
        else:
            D_bulk = 0.0

        if self._has_dmi_int and self._n0_int is not None:
            D_int = self._D_int
            n0_int_vec = self._n0_int
        else:
            D_int = 0.0
            n0_int_vec = np.zeros(3 * len(self.mesh.geometry.x), dtype=np.float64)

        if self._has_cubic and self._u1_cub is not None and self._u2_cub is not None:
            Kc1 = self._Kc1
            u1_cub = self._u1_cub
            u2_cub = self._u2_cub
        else:
            Kc1 = 0.0
            u1_cub = None
            u2_cub = None

        H0_static = self._H0_vec   # can be none

        self.hef = EffectiveField(
            self.mesh,
            self.Ms,
            Aex,
            Ku,
            n_ani_vec,
            D_bulk,
            D_int,
            n0_int_vec,
            Kc1=Kc1,
            u1_cub=u1_cub,
            u2_cub=u2_cub,
            gamma=self.gamma,
            alpha=self.alpha,
            do_precess=self.do_precess,
            use_demag=self._has_demag,
            H0_static=H0_static,
        )

    def _cancel_ts_monitors(self):

        if self.ts is None:
            return
        try:
            self.ts.monitorCancel()
        except Exception:
            try:
                self.ts.setMonitor(None)
            except Exception:
                pass


    def _reset_ts_run(self, t0, t_final, dt_init):
        ts = self.ts
        ts.setTime(float(t0))
        ts.setMaxTime(float(t_final))
        ts.setTimeStep(float(dt_init))
        ts.setExactFinalTime(PETSc.TS.ExactFinalTime.STEPOVER)

        ts.restartStep()  

        try:
            ts.setStepNumber(0)
        except Exception:
            pass

        if self.stopper is not None:
            self.stopper.reset(self.y)


    def _run_ts(self):
        """run TS.solve(self.y) and synchronize hef.m at the end."""
        ts = self.ts
        hef = self.hef
        y = self.y

        tstart = perf_counter()
        ts.solve(y)
        elapsed = perf_counter() - tstart

        y.copy(hef.m.x.petsc_vec)
        hef.m.x.scatter_forward()

        stats = {
            "t_end": float(ts.getTime()),
            "dt_last": float(ts.getTimeStep()),
            "nsteps": int(ts.getStepNumber()),
            "reason": int(ts.getConvergedReason()),
            "maxdmdt_deg_ns": float(self.stopper.last_max_dmdt_deg_ns) if self.stopper is not None else float("nan"),
        }
        return elapsed, stats




    def _reset_run(self, t_final, dt_init=None):
        ts = self.ts
        if dt_init is not None:
            ts.setTimeStep(dt_init)
        ts.setTime(0.0)
        ts.setMaxTime(t_final)
        try:
            ts.setStepNumber(0)
        except Exception:
            pass
        self.stopper.reset(self.y)

    def _ensure_solver(self, m0_array, dt_init,
                    ts_rtol=1e-6, ts_atol=1e-6,
                    snes_rtol=1e-2, snes_atol=1e-4,
                    ksp_rtol=1e-4,
                    stopping_dmdt=0.0,
                    check_every_stop=10, stop_print=False):

        if self.hef is None:
            self._build_effective_field()
        hef = self.hef

        if not self._solver_ready:
            if m0_array is not None:
                hef.m.x.array[:] = m0_array
                hef.m.x.scatter_forward()

            ts = PETSc.TS().create(self.mesh.comm)

            opts = PETSc.Options()
            opts["ts_type"] = "bdf"
            opts["ts_adapt_type"] = "basic"
            opts["ts_adapt_clip"] = "0.1, 3.0"
            opts["ts_adapt_safety"] = 0.9
            opts["ts_adapt_reject_safety"] = 0.1
            opts["ts_adapt_scale_solve_failed"] = 0.25
            opts["ts_adapt_dt_min"] = 1e-17
            opts["ts_adapt_dt_max"] = 1e-10

            opts["snes_type"] = "newtonls"
            opts["snes_linesearch_type"] = "bt"
            opts["snes_linesearch_order"] = 2

            opts["ts_rtol"] = ts_rtol
            opts["ts_atol"] = ts_atol
            opts["snes_rtol"] = snes_rtol
            opts["snes_atol"] = snes_atol
            opts["snes_max_it"] = 8
            opts["ksp_type"] = "gmres"
            opts["ksp_rtol"] = ksp_rtol
            opts["ksp_reuse_preconditioner"] = "true"
            opts["snes_lag_preconditioner"] = 1
            opts["ts_max_snes_failures"] = -1
            opts["ts_max_steps"] = 5000000
            #opts["ksp_converged_reason"] = ""
            #opts['ksp_converged_reason'] = None
            #opts['snes_converged_reason'] = None
            #opts["log_view"] = "" 
            ts.setTime(0.0)
            ts.setTimeStep(dt_init)
            ts.setExactFinalTime(PETSc.TS.ExactFinalTime.STEPOVER)

            snes = ts.getSNES()
            n_loc = hef.m.x.petsc_vec.getLocalSize()
            n_glob = hef.m.x.petsc_vec.getSize()

            J = PETSc.Mat().create(comm=self.mesh.comm)
            ctx = JvContext(hef)
            J.setSizes([[n_loc, n_glob], [n_loc, n_glob]])
            J.setType("python")
            J.setPythonContext(ctx)
            J.setUp()

            ksp = snes.getKSP()
            pc = ksp.getPC()
            pc.setType("python")
            pc.setPythonContext(ctx)

            def IJac(ts_, t, y, ydot, shift, A, B):
                y.copy(hef.m.x.petsc_vec)
                hef.m.x.scatter_forward()
                mloc = hef.m.x.petsc_vec.getArray(readonly=True)
                hef.update_jac_state(mloc)
                ctx.update_pc_full_fast(shift, include_stab=True, use_abs_kappa=True)
                return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

            ts.setIFunction(hef.ifunction)
            ts.setIJacobian(IJac, J)
            ts.setFromOptions()

            y = hef.m.x.petsc_vec.copy()
            y.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES,
                        mode=PETSc.ScatterMode.FORWARD)
            ts.setSolution(y)

            stopper = StopByMaxDmdtFD(
                self.mesh.comm,
                stopping_dm_dt_deg_ns=stopping_dmdt,
                vec_template=y,
                check_every=check_every_stop,
                print_every_hit=stop_print,  
            )
            ts.setPostStep(stopper)
            self.stopper = stopper

            self.ts, self.ctx, self.J, self.y, self.stopper = ts, ctx, J, y, stopper
            self._solver_ready = True

        else:

            if m0_array is not None:
                hef.m.x.array[:] = m0_array
                hef.m.x.scatter_forward()
                self.y.copy(hef.m.x.petsc_vec)





    def relax(
        self,
        m0_array,
        t0,
        t_final,
        dt_init,
        dt_save=None,
        dt_snap=None,
        output_dir="output",
        ts_rtol=1e-6,
        ts_atol=1e-6,
        snes_rtol=1e-2,
        snes_atol=1e-4,
        ksp_rtol=1e-4,
        stopping_dmdt=0.0,
        monitor_fn=None,
        save_final_state=True,
        check_every_stop=5, 
        stop_print=False,
        return_stats=False,
    ):
        """
        Single temporary integrator. Reuses the persistent TS (_ensure_solver).

        - If dt_save/dt_snap are not None: log + snapshots over time (like today).

        - Saves final state (XDMF + BP) if save_final_state=True.

        Returns: (y, ctx, elapsed) or (y, ctx, elapsed, stats) if return_stats=True.
        """

        # 1) asegurar solver persistente (crea TS/J/PC una sola vez)
        self._ensure_solver(
            m0_array=m0_array,
            dt_init=dt_init,
            ts_rtol=ts_rtol, ts_atol=ts_atol,
            snes_rtol=snes_rtol, snes_atol=snes_atol,
            ksp_rtol=ksp_rtol,
            stopping_dmdt=stopping_dmdt,
            check_every_stop=check_every_stop,
            stop_print=stop_print,
        )

        ts = self.ts
        hef = self.hef
        y = self.y
        comm = self.mesh.comm

        if m0_array is not None:
            hef.m.x.array[:] = m0_array
            hef.m.x.scatter_forward()
            y.copy(hef.m.x.petsc_vec)

        if dt_save is not None:
            if dt_snap is None:
                dt_snap = dt_save
            if comm.rank == 0:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            comm.barrier()

        self._cancel_ts_monitors()

        if dt_save is not None:
            log_path = Path(output_dir) / "log.txt"
            last_save_n = {"n": -1}
            last_snap_n = {"n": -1}
            snap_counter = {"k": 0}
            first_print = {"done": False}

            def default_monitor(ts_, step, t, u, hef_, mesh_):
                dt_ts = ts_.getTimeStep()

                Exch = hef_.exchange_field.Energy(hef_.m)
                Demag = hef_.demag_field.Energy(hef_.m) if hef_.demag_field is not None else 0.0
                Ani = hef_.anisotropy_field.Energy(hef_.m) if getattr(hef_, "Ku", 0.0) != 0.0 else 0.0
                DMI_bulk = hef_.DMIBULK.Energy(hef_.m) if getattr(hef_, "D_bulk", 0.0) != 0.0 else 0.0
                DMI_int = hef_.DMI_int.Energy(hef_.m) if (getattr(hef_, "D_int", 0.0) != 0.0 and hef_.DMI_int is not None) else 0.0
                E_cub = hef_.cubic_field.Energy(hef_.m) if hef_.cubic_field is not None else 0.0


                Exch_total = mesh_.comm.gather(Exch, root=0)
                Demag_total = mesh_.comm.gather(Demag, root=0)
                Ani_total = mesh_.comm.gather(Ani, root=0)
                DMI_bulk_total = mesh_.comm.gather(DMI_bulk, root=0)
                DMI_int_total = mesh_.comm.gather(DMI_int, root=0)
                E_cub_total = mesh_.comm.gather(E_cub, root=0)


                mag = mesh_.comm.gather(hef_.m.x.petsc_vec.getArray(readonly=True), root=0)

                torque_norm = np.sqrt(hef_.mcx**2 + hef_.mcy**2 + hef_.mcz**2)
                max_torque_local = float(np.max(torque_norm)) if torque_norm.size else 0.0
                max_torque_all = mesh_.comm.gather(max_torque_local, root=0)

                Hext_local = np.zeros_like(hef_.H_eff.x.array)
                Hext_local += hef_.H0_static  
                Hext_all = mesh_.comm.gather(Hext_local, root=0)

                n_snap = int(np.trunc(t / dt_snap))
                if n_snap != last_snap_n["n"]:
                    last_snap_n["n"] = n_snap
                    filename = Path(output_dir) / f"m{snap_counter['k']:03d}.xdmf"
                    snap_counter["k"] += 1
                    with io.XDMFFile(mesh_.comm, str(filename), "w") as xdmf:
                        xdmf.write_mesh(mesh_)
                        xdmf.write_function(hef_.m)

                if mesh_.comm.rank == 0:
                    mag = np.reshape(np.concatenate(mag), (-1, 3))
                    E_exch = float(np.sum(Exch_total))
                    E_demag = float(np.sum(Demag_total))
                    E_ani = float(np.sum(Ani_total))
                    E_db = float(np.sum(DMI_bulk_total))
                    E_di = float(np.sum(DMI_int_total))
                    E_cub = float(np.sum(E_cub_total))

                    E_tot = E_exch + E_demag + E_ani + E_db + E_di+ E_cub

                    Hext = np.reshape(np.concatenate(Hext_all), (-1, 3))
                    Hx_ext_mean = Hext[:, 0].mean()
                    Hy_ext_mean = Hext[:, 1].mean()
                    Hz_ext_mean = Hext[:, 2].mean()


                    maxtorque = 4 * np.pi * 1e-7 * float(max(max_torque_all)) if max_torque_all else 0.0
                    maxdmdt_deg_ns = float(self.stopper.last_max_dmdt_deg_ns) if self.stopper is not None else 0.0
                    if not np.isfinite(maxdmdt_deg_ns):
                        maxdmdt_deg_ns = 0.0

                    if not first_print["done"]:
                        header = (
                            f"{'time':>10} {'dt':>10} {'<mx>':>15} {'<my>':>15} {'<mz>':>15} "
                            f"{'Hx_ext':>15} {'Hy_ext':>15} {'Hz_ext':>15} "
                            f"{'maxdmdt(deg/ns)':>18} {'max(mxh)':>15} "
                            f"{'E_demag':>15} {'E_exch':>15} {'E_ani':>15} "
                            f"{'E_dmi_bulk':>15} {'E_dmi_int':>15} {'E_cubic':>15}  {'E_total':>15}"
                        )
                        print(header)
                        with open(log_path, "w") as f:
                            f.write(header + "\n")
                        first_print["done"] = True

                    line = (
                        f"{t*1e9:10.4f} {dt_ts*1e9:10.4f}"
                        f"{mag[:,0].mean():15.6f} {mag[:,1].mean():15.6f} {mag[:,2].mean():15.6f} "
                        f"{Hx_ext_mean:15.6e} {Hy_ext_mean:15.6e} {Hz_ext_mean:15.6e} "
                        f"{maxdmdt_deg_ns:18.6e} {maxtorque:15.4e} "
                        f"{E_demag:15.4e} {E_exch:15.4e} {E_ani:15.4e} "
                        f"{E_db:15.4e} {E_di:15.4e} {E_cub:15.4e} {E_tot:15.4e}"
                    )
                    print(line)
                    with open(log_path, "a") as f:
                        f.write(line + "\n")
                    sys.stdout.flush()

            def monitor(ts_, step, t, u):
                n = int(np.trunc(t / dt_save))
                if n != last_save_n["n"]:
                    last_save_n["n"] = n
                    if monitor_fn is not None:
                        monitor_fn(ts_, step, t, u, hef, self.mesh)
                    else:
                        default_monitor(ts_, step, t, u, hef, self.mesh)

            ts.setMonitor(monitor)


        self._reset_ts_run(t0=t0, t_final=t_final, dt_init=dt_init)
        elapsed, stats = self._run_ts()


        if comm.rank == 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        comm.barrier()

        filename = Path(output_dir) / "Relax.xdmf"
        with io.XDMFFile(self.mesh.comm, str(filename), "w") as xdmf:
            xdmf.write_mesh(self.mesh)
            xdmf.write_function(self.hef.m)

        if save_final_state:
            fname = Path(output_dir) / "Relax.bp"
            ad.write_mesh(fname, self.mesh)
            ad.write_function(fname, self.hef.m, time=0.0, name="m")

        if return_stats:
            return y, self.ctx, elapsed, stats
        return y, self.ctx, elapsed








    def hysteresis(
        self,
        m0_array,
        H_steps,                
        t_final_per_step,
        dt_init,
        output_dir="hyst_out",
        ts_rtol=1e-6,
        ts_atol=1e-6,
        snes_rtol=1e-2,
        snes_atol=1e-4,
        ksp_rtol=1e-4,
        stopping_dmdt=0.0,
        check_every_stop=5,
        stop_print=False,
        xdmf_name="Hysteresis.xdmf",          
        log_name="hysteresis_log.txt",
        write_xdmf_series=False,             
        write_xdmf_per_step=True,          
        write_bp_series=True,               
        bp_name="Hysteresis.bp",
    ):
        H_steps = np.asarray(list(H_steps), dtype=float).reshape((-1, 3))
        comm = self.mesh.comm


        self._ensure_solver(
            m0_array=m0_array,
            dt_init=dt_init,
            ts_rtol=ts_rtol, ts_atol=ts_atol,
            snes_rtol=snes_rtol, snes_atol=snes_atol,
            ksp_rtol=ksp_rtol,
            stopping_dmdt=stopping_dmdt,
            check_every_stop=check_every_stop,
            stop_print=stop_print,
        )

        ts = self.ts
        hef = self.hef
        y = self.y

    
        self._cancel_ts_monitors()

        if comm.rank == 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        comm.barrier()


        log_path = Path(output_dir) / log_name
        if comm.rank == 0:
            with open(log_path, "w") as f:
                f.write("# step Hx Hy Hz  <mx> <my> <mz>  maxdmdt(deg/ns)  reason nsteps dt_last\n")


        xdmf = None
        if write_xdmf_series:
            xdmf_path = Path(output_dir) / xdmf_name
            xdmf = io.XDMFFile(comm, str(xdmf_path), "w")
            xdmf.write_mesh(self.mesh)

        bp_path = Path(output_dir) / bp_name
        if write_bp_series:

            ad.write_mesh(bp_path, self.mesh)

        def global_mean_m():

            mloc = hef.m.x.petsc_vec.getArray(readonly=True).reshape((-1, 3))
            s_loc = mloc.sum(axis=0) if mloc.size else np.zeros(3)
            n_loc = mloc.shape[0]
            s_glob = np.array(comm.allreduce(s_loc, op=MPI.SUM), dtype=float)
            n_glob = comm.allreduce(n_loc, op=MPI.SUM)
            return s_glob / float(n_glob)

        results = []

        for i, (Hx, Hy, Hz) in enumerate(H_steps):

            hef.set_uniform_field(Hx, Hy, Hz)
            self._reset_ts_run(t0=0.0, t_final=t_final_per_step, dt_init=dt_init)
            elapsed, stats = self._run_ts()


            if xdmf is not None:
                xdmf.write_function(hef.m, float(i))


            if write_xdmf_per_step:
                fname = Path(output_dir) / f"m_{i:05d}.xdmf"
                with io.XDMFFile(comm, str(fname), "w") as xf:
                    xf.write_mesh(self.mesh)
                    xf.write_function(hef.m)


            if write_bp_series:
                ad.write_function(bp_path, hef.m, time=float(i), name="m")



            mmean = global_mean_m()
            maxdmdt = float(self.stopper.last_max_dmdt_deg_ns) if self.stopper is not None else 0.0
            if not np.isfinite(maxdmdt):
                maxdmdt = 0.0

            entry = {
                "step": int(i),
                "H": (float(Hx), float(Hy), float(Hz)),
                "m_mean": (float(mmean[0]), float(mmean[1]), float(mmean[2])),
                "elapsed": float(elapsed),
                **stats,
            }
            results.append(entry)

            if comm.rank == 0:
                with open(log_path, "a") as f:
                    f.write(
                        f"{i:d} {Hx:.6e} {Hy:.6e} {Hz:.6e} "
                        f"{mmean[0]:.6e} {mmean[1]:.6e} {mmean[2]:.6e} "
                        f"{maxdmdt:.6e} {stats['reason']:d} {stats['nsteps']:d} {stats['dt_last']:.6e}\n"
                    )

            if comm.rank == 0:

                print(
                    f"[HYST] i={i:05d}  H(mT)=({Hx*4*np.pi*1e-4:+.6e},{Hy*4*np.pi*1e-4:+.6e},{Hz*4*np.pi*1e-4:+.6e})  "
                    f"<m>=({mmean[0]:+.6e},{mmean[1]:+.6e},{mmean[2]:+.6e})  "
                    f"max|dm/dt|={maxdmdt:.6e} deg/ns  "
                    f"nsteps={stats['nsteps']}  t_end={stats['t_end']*1e9:.6f} ns",
                    flush=True
                )

        if xdmf is not None:
            xdmf.close()

        return results
