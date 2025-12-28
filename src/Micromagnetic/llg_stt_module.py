# llg_stt_module.py

import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import fem, io
from dolfinx.fem import Function, Constant, form
from dolfinx.fem.petsc import assemble_vector

import adios4dolfinx as ad

from .Exchange import ExchangeField
from .Demag import DemagField
from .Anisotropy import AnisotropyField
from .DMI_Bulk import DMIBULK
from .DMI_Interfacial import DMIInterfacial
from .Zhang_Li import ZhangLi

# ---------------------------------------------------------
#  CLASE EffectiveFieldSTT (LLG + STT Zhang-Li)
# ---------------------------------------------------------
class EffectiveFieldSTT:
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
        H0_vec,
        gamma=2.211e5,
        alpha=0.5,
        do_precess=1,
        Jmagnitude=0.0,
        Jdir_vec=None,
        P=0.0,
        beta=0.0,
        use_demag=True, 
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

        self.n_ani = fem.Function(self.V)
        self.n_ani.x.array[:] = n_ani_vec

        self.n0_int = fem.Function(self.V)
        self.n0_int.x.array[:] = n0_int_vec

        self.H0 = H0_vec.copy()

        self.comm = self.mesh.comm
        self.m = fem.Function(self.V)
        self.H_eff = Function(self.V)

        self.prefactor = -self.gamma / (1.0 + self.alpha**2)
        self.Stab = self.Ms * self.gamma / (1 + self.alpha**2)*0.5

        self.P = P
        self.Jmagnitude = Jmagnitude
        self.beta = beta
        self.Jdir_vec = Jdir_vec

        e = 1.6021766e-19
        muB = 9.27400915e-24

        self.prefZhang = (
            self.Jmagnitude
            * self.P
            * muB
            / (e * self.Ms * (1.0 + self.beta**2))
            * 1.0
            / (1.0 + self.alpha**2)
            / 1e-9
        )

        self.n_nodes_local = len(self.mesh.geometry.x)

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

        # STT arrays
        self.Zcx = np.zeros(self.n_nodes_local)
        self.Zcy = np.zeros(self.n_nodes_local)
        self.Zcz = np.zeros(self.n_nodes_local)

        self.ZZcx = np.zeros(self.n_nodes_local)
        self.ZZcy = np.zeros(self.n_nodes_local)
        self.ZZcz = np.zeros(self.n_nodes_local)

        self.Zhang_x = np.zeros(self.n_nodes_local)
        self.Zhang_y = np.zeros(self.n_nodes_local)
        self.Zhang_z = np.zeros(self.n_nodes_local)

        self.zhang_le = np.zeros(3 * self.n_nodes_local)


        v = ufl.TestFunction(self.V)
        paso1 = ufl.dot(
            v, Constant(self.mesh, PETSc.ScalarType((1.0, 1.0, 1.0)))
        ) * ufl.dx

        volNodos_f = fem.Function(self.V)
        volNodos_f.x.petsc_vec.set(0.0)
        assemble_vector(volNodos_f.x.petsc_vec, form(paso1))
        volNodos_f.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES,
            mode=PETSc.ScatterMode.REVERSE,
        )
        volNodos_f.x.scatter_forward()
        volNodos = volNodos_f.x.array
        self.volNodos = volNodos

        self.Hvec = np.zeros(3 * self.n_nodes_local)
        self.dmdt = Function(self.V)


        self.demag_field = None             
        if self.use_demag:                  
            self.demag_field = DemagField(self.mesh, self.V, self.V1, self.Ms)
            self.demag_field.compute(self.m)

        self.exchange_field = ExchangeField(
            self.mesh, self.V, self.A, self.Ms, volNodos
        )

        self.anisotropy_field = AnisotropyField(
            self.mesh, self.V, self.Ku, self.Ms, n_ani_vec, volNodos
        )

        self.DMIBULK = DMIBULK(
            self.mesh, self.V, self.V1, self.D_bulk, self.Ms, volNodos
        )


        self.DMI_int = None
        if abs(self.D_int) > 0.0:
            self.DMI_int = DMIInterfacial(
                self.mesh,
                self.V,
                self.V1,
                self.D_int,
                n0_int_vec,
                self.Ms,
                volNodos,
            )


        self.ZhangLi = ZhangLi(self.mesh, self.V, self.Jdir_vec, volNodos)

        self.pasosJac = 0
        self.pasosLLG = 0


        self.K_total = self.exchange_field.K + self.anisotropy_field.K + self.DMIBULK.K
        if self.DMI_int is not None:
            self.K_total = self.K_total + self.DMI_int.K

        self.m_jac = Function(self.V)
        self.v_jac = Function(self.V)

        self.start, self.end = self.V.dofmap.index_map.local_range

        self.H_m = Function(self.V)
        self.H_v = Function(self.V)

        owned_dofs = self.end - self.start

        self.Jv_buffer = np.zeros(3 * owned_dofs, dtype=np.float64)

        self.M_cached = np.zeros((owned_dofs, 3))
        self.Hm_cached = np.zeros((owned_dofs, 3))
        self.JdotGrad_m_cache = np.zeros((owned_dofs, 3))

        self.local_dofs = owned_dofs
        self.local_size = 3 * self.local_dofs


    def compute_H_eff(self, m):
  
        H = self.exchange_field.compute(m).x.array

        if self.demag_field is not None:  
            H += self.demag_field.compute(m).x.array

        if abs(self.Ku) > 0.0:
            H += self.anisotropy_field.compute(m).x.array

        if abs(self.D_bulk) > 0.0:
            H += self.DMIBULK.compute(m).x.array

        if self.DMI_int is not None and abs(self.D_int) > 0.0:
            H += self.DMI_int.compute(m).x.array

        H += self.H0

        self.H_eff.x.array[:] = H
        self.H_eff.x.scatter_forward()
        return self.H_eff.x.array

 
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

        return E_exch + E_demag + E_ani + E_dmi_bulk + E_dmi_int


    def update_jac_state(self, m_vec):

        self.m_jac.x.array[:self.local_size] = m_vec
        self.m_jac.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT_VALUES,
            mode=PETSc.ScatterMode.FORWARD,
        )


        self.H_m.x.petsc_vec.set(0.0)
        self.K_total.mult(self.m_jac.x.petsc_vec, self.H_m.x.petsc_vec)
        self.H_m.x.scatter_forward()

        M_loc = m_vec.reshape(-1, 3)
        Hm_loc = self.H_m.x.array[:self.local_size].reshape(-1, 3)

        self.M_cached[:, :] = M_loc
        self.Hm_cached[:, :] = Hm_loc


        Jgm = self.ZhangLi.compute(self.m_jac).x.petsc_vec.getArray(
            readonly=True
        ).reshape(-1, 3)
        self.JdotGrad_m_cache[:, :] = Jgm

    def jac_vec_times_STT(self, m_unused, v, out):


        self.pasosJac += 1

        self.H_v.x.petsc_vec.set(0.0)
        self.v_jac.x.array[:self.local_size] = v
        self.v_jac.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT_VALUES,
            mode=PETSc.ScatterMode.FORWARD,
        )

        self.K_total.mult(self.v_jac.x.petsc_vec, self.H_v.x.petsc_vec)
        self.H_v.x.scatter_forward()

        JdotGrad_v = self.ZhangLi.compute(self.v_jac).x.petsc_vec.getArray(
            readonly=True
        ).reshape(-1, 3)

        M = self.M_cached
        Hm = self.Hm_cached
        JdotGrad_m = self.JdotGrad_m_cache

        V = v.reshape(-1, 3)
        Hv = self.H_v.x.array[:self.local_size].reshape(-1, 3)

        cross_vHm = np.cross(V, Hm)
        cross_mHv = np.cross(M, Hv)
        prec = self.do_precess * (cross_vHm + cross_mHv)

        mdHm = np.sum(M * Hm, axis=1)[:, None]
        mdHv = np.sum(M * Hv, axis=1)[:, None]
        vdHm = np.sum(V * Hm, axis=1)[:, None]
        mdv = np.sum(M * V, axis=1)[:, None]
        mdmm = np.sum(M * M, axis=1)[:, None]

        MjgradV = np.sum(M * JdotGrad_v, axis=1)[:, None]
        VjgradM = np.sum(V * JdotGrad_m, axis=1)[:, None]
        MjgradM = np.sum(M * JdotGrad_m, axis=1)[:, None]

        prec_STT = -(self.beta - self.alpha) * self.prefZhang * (
            np.cross(M, JdotGrad_v) + np.cross(V, JdotGrad_m)
        )

        Damp_STT = -(1.0 + self.beta * self.alpha) * self.prefZhang * (
            V * MjgradM
            + M * VjgradM
            + M * MjgradV
            - JdotGrad_v * mdmm
            - 2.0 * JdotGrad_m * mdv
        )

        term1 = V * mdHm
        term2 = -2.0 * Hm * mdv
        term3 = M * (vdHm + mdHv)
        term4 = -Hv * mdmm
        damp = term1 + term2 + term3 + term4

        coef = -self.gamma / (1.0 + self.alpha**2)
        Jv = coef * (prec + self.alpha * damp) + prec_STT + Damp_STT

        extra = self.Stab * (V * (1.0 - mdmm) - 2.0 * M * mdv)
        Jv += extra

        out[:] = Jv.reshape(-1)


    def llg_rhs_STT(self, m):
        self.pasosLLG += 1

        m_numeric = m.x.array
        self.Hfield[:] = self.compute_H_eff(m)
        self.zhang_le[:] = self.ZhangLi.compute(m).x.array

        self.mx[:] = m_numeric[0::3]
        self.my[:] = m_numeric[1::3]
        self.mz[:] = m_numeric[2::3]

        self.norma[:] = (
            self.mx[:] * self.mx[:]
            + self.my[:] * self.my[:]
            + self.mz[:] * self.mz[:]
        )

        self.Zhang_x[:] = self.zhang_le[0::3]
        self.Zhang_y[:] = self.zhang_le[1::3]
        self.Zhang_z[:] = self.zhang_le[2::3]

        self.Zcx[:] = self.my * self.Zhang_z - self.mz * self.Zhang_y
        self.Zcy[:] = self.mz * self.Zhang_x - self.mx * self.Zhang_z
        self.Zcz[:] = self.mx * self.Zhang_y - self.my * self.Zhang_x

        self.ZZcx[:] = self.my * self.Zcz - self.mz * self.Zcy
        self.ZZcy[:] = self.mz * self.Zcx - self.mx * self.Zcz
        self.ZZcz[:] = self.mx * self.Zcy - self.my * self.Zcx

        self.Hx[:] = self.Hfield[0::3]
        self.Hy[:] = self.Hfield[1::3]
        self.Hz[:] = self.Hfield[2::3]

        self.mcx[:] = self.my * self.Hz - self.mz * self.Hy
        self.mcy[:] = self.mz * self.Hx - self.mx * self.Hz
        self.mcz[:] = self.mx * self.Hy - self.my * self.Hx

        self.mcmx[:] = self.my * self.mcz - self.mz * self.mcy
        self.mcmy[:] = self.mz * self.mcx - self.mx * self.mcz
        self.mcmz[:] = self.mx * self.mcy - self.my * self.mcx

        self.dmdt.x.array[0::3] = (
            self.prefactor * (self.do_precess * self.mcx + self.alpha * self.mcmx)
            - self.prefZhang
            * (
                (self.beta - self.alpha) * self.Zcx
                + (1.0 + self.alpha * self.beta) * self.ZZcx
            ) + self.Stab*(1.0 - self.norma) * self.mx[:]
        )

        self.dmdt.x.array[1::3] = (
            self.prefactor * (self.do_precess * self.mcy + self.alpha * self.mcmy)
            - self.prefZhang
            * (
                (self.beta - self.alpha) * self.Zcy
                + (1.0 + self.alpha * self.beta) * self.ZZcy
            ) + self.Stab*(1.0 - self.norma) * self.my[:]
        )

        self.dmdt.x.array[2::3] = (
            self.prefactor * (self.do_precess * self.mcz + self.alpha * self.mcmz)
            - self.prefZhang
            * (
                (self.beta - self.alpha) * self.Zcz
                + (1.0 + self.alpha * self.beta) * self.ZZcz
            ) + self.Stab*(1.0 - self.norma) * self.mz[:]
        )

        return self.dmdt

    def ifunction_STT(self, ts, t, y, ydot, f):
        self.pasosLLG += 1

        y.copy(self.m.x.petsc_vec)
        self.m.x.scatter_forward()

        dmdt = self.llg_rhs_STT(self.m)
        dmdt.x.scatter_forward()

        f.waxpy(-1.0, dmdt.x.petsc_vec, ydot)
        return 0



class LLG_STT:
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


        self._Jmag = 0.0
        self._Jdir_vec = None
        self._P = 0.0
        self._beta = 0.0

        self._has_exchange = False
        self._has_demag = False  
        self._has_anisotropy = False
        self._has_dmi_bulk = False
        self._has_dmi_int = False
        self._has_H0 = False
        self._has_current = False

        self.hef: EffectiveFieldSTT | None = None


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

    def add_external_field(self, H0_vec):

        self._H0_vec = H0_vec
        self._has_H0 = True

    def add_current(self, Jmagnitude, Jdir_vec, P, beta):

        self._Jmag = Jmagnitude
        self._Jdir_vec = Jdir_vec
        self._P = P
        self._beta = beta
        self._has_current = True


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

        if self._has_H0 and self._H0_vec is not None:
            H0_vec = self._H0_vec
        else:
            H0_vec = np.zeros(3 * len(self.mesh.geometry.x), dtype=np.float64)

        if self._has_current and self._Jdir_vec is not None:
            Jmag = self._Jmag
            Jdir_vec = self._Jdir_vec
            P = self._P
            beta = self._beta
        else:
            Jmag = 0.0
            Jdir_vec = np.zeros(3 * len(self.mesh.geometry.x), dtype=np.float64)
            P = 0.0
            beta = 0.0

        self.hef = EffectiveFieldSTT(
            self.mesh,
            self.Ms,
            Aex,
            Ku,
            n_ani_vec,
            D_bulk,
            D_int,
            n0_int_vec,
            H0_vec,
            gamma=self.gamma,
            alpha=self.alpha,
            do_precess=self.do_precess,
            Jmagnitude=Jmag,
            Jdir_vec=Jdir_vec,
            P=P,
            beta=beta,
            use_demag=self._has_demag,
        )

    def solve(
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
        monitor_fn=None,
    ):
        

        if self.hef is None:
            self._build_effective_field()
        hef = self.hef

        # initial state
        hef.m.x.array[:] = m0_array
        hef.m.x.scatter_forward()

        ts = PETSc.TS().create(self.mesh.comm)

        # Opciones TS/SNES/KSP
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
        opts["ts_max_steps"] = 5000000
        opts["snes_rtol"] = snes_rtol
        opts["snes_atol"] = snes_atol
        opts["snes_max_it"] = 8
        opts["ksp_type"] = "gmres"
        opts["ksp_rtol"] = ksp_rtol
        opts["ts_max_snes_failures"] = -1

        ts.setTime(t0)
        ts.setTimeStep(dt_init)
        ts.setMaxTime(t_final)
        ts.setExactFinalTime(PETSc.TS.ExactFinalTime.STEPOVER)

        dm = PETSc.DMShell().create(comm=self.mesh.comm)
        dm.setGlobalVector(hef.m.x.petsc_vec.copy())
        ts.setDM(dm)

        snes = ts.getSNES()

        n_loc = hef.m.x.petsc_vec.getLocalSize()
        n_glob = hef.m.x.petsc_vec.getSize()


        class JvContext:
            def __init__(self, hef_):
                self.hef = hef_
                self.shift = 0.0
                self.calls = 0
                self.callsPre = 0

                diag = self.hef.K_total.getDiagonal()
                self.inv_diag = diag.copy()
                self.inv_diag.reciprocal()



                diag = self.hef.K_total.getDiagonal()
                self.diagK = diag.getArray(readonly=True).copy()

                self.gamma = float(self.hef.gamma)
                self.do_precess = float(self.hef.do_precess)

                self.enable_pc = True

                # PC buffers 
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

                self.beta = float(self.hef.beta)
                self.alpha = float(self.hef.alpha)
                self.prefZ = float(self.hef.prefZhang)
                self.c3 = (self.beta - self.alpha) * self.prefZ
                self.c4 = -(1.0 + self.alpha*self.beta) * self.prefZ

            def update_pc_full_fast(self, shift, include_stab=True, use_abs_kappa=True,
                                    eps_reg=1e-14, det_eps=1e-30):
                self.shift = float(shift)

                M = self.hef.M_cached  
                H = self.hef.Hm_cached 
                mx, my, mz = M[:,0], M[:,1], M[:,2]
                hx, hy, hz = H[:,0], H[:,1], H[:,2]

                kappa = self.kappa_abs if use_abs_kappa else self.kappa_sgn

                c1 = self.c1
                c2 = self.c2

                G = self.hef.JdotGrad_m_cache  
                gx, gy, gz = G[:,0], G[:,1], G[:,2]

                c3 = self.c3
                c4 = self.c4



                Stab = self.Stab

                mdH = mx*hx + my*hy + mz*hz
                mdm = mx*mx + my*my + mz*mz

                # S_H =
                # [ 0  -hz  hy
                #   hz  0  -hx
                #  -hy  hx  0 ]

                Jp00 = 0.0
                Jp01 = -c1*(-hz - kappa*(-mz))   # -c1*(S_H01 - kappa*S_m01)
                Jp02 = -c1*( hy - kappa*( my))
                Jp10 = -c1*( hz - kappa*( mz))
                Jp11 = 0.0
                Jp12 = -c1*(-hx - kappa*(-mx))
                Jp20 = -c1*(-hy - kappa*(-my))
                Jp21 = -c1*( hx - kappa*( mx))
                Jp22 = 0.0

                # Parte damping local: Jd = c2*(B + kappa*C)
                # B_ij = (m. H) delta_ij + m_i H_j - 2 H_i m_j
                # C_ij = m_i m_j - (m.m) delta_ij

                B00 = mdH + mx*hx - 2*hx*mx  # = mdH - hx*mx
                B11 = mdH - hy*my
                B22 = mdH - hz*mz
                C00 = mx*mx - mdm
                C11 = my*my - mdm
                C22 = mz*mz - mdm


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
                    Js00 = Stab*(s0 - 2*mx*mx)
                    Js11 = Stab*(s0 - 2*my*my)
                    Js22 = Stab*(s0 - 2*mz*mz)
                    Js01 = Stab*(-2*mx*my); Js02 = Stab*(-2*mx*mz)
                    Js10 = Stab*(-2*my*mx); Js12 = Stab*(-2*my*mz)
                    Js20 = Stab*(-2*mz*mx); Js21 = Stab*(-2*mz*my)
                else:
                    Js00=Js11=Js22=0.0
                    Js01=Js02=Js10=Js12=Js20=Js21=0.0


                J00 = (0.0)     + Jd00 + Js00
                J11 = (0.0)     + Jd11 + Js11
                J22 = (0.0)     + Jd22 + Js22

                J01 = Jp01 + Jd01 + Js01
                J02 = Jp02 + Jd02 + Js02
                J10 = Jp10 + Jd10 + Js10
                J12 = Jp12 + Jd12 + Js12
                J20 = Jp20 + Jd20 + Js20
                J21 = Jp21 + Jd21 + Js21

                Jstt01 = c3 * (-gz)   
                Jstt02 = c3 * ( gy)
                Jstt10 = c3 * ( gz)
                Jstt12 = c3 * (-gx)
                Jstt20 = c3 * (-gy)
                Jstt21 = c3 * ( gx)


                mdG = mx*gx + my*gy + mz*gz

                Bg00 = mdG - gx*mx
                Bg11 = mdG - gy*my
                Bg22 = mdG - gz*mz

                Bg01 = mx*gy - 2*gx*my
                Bg02 = mx*gz - 2*gx*mz
                Bg10 = my*gx - 2*gy*mx
                Bg12 = my*gz - 2*gy*mz
                Bg20 = mz*gx - 2*gz*mx
                Bg21 = mz*gy - 2*gz*my

                J00 += c4*Bg00;  J11 += c4*Bg11;  J22 += c4*Bg22
                J01 += Jstt01 + c4*Bg01
                J02 += Jstt02 + c4*Bg02
                J10 += Jstt10 + c4*Bg10
                J12 += Jstt12 + c4*Bg12
                J20 += Jstt20 + c4*Bg20
                J21 += Jstt21 + c4*Bg21



                s = self.shift + eps_reg
                A00 = self.A00; A01 = self.A01; A02 = self.A02
                A10 = self.A10; A11 = self.A11; A12 = self.A12
                A20 = self.A20; A21 = self.A21; A22 = self.A22

                A00[:] = s - J00;  A01[:] =   - J01;  A02[:] =   - J02
                A10[:] =   - J10;  A11[:] = s - J11;  A12[:] =   - J12
                A20[:] =   - J20;  A21[:] =   - J21;  A22[:] = s - J22


                m00 = A11*A22 - A12*A21
                m01 = A10*A22 - A12*A20
                m02 = A10*A21 - A11*A20
                det = A00*m00 - A01*m01 + A02*m02

                #det_abs = np.abs(det)
                #det = np.where(det_abs < det_eps, det + np.sign(det + det_eps)*det_eps, det)
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


            def mult(self, A, x, y):
                self.calls += 1
                xv = x.getArray(readonly=True)
                yv = y.getArray()

                m_vec = self.hef.m.x.petsc_vec.getArray(readonly=True)
                self.hef.jac_vec_times_STT(m_vec, xv, out=self.hef.Jv_buffer)

                yv[:] = self.shift * xv - self.hef.Jv_buffer











        J = PETSc.Mat().create(comm=self.mesh.comm)
        ctx = JvContext(hef)
        J.setSizes([[n_loc, n_glob], [n_loc, n_glob]])
        J.setType("python")
        J.setPythonContext(ctx)
        J.setUp()

        ksp = snes.getKSP()
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.PYTHON)
        pc.setPythonContext(ctx)



        def IJac(ts_, t, y, ydot, shift, A, B):
            y.copy(hef.m.x.petsc_vec)
            hef.m.x.scatter_forward()

            mloc = hef.m.x.petsc_vec.getArray(readonly=True)
            hef.update_jac_state(mloc)

            ctx.update_pc_full_fast(shift, include_stab=True, use_abs_kappa=True)

            return PETSc.Mat.Structure.SAME_NONZERO_PATTERN


        ts.setIFunction(hef.ifunction_STT)
        ts.setIJacobian(IJac, J)

        ts.setFromOptions()

        y = hef.m.x.petsc_vec.copy()
        y.ghostUpdate(
            addv=PETSc.InsertMode.INSERT_VALUES,
            mode=PETSc.ScatterMode.FORWARD,
        )

        # ----------------- Monitor -----------------
        if dt_save is not None:
            if dt_snap is None:
                dt_snap = dt_save

            if self.mesh.comm.rank == 0:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            log_path = Path(output_dir) / "log.txt"

            last_save_n = {"n": -1}
            last_snap_n = {"n": -1}
            snap_counter = {"k": 0}
            first_print = {"done": False}


            def renormalize_m(u_vec, hef_):

                u_vec.copy(hef_.m.x.petsc_vec)
                hef_.m.x.scatter_forward()

                arr = hef_.m.x.array 
                mx = arr[0::3]
                my = arr[1::3]
                mz = arr[2::3]

                norma = np.sqrt(mx * mx + my * my + mz * mz)
                mask = norma > 0.0
                mx[mask] /= norma[mask]
                my[mask] /= norma[mask]
                mz[mask] /= norma[mask]

                hef_.m.x.scatter_forward()




            def default_monitor(ts_, step, t, u, hef_, mesh_):

                Exch = hef_.exchange_field.Energy(hef_.m)
                Demag = 0.0                      
                if hef_.demag_field is not None:  
                    Demag = hef_.demag_field.Energy(hef_.m)

                Ani = 0.0
                if getattr(hef_, "Ku", 0.0) != 0.0:
                    Ani = hef_.anisotropy_field.Energy(hef_.m)

                DMI_bulk = 0.0
                if getattr(hef_, "D_bulk", 0.0) != 0.0:
                    DMI_bulk = hef_.DMIBULK.Energy(hef_.m)

                DMI_int = 0.0
                if getattr(hef_, "D_int", 0.0) != 0.0 and hef_.DMI_int is not None:
                    DMI_int = hef_.DMI_int.Energy(hef_.m)

                Exch_total = mesh_.comm.gather(Exch, root=0)
                Demag_total = mesh_.comm.gather(Demag, root=0)
                Ani_total = mesh_.comm.gather(Ani, root=0)
                DMI_bulk_total = mesh_.comm.gather(DMI_bulk, root=0)
                DMI_int_total = mesh_.comm.gather(DMI_int, root=0)

                mag = mesh_.comm.gather(
                    hef_.m.x.petsc_vec.getArray(readonly=True), root=0
                )


                torque_norm = np.sqrt(
                    hef_.mcx**2 + hef_.mcy**2 + hef_.mcz**2
                )
                max_torque_local = np.max(torque_norm)
                max_torque_all = mesh_.comm.gather(max_torque_local, root=0)

                # snapshots
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

                    E_exch = np.sum(Exch_total)
                    E_demag = np.sum(Demag_total)
                    E_ani = np.sum(Ani_total)
                    E_db = np.sum(DMI_bulk_total)
                    E_di = np.sum(DMI_int_total)
                    E_tot = E_exch + E_demag + E_ani + E_db + E_di

                    maxtorque = 4 * np.pi * 1e-7 * max(max_torque_all)

                    if not first_print["done"]:
                        header = (
                            f"{'time':>10} {'<mx>':>15} {'<my>':>15} {'<mz>':>15} "
                            f"{'maxtorque':>15} "
                            f"{'E_demag':>15} {'E_exch':>15} {'E_ani':>15} "
                            f"{'E_dmi_bulk':>15} {'E_dmi_int':>15} {'E_total':>15}"
                        )
                        print(header)
                        with open(log_path, "w") as f:
                            f.write(header + "\n")
                        first_print["done"] = True

                    line = (
                        f"{t*1e9:10.4f} "
                        f"{mag[:,0].mean():15.6f} {mag[:,1].mean():15.6f} {mag[:,2].mean():15.6f} "
                        f"{maxtorque:15.4e} "
                        f"{E_demag:15.4e} {E_exch:15.4e} {E_ani:15.4e} "
                        f"{E_db:15.4e} {E_di:15.4e} {E_tot:15.4e}"
                    )
                    print(line)
                    with open(log_path, "a") as f:
                        f.write(line + "\n")
                    sys.stdout.flush()

            def monitor(ts_, step, t, u):
                n = int(np.trunc(t / dt_save))

                #if step % 2 == 0:
                #renormalize_m(u, hef)

                if n != last_save_n["n"]:
                    last_save_n["n"] = n
                    if monitor_fn is not None:
                        monitor_fn(ts_, step, t, u, hef, self.mesh)
                    else:
                        default_monitor(ts_, step, t, u, hef, self.mesh)

            ts.setMonitor(monitor)

        ts.setSolution(y)


        tstart = perf_counter()
        ts.solve(y)
        elapsed = perf_counter() - tstart


        comm = self.mesh.comm
        #mag_all = comm.gather(y.array, root=0)
        #Pre_calls = comm.gather(ctx.callsPre, root=0)
        #Jac_calls = comm.gather(ctx.calls, root=0)
        #LLG_calls = comm.gather(hef.pasosLLG, root=0)
        #sizecores = comm.Get_size()

        if comm.rank == 0:
            print(f"\n  ts.solve : {elapsed:.3f} s")
            #print("jac calls", Jac_calls)
            #print("prec calls", Pre_calls)
            #print("LLG calls", LLG_calls)




        return y, ctx, elapsed