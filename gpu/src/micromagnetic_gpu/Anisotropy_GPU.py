import ufl
import numpy as np
from dolfinx import fem
from petsc4py import PETSc
from dolfinx.fem.petsc import assemble_matrix

"""
GPU uniaxial anisotropy field for finite-element micromagnetics.

The anisotropy matrix is assembled with DOLFINx, scaled by the lumped nodal
volume, and converted to a PETSc AIJCUSPARSE matrix for CUDA execution.

Conventions
-----------
- Mesh coordinates are assumed to be in m.
- compute(m_gpu) expects a PETSc.Vec CUDA vector.
- Energy(m_fun) expects a dolfinx.fem.Function on the host.
"""



class AnisotropyField:

    """
    Uniaxial anisotropy effective field.

    The effective field is computed as

        H_ani = 2 Ku/(mu0 Ms) * (m cdot n) n,

    using a lumped-mass finite-element discretization. 

    AniVec : Easy axis anisotropy (n)
    Ku     : Anisotropy constant J/m^3.
    Ms     : Saturation magnetization A/m.
    VolN   : Nodal volume nm^3.
    
    """


    def __init__(self, mesh, V, Ku, Ms, AniVec, VolN):

        self.mesh = mesh
        self.V = V
        self.Ku = float(Ku)
        self.M_s = float(Ms)
        self.mu_0 = 4.0 * np.pi * 1e-7

        self.n = fem.Function(self.V)
        self.n.x.array[:] = AniVec[:]
        self.n.x.scatter_forward()

        v = ufl.TestFunction(self.V)
        u = ufl.TrialFunction(self.V)

        a = ufl.dot(u, self.n) * ufl.dot(v, self.n) * ufl.dx

        K_cpu = assemble_matrix(fem.form(a))
        K_cpu.assemble()

        prefactor = fem.Function(self.V)
        prefactor.x.array[:] = (
            2.0 * self.Ku / (self.mu_0 * self.M_s) / VolN[:]
        )
        prefactor.x.scatter_forward()

        K_cpu.diagonalScale(prefactor.x.petsc_vec, None)

        self.K = K_cpu.convert(PETSc.Mat.Type.AIJCUSPARSE)
        self.K.bindToCPU(False)

        self.H_anis = fem.Function(self.V)

        self.h_gpu = self.K.createVecLeft()
        self.h_gpu.setType(PETSc.Vec.Type.CUDA)
        self.h_gpu.bindToCPU(False)

    def compute(self, m_gpu):
        """
        m_gpu should be PETSc.Vec CUDA.
        """
        self.K.mult(m_gpu, self.h_gpu)
        self.h_gpu.copy(self.H_anis.x.petsc_vec)

        return self.H_anis

    def Energy(self, m):
        """
        m should be fem.Function CPU/host.
        """
        E_int = ufl.dot(m, self.n) * ufl.dot(m, self.n) * ufl.dx
        energy = -self.Ku * fem.assemble_scalar(fem.form(E_int))

        return float(energy * 1e-27)




if __name__ == "__main__":

    from time import perf_counter

    import numpy as np
    import dolfinx
    import ufl

    from mpi4py import MPI
    from petsc4py import PETSc
    from dolfinx import fem
    from dolfinx.fem import Constant, form
    from dolfinx.fem.petsc import assemble_vector

    comm = MPI.COMM_WORLD

    if comm.size != 1:
        raise RuntimeError(
            "This GPU test is intended to run with a single MPI rank. "
            "Run: python Anisotropy_GPU.py"
        )

    # ------------------------------------------------------------
    # Mesh and function space
    # ------------------------------------------------------------
    Nx, Ny, Nz = 40, 40, 10
    L, B, H = 80.0, 80.0, 10.0

    mesh = dolfinx.mesh.create_box(
        comm,
        [
            np.array([-L / 2, -B / 2, -H / 2]),
            np.array([ L / 2,  B / 2,  H / 2]),
        ],
        [Nx, Ny, Nz],
    )

    V = fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))

    # ------------------------------------------------------------
    # Lumped nodal volumes
    # ------------------------------------------------------------
    v = ufl.TestFunction(V)
    vol_form = (
        ufl.dot(v, Constant(mesh, PETSc.ScalarType((1.0, 1.0, 1.0))))
        * ufl.dx
    )

    volN_f = fem.Function(V)
    volN_f.x.petsc_vec.set(0.0)

    assemble_vector(volN_f.x.petsc_vec, form(vol_form))
    volN_f.x.petsc_vec.ghostUpdate(
        addv=PETSc.InsertMode.ADD_VALUES,
        mode=PETSc.ScatterMode.REVERSE,
    )
    volN_f.x.scatter_forward()

    VolN = volN_f.x.array.copy()

    # ------------------------------------------------------------
    # Initial magnetization on host
    # ------------------------------------------------------------
    m = fem.Function(V)

    def m_init(x):
        out = np.zeros((3, x.shape[1]), dtype=np.float64)

        out[0] = 1.0 / np.sqrt(2.0)
        out[1] = 1.0 / np.sqrt(2.0)
        out[2] = 0.0

        return out

    m.interpolate(m_init)
    m.x.scatter_forward()

    # ------------------------------------------------------------
    # Easy-axis field on host
    # ------------------------------------------------------------
    n_fun = fem.Function(V)

    def n_init(x):
        out = np.zeros((3, x.shape[1]), dtype=np.float64)

        out[0] = 1.0 / np.sqrt(2.0)
        out[1] = 1.0 / np.sqrt(2.0)
        out[2] = 0.0

        return out

    n_fun.interpolate(n_init)
    n_fun.x.scatter_forward()

    AniVec = n_fun.x.array.copy()

    # ------------------------------------------------------------
    # Copy host magnetization to PETSc CUDA vector
    # ------------------------------------------------------------
    m_gpu = m.x.petsc_vec.duplicate()
    m_gpu.setType(PETSc.Vec.Type.CUDA)
    m_gpu.bindToCPU(False)
    m.x.petsc_vec.copy(m_gpu)

    # ------------------------------------------------------------
    # Anisotropy field
    # ------------------------------------------------------------
    anisotropy = AnisotropyField(
        mesh=mesh,
        V=V,
        Ku=1.0e5,
        Ms=8.6e5,
        AniVec=AniVec,
        VolN=VolN,
    )

    H_fun = anisotropy.compute(m_gpu)
    H_fun.x.scatter_forward()

    E = anisotropy.Energy(m)

    # ------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------
    nrep = 100

    t0 = perf_counter()
    for _ in range(nrep):
        H_fun = anisotropy.compute(m_gpu)

    _ = anisotropy.h_gpu.norm()
    elapsed = perf_counter() - t0

    H_fun.x.scatter_forward()

    # ------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------
    imap = V.dofmap.index_map
    bs = V.dofmap.index_map_bs
    owned = bs * imap.size_local

    H_owned = H_fun.x.array[:owned].reshape((-1, 3))

    if comm.rank == 0:
        print("==== AnisotropyField GPU test ====")
        print(f"Anisotropy energy: {E:.12e} J")
        print(f"Total time for {nrep} compute calls: {elapsed:.6e} s")
        print(f"Average compute time: {elapsed / nrep:.6e} s")
        print("Field averages:", H_owned.mean(axis=0))
        print("Field minima:  ", H_owned.min(axis=0))
        print("Field maxima:  ", H_owned.max(axis=0))
