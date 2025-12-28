import ufl
import numpy as np
from dolfinx import fem
from petsc4py import PETSc
from dolfinx.fem.petsc import assemble_matrix
from mpi4py import MPI

class AnisotropyField:
    def __init__(self, mesh,  V,  Ku, Ms, AniVec, VolN):

        '''
            AniVec is the normalized easy anisotropy axis.
            Ku: is the anisotropy constant in units of J/m^3.
            Ms: is the  saturation magnetization in units of A/m.
            H_anis:  uniaxial anisotropy field in unit of A/m
        '''

        self.mesh = mesh
        self.Ku = Ku
        self.M_s = Ms
        self.mu_0 = 4 * np.pi * 1e-7
        self.V =  V

        self.n = fem.Function(self.V)
        self.n.x.array[:] = AniVec[:]


        self.v = ufl.TestFunction(self.V)
        self.u = ufl.TrialFunction(self.V)

        self.a = ufl.dot(self.u, self.n) * ufl.dot(self.v, self.n) * ufl.dx
        self.K = assemble_matrix(fem.form(self.a))
        self.K.assemble()


        self.H_anis = fem.Function(self.V)

        prefactor = fem.Function(self.V)
        prefactor.x.array[:] = 2 * self.Ku / (self.mu_0 * self.M_s) /VolN[:]
        self.K.diagonalScale(prefactor.x.petsc_vec, None)


    def compute(self, m):
        self.H_anis.x.petsc_vec.set(0.0)
        self.K.mult(m.x.petsc_vec, self.H_anis.x.petsc_vec )
        self.H_anis.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES,     mode=PETSc.ScatterMode.FORWARD)    
    
        return self.H_anis
        #return self.prefactor * (self.temp_vec.array[self.start:self.end] / self.volNodos[self.start:self.end])


    def Energy(self, m):

        dE =- ufl.dot(m, self.H_anis ) * ufl.dx
        return 1/2*self.mu_0 * self.M_s*fem.assemble_scalar(fem.form(dE))*1e-27

