"""
Relaxation of a Neel skyrmion in a disk.

Model:
  - Exchange
  - Interfacial Dzyaloshinskii-Moriya interaction (DMI)
  - Uniaxial anisotropy (easy axis along +z)
  - Magnetostatic interaction is included thought a shape anisotropy.

This script performs energy relaxation by disabling precession (do_precess = 0) and using a high damping value (alpha = 1.0)
for faster convergence.

We consider that the Neel skyrmion is  stabilized in a nanotube with R = 40 nm and h = 1.5 nm.

Threading recommendations (avoid oversubscription when using MPI):

  export OMP_NUM_THREADS=1
  export NUMBA_NUM_THREADS=4   # affects numba-based kernels (e.g., bempp-cl)

Run:

  mpiexec -n 6 python Relax.py

"""


from mpi4py import MPI
import numpy as np
import dolfinx
from dolfinx import fem

from Micromagnetic import LLG 

comm = MPI.COMM_WORLD

    

# 1.  Load mesh (assumed to be in meters):

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "tube.xdmf", "r") as xdmf:
     mesh = xdmf.read_mesh(name="Grid")

# 2.  Initial state, anisotropy easy axis and DMI (considered along of rho-axis).

xyz = mesh.geometry.x
n = xyz.shape[0]

m0 = np.zeros((n, 3))
uaxis = np.zeros((n, 3))
Daxis = np.zeros((n, 3))

for i in range(n):
    x, y, z = xyz[i]
    if x**2 + (z-150)**2 <= 10*10 and y>0:
        m0[i] = [-x, -y, 0]
    else:
        m0[i] = [x,y,0]
    axis = [x,y,0] 
    axis  /= np.linalg.norm(axis)

    m0[i] /= np.linalg.norm(m0[i])
    uaxis[i] = axis
    Daxis[i] = axis

m0_array = m0.flatten()
uaxis = uaxis.flatten()
Daxis = uaxis.flatten()

# 3. Material parameters (i,e 10.1103/PhysRevB.105.054425)

Ms = 1.09817e6                # A/m
Aex = 1.6e-11             # J/m
D_int = 2.6e-3             # J/m^2
Ku = 5.4e5                # J/m^3


llg = LLG(mesh, Ms, gamma=2.211e5, alpha=0.5, do_precess=0)

llg.add_exchange(Aex=Aex)
llg.add_anisotropy(Ku, uaxis)
llg.add_dmi_interfacial(D_int, Daxis)

# 5. Time stepping setup (relaxation with stopping criterion)

t0 = 0.0                  # Initial time of the simulation
t_final = 5e-9            # Final time of the simulation If the stopping_dmdt is not reached
dt_init = 1.0e-14         # Initial time step

dt_print = 2.e-10          # simulated-time interval between solver log outputs (monitoring)
dt_snap  = 5.0e-10         # simulated-time interval between saved magnetization snapshots (XDMF)


y, ctx, elapsed, stats = llg.relax(
    m0_array=m0_array,
    t0=t0,
    t_final=t_final,         
    dt_init=dt_init,         
    dt_save=dt_print,         
    dt_snap=dt_snap,       
    output_dir="relax",
    stopping_dmdt=0.1,     # stopping criterion
    return_stats=True,
    check_every_stop=5,
    ts_rtol=1e-6,
    ts_atol=1e-6,
)

if mesh.comm.rank == 0:
    print(f"Tiempo wall-clock ts.solve : {elapsed:.3f} s")
