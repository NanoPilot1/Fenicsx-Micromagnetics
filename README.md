# Fenicsx-Micromagnetics
This repository provides a research-oriented micromagnetic framework prototype (early-stage) for solving the Landau–Lifshitz–Gilbert (LLG) equation with the finite element method in FEniCSx (PETSc/MPI).

## Features

Implemented magnetic interactions and physical effects:

- Exchange interaction
- Space-dependent uniaxial magnetic anisotropy
- Bulk Dzyaloshinskii–Moriya interaction (DMI)
- Interfacial DMI with an arbitrary symmetry-breaking axis
- Space-dependent external magnetic field
- Magnetostatic interaction (experimental; computed via a hybrid FEM–BEM approach using bempp-cl)
- Spin-transfer torque (Zhang–Li model)

The LLG equation is integrated in time using PETSc TS (Time Stepping ODE and DAE Solvers).  
A Backward Differentiation Formula (BDF) scheme is employed for time integration.  
The implementation supports MPI-based parallel execution.  
For simplicity, the entire codebase is written in Python.

For implicit time stepping we use a matrix-free Jacobian–vector product (JVP) strategy,
as commonly employed in Newton–Krylov methods for large-scale ODE/DAE systems and in
micromagnetic solvers. Comparable formulations appear in established frameworks such
as Finmag and Nmag.

## Software Requirements

To run the code, the following software is required:

- FEniCSx 0.9
- adios4dolfinx
- bempp-cl 0.4.2
- MPICH (MPI implementation)
- pyvista
- pathlib
- pandas
