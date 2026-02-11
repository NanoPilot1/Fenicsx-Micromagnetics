# Fenicsx-Micromagnetics
This repository provides a research-oriented micromagnetic framework prototype (early-stage) for solving the Landau-Lifshitz-Gilbert (LLG) equation with the finite element method in FEniCSx (PETSc/MPI).

## Features

Implemented magnetic interactions and physical effects:

- Exchange interaction
- Space-dependent uniaxial magnetic anisotropy
- Bulk Dzyaloshinskii–Moriya interaction (DMI)
- Interfacial DMI with an arbitrary symmetry-breaking axis
- Cubic Anisotropy
- Space-dependent external magnetic field
- Magnetostatic interaction (experimental; computed via a hybrid FEM–BEM approach using bempp-cl)
- Spin-transfer torque (Zhang–Li model)

The LLG equation is integrated in time using PETSc TS (Time Stepping ODE and DAE Solvers).  
A Backward Differentiation Formula (BDF) scheme is employed for time integration.  
The implementation supports MPI-based parallel execution.  
For simplicity, the entire codebase is written in Python.

## Software Requirements

To run the code, the following software is required:

- FEniCSx /dolfinx 0.9
- adios4dolfinx
- bempp-cl 0.4.2
- MPICH (MPI implementation)
- pyvista
- numpy, scipy, pandas, h5py, numba
- meshio

## Installation

Create a conda environment and install dependencies:


```bash

conda create -n fenicsx-micromag python=3.10 -y
conda activate fenicsx-micromag

conda install -c conda-forge -y numpy scipy pandas h5py numba meshio pyvista mpich "fenics-dolfinx=0.9.*" adios4dolfinx

pip install bempp-cl

git clone https://github.com/NanoPilot1/Fenicsx-Micromagnetics.git
cd Fenicsx-Micromagnetics/src/
pip install -e 

 


