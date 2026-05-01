# Micromagnetic GPU module

Experimental GPU module for finite-element micromagnetic simulations based on the Landau-Lifshitz-Gilbert equation.

This package provides GPU-oriented tools for computing micromagnetic effective fields, evaluating the demagnetizing field, and integrating the LLG equation in a CUDA-enabled environment.

The backend is built on top of:

- [DOLFINx](https://github.com/FEniCS/dolfinx)
- PETSc with CUDA support
- CuPy
- JAX
- jaxFMM

## Project Status

This project is experimental and under active development. APIs, numerical methods, Docker dependencies, and examples may change without notice.

The current implementation is intended for research and testing purposes.

## Features

Currently supported or under development:

- Exchange field
- Uniaxial anisotropy field
- Bulk Dzyaloshinskii-Moriya interaction
- Interfacial Dzyaloshinskii-Moriya interaction
- Demagnetizing field computation using:
  - hybrid FEM-BEM method
  - Fast Multipole Method through jaxFMM (useful for larger mesh)
- Landau-Lifshitz-Gilbert time integration
- Spin-transfer torque
---

## Requirements

This module is designed to run inside a CUDA-enabled Docker environment.


```bash
docker build --no-cache -t cudolfinx-micromag:cuda12.6-jaxfmm .
```

Usage

```bash
docker run --rm --gpus all -it   -v "$PWD":/workspace   -w /workspace   cudolfinx-micromag:cuda12.6-jaxfmm  run_clean_gpu_env python3 example.py
```

## Third-Party Components and Attribution

This module relies on third-party open-source projects.

### CUDA-DOLFINx Docker Environment

The Docker environment used by this module is based on the CUDA-enabled DOLFINx project:

```text
https://github.com/bpachev/cuda-dolfinx
```

The original CUDA-DOLFINx project provides GPU acceleration support for DOLFINx/PETSc workflows.

This repository does not claim ownership of CUDA-DOLFINx. The Docker image used here is derived from that environment and extended with additional Python packages required by the micromagnetic GPU backend, including CuPy, JAX, and jaxFMM.

### jaxFMM

The demagnetizing field can be evaluated using JAXFMM as an external Fast Multipole Method library:

https://gitlab.com/jaxfmm/jaxfmm

jaxFMM is used as an external computational library for accelerating long-range magnetostatic interactions on GPU.

The jaxFMM authors note that accurate results require high-quality tetrahedral meshes with low aspect ratios.

This repository does not implement or claim ownership of the core jaxFMM algorithm. When publishing results obtained with the demagnetizing-field module, users should acknowledge the original jaxFMM project when appropriate.

### Other Dependencies

This backend may also depend on:

- DOLFINx;
- PETSc;
- mpi4py;
- NumPy;
- CuPy;
- JAX;
- JAXLIB;
- ADIOS4DOLFINx;
- PyVista.

