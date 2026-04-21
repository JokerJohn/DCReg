# Algorithm Overview

DCReg is organized around three core modules.

## 1. Spectral Degeneracy Detection

Starting from the ICP normal equation `H = J^T J`, DCReg forms the Schur complements for rotation and translation:

- `S_R = H_RR - H_Rt H_tt^{-1} H_tR`
- `S_t = H_tt - H_tR H_RR^{-1} H_tR`

This removes misleading coupling effects and exposes subspace-level observability directly.

![Method overview](../README/image-20250923182954540.png)

## 2. Physical-Axis Degeneracy Characterization

The raw Schur eigenvectors are aligned to physical motion axes:

- rotation: `roll / pitch / yaw`
- translation: `x / y / z`

This produces:

- raw Schur eigenvalues
- aligned eigenvalues in physical coordinates
- degeneracy mask
- axis contribution ratios
- clamped eigenvalues used by the solver

| Detection | Characterization |
| --- | --- |
| ![Detection](../README/image-20250923183115035.png) | ![Characterization](../README/image-20250923183019366.png) |

## 3. Preconditioned Linear Solve

DCReg clamps only the weak aligned eigenvalues, builds a block-diagonal preconditioner, and solves the normal equation with PCG. This stabilizes the weak directions without over-regularizing the fully observable ones.

## Source Entry Points

- [DCReg/include/dcreg.hpp](../DCReg/include/dcreg.hpp)
- [DCReg/include/utils.hpp](../DCReg/include/utils.hpp)
- [DCReg/src/dcreg_minimal_example.cpp](../DCReg/src/dcreg_minimal_example.cpp)

