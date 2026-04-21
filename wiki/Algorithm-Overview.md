# Algorithm Overview

> Language: [English](Algorithm-Overview) | [中文](Algorithm-Overview.zh-CN)

DCReg is organized around three core modules.

## 1. Spectral Degeneracy Detection

Starting from the ICP normal equation `H = J^T J`, DCReg forms the Schur complements for rotation and translation:

- `S_R = H_RR - H_Rt H_tt^{-1} H_tR`
- `S_t = H_tt - H_tR H_RR^{-1} H_tR`

This removes misleading coupling effects and exposes observability directly in the decoupled subspaces.

![Method overview](https://raw.githubusercontent.com/JokerJohn/DCReg/main/README/image-20250923182954540.png)

## 2. Physical-Axis Degeneracy Characterization

The raw Schur eigenvectors are aligned to physical motion axes:

- rotation: `roll / pitch / yaw`
- translation: `x / y / z`

This step produces:

- raw Schur eigenvalues
- aligned eigenvalues in physical coordinates
- degeneracy mask
- axis contribution ratios
- clamped eigenvalues used by the preconditioner

| Schur-Based Detection | Physical-Axis Characterization |
| --- | --- |
| ![Detection](https://raw.githubusercontent.com/JokerJohn/DCReg/main/README/image-20250923183115035.png) | ![Characterization](https://raw.githubusercontent.com/JokerJohn/DCReg/main/README/image-20250923183019366.png) |

## 3. Preconditioned Linear Solve

DCReg clamps only the weak aligned eigenvalues, builds a block-diagonal preconditioner, and solves the normal equation with PCG. This improves stability without over-regularizing the fully observable directions.

## Source Entry Points

- [`DCReg/include/dcreg.hpp`](https://github.com/JokerJohn/DCReg/blob/main/DCReg/include/dcreg.hpp)
- [`DCReg/include/utils.hpp`](https://github.com/JokerJohn/DCReg/blob/main/DCReg/include/utils.hpp)
- [`DCReg/src/dcreg_minimal_example.cpp`](https://github.com/JokerJohn/DCReg/blob/main/DCReg/src/dcreg_minimal_example.cpp)
