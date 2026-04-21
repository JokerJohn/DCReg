<div align="center">

# DCReg: Decoupled Characterization for Efficient Degenerate LiDAR Registration

[**Xiangcheng Hu**](https://github.com/JokerJohn)<sup>1</sup> ·
[**Xieyuanli Chen**](https://chen-xieyuanli.github.io/)<sup>2</sup> ·
[**Mingkai Jia**](https://scholar.google.com/citations?user=fcpTdvcAAAAJ&hl=en)<sup>1</sup> ·
[**Jin Wu**](https://zarathustr.github.io/)<sup>3*</sup> ·
[**Ping Tan**](https://facultyprofiles.hkust.edu.hk/profiles.php?profile=ping-tan-pingtan#publications)<sup>1</sup> ·
[**Steven L. Waslander**](https://www.trailab.utias.utoronto.ca/steven-waslander)<sup>4&dagger;</sup>

<sup>1</sup>HKUST&emsp;<sup>2</sup>NUDT&emsp;<sup>3</sup>USTB&emsp;<sup>4</sup>University of Toronto

&dagger;Project lead&emsp;*Corresponding author

<a href="https://arxiv.org/abs/2509.06285"><img src="https://img.shields.io/badge/arXiv-2509.06285-b31b1b" alt="arXiv"></a>
[![video](https://img.shields.io/badge/Video-Bilibili-74b9ff?logo=bilibili&logoColor=red)](https://www.bilibili.com/video/BV1jsHQzCEra/?share_source=copy_web)
[![GitHub Stars](https://img.shields.io/github/stars/JokerJohn/DCReg.svg)](https://github.com/JokerJohn/DCReg/stargazers)
[![GitHub Issues](https://img.shields.io/github/issues/JokerJohn/DCReg.svg)](https://github.com/JokerJohn/DCReg/issues)

</div>

![Overview](./README/image-20250923182814673.png)

`main` now contains the verified full public release of the DCReg algorithm. The previous public snapshot, which exposed the baseline-oriented code and historical result files, is preserved on the `baseline` branch.

## Branch Layout

- `main`: full DCReg release with the Schur-based characterization module, preconditioned solver, runnable simulation example, and minimal characterization example.
- `baseline`: original public snapshot before the full DCReg code release.

The repository root keeps three major parts:

- `DCReg/`: current C++ implementation used in this release.
- `baseline/`: reference baseline projects kept for comparison.
- `results/`: published experiment artifacts and paper-facing result files.

## What Is Open-Sourced On `main`

- Full DCReg solver with the three core modules:
  1. spectral degeneracy detection from Schur complements
  2. physical-axis degeneracy characterization in `rpy/xyz`
  3. targeted preconditioned linear solve
- A runnable simulation executable that keeps the plain ICP baseline (`Algorithm::kNone`) and the DCReg path (`Algorithm::kDCReg`) in the same codebase.
- A minimal example executable that prints the three DCReg modules step by step and is intended for integration into other SLAM systems.
- The previous baseline branch and paper result folders remain available in this repository for comparison.

## Dependencies

Tested on Ubuntu 20.04 with C++17.

Required on `main`:

- Eigen3
- PCL

Optional:

- TBB
- OpenMP

No longer required on `main`:

- Ceres
- yaml-cpp
- Open3D

## Build

```bash
cd DCReg
mkdir -p build
cd build
cmake ..
cmake --build . -j8
```

This builds two executables:

- `dcreg_runner`: full simulation runner with baseline and DCReg.
- `dcreg_minimal_example`: minimal three-module DCReg example.

## Run

```bash
cd DCReg/build
./dcreg_minimal_example
./dcreg_runner
```

The default simulation case is repo-local and uses:

- `DCReg/dataset/shifted_cylinder/measured_cloud_shifted_cylinder.pcd`

The real-world parking-lot case is still available in code, but it expects your local data path. Edit `kParkingLotPk01Case` in [DCReg/include/utils.hpp](./DCReg/include/utils.hpp) if you want to run that case.

## Three Core Modules

### Module 1: Spectral Degeneracy Detection

DCReg first forms the rotation and translation Schur complements from the ICP normal equation and inspects their spectra:

- `S_R = H_RR - H_Rt H_tt^{-1} H_tR`
- `S_t = H_tt - H_tR H_RR^{-1} H_tR`

This step provides the raw Schur eigenvalues and coarse condition numbers.

### Module 2: Physical-Axis Degeneracy Characterization

The raw Schur eigenvectors are then mapped into physical motion axes:

- rotation: `roll / pitch / yaw`
- translation: `x / y / z`

DCReg reports:

- the degenerate mask
- raw Schur eigenvalues
- aligned eigenvalues in physical coordinates
- per-axis contribution ratios
- clamped eigenvalues used by the preconditioner

### Module 3: Preconditioned Linear Solve

Finally, DCReg constructs a targeted block-diagonal preconditioner from the characterized weak directions and solves the normal equation with PCG. If the iterative solve becomes numerically invalid, it falls back to the dense QR solve.

## Example Log: `dcreg_minimal_example`

```text
Synthetic DCReg example
[Module 1] Spectral degeneracy detection
cond_full: 1122.345689
cond_schur_rot: 36.087707
cond_schur_trans: 742.066396

[Module 2] Physical-axis degeneracy characterization
degenerate_mask: 001001
raw_lambda_rot:  1.001796 19.881966 36.152505
raw_lambda_trans:  0.032394 12.252030 24.038714
aligned_lambda_rpy: 36.152505 19.881966  1.001796
aligned_lambda_xyz: 24.038714 12.252030  0.032394
rot_axis_contribution_ratio(rows=rpy, cols=aligned_rpy):
                      mode_r      mode_p      mode_y
roll                0.995659    0.000004    0.004337
pitch               0.000001    0.999791    0.000208
yaw                 0.004340    0.000205    0.995456
trans_axis_contribution_ratio(rows=xyz, cols=aligned_xyz):
                      mode_x      mode_y      mode_z
x                   0.999989    0.000000    0.000011
y                   0.000000    0.999834    0.000166
z                   0.000011    0.000166    0.999823
clamped_lambda_rpy: 36.152505 19.881966  3.615250
clamped_lambda_xyz: 24.038714 12.252030  2.403871

[Module 3] Preconditioned linear solve
preconditioned_delta:  0.190398 -0.306306 -2.620922  0.095321 -0.515479 41.997563
pcg_iterations: 6
pcg_relative_residual: 0.000000
qr_fallback: 0
```

## Verified Simulation Result On `dcreg_runner`

The default `shifted_cylinder` case was re-validated after the refactor. The final numbers are:

| Parameterization | Iter | RMSE | Fitness | Translation Error (m) | Rotation Error (deg) | LinIt | QRfb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Euler | 9 | 0.0314988 | 0.116239 | 0.0258485 | 0.0516469 | 6 | 0 |
| SE3 | 10 | 0.0315708 | 0.116636 | 0.0271197 | 0.0507196 | 6 | 0 |
| SO3 | 10 | 0.0315708 | 0.116636 | 0.0271196 | 0.0507195 | 6 | 0 |
| Quaternion | 10 | 0.0315708 | 0.116636 | 0.0271196 | 0.0507195 | 6 | 0 |

## Workflow

Typical workflow on `main`:

1. Build under `DCReg/build`.
2. Run `./dcreg_minimal_example` to inspect the three DCReg modules and verify the characterization output.
3. Run `./dcreg_runner` to reproduce the default `shifted_cylinder` experiment.
4. Switch the test case or parameterization in [DCReg/src/main.cpp](./DCReg/src/main.cpp) if you want long-run or real-world evaluation.

## Notes For Integration

- The compact public implementation is centered on [DCReg/include/dcreg.hpp](./DCReg/include/dcreg.hpp) and [DCReg/include/utils.hpp](./DCReg/include/utils.hpp).
- `Algorithm::kNone` keeps the plain ICP baseline path.
- `Algorithm::kDCReg` enables the Schur-based characterization and preconditioned solver.
- The minimal example is the cleanest entry point if you want to port the solver into another SLAM system.

## Historical Results And Baselines

- The old public release is preserved on the `baseline` branch.
- The `baseline/` directory keeps the reference baseline projects distributed with this repository.
- The `results/` directory keeps paper-facing result folders and published comparison artifacts.

## Citation

```bibtex
@misc{hu2025dcreg,
  title={DCReg: Decoupled Characterization for Efficient Degenerate LiDAR Registration},
  author={Xiangcheng Hu and Xieyuanli Chen and Mingkai Jia and Jin Wu and Ping Tan and Steven L. Waslander},
  year={2025},
  eprint={2509.06285},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2509.06285}
}
```
