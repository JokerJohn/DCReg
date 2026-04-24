# DCReg Wiki

> Language: [English](Home) | [中文](Home.zh-CN)

**DCReg** is a Schur-complement-based framework for efficient degenerate LiDAR registration. It decouples rotation and translation observability, maps the raw eigenspace into physical motion axes, and stabilizes only the weak directions through a targeted preconditioned solve.

![Overview](https://raw.githubusercontent.com/JokerJohn/DCReg/main/README/image-20250923182814673.png)

## Start Here

- [Algorithm Overview](Algorithm-Overview)
- [Installation and Quick Start](Installation-and-Quick-Start)
- [Experiment Results](Experiment-Results)
- [Parking-Lot real-scene demo](Installation-and-Quick-Start#parking-lot-real-scene-demo)

## Highlights

- Schur-complement-based spectral degeneracy detection removes misleading rotation-translation coupling before observability analysis.
- Physical-axis characterization maps weak modes to `roll/pitch/yaw` and `x/y/z` with aligned eigenvalues and contribution ratios.
- Targeted preconditioning stabilizes only the weak directions instead of damping the full coupled system.
- Lightweight runtime stack: `Eigen + PCL`, with optional `TBB/OpenMP`.

## Repository Links

- [Project Repository](https://github.com/JokerJohn/DCReg)
- [Paper on arXiv](https://arxiv.org/abs/2509.06285)
- [Video Demo](https://www.bilibili.com/video/BV1jsHQzCEra/?share_source=copy_web)

## Notes

- Runtime code lives under [`DCReg/`](https://github.com/JokerJohn/DCReg/tree/main/DCReg).
- The default sample input is [`DCReg/data/shifted_cylinder/measured_cloud_shifted_cylinder.pcd`](https://github.com/JokerJohn/DCReg/blob/main/DCReg/data/shifted_cylinder/measured_cloud_shifted_cylinder.pcd).
- The parking-lot source frame is bundled under [`DCReg/dataset/Parking-Lot-example/`](https://github.com/JokerJohn/DCReg/tree/main/DCReg/dataset/Parking-Lot-example); download the large prior map from the README data link before running the real-scene demo.
- Next up: an open-source DCReg-based localization system that demonstrates integration into larger pipelines.
