# DCReg Wiki

> Language: [English](Home) | [中文](Home.zh-CN)

**DCReg** is a Schur-complement-based framework for efficient degenerate LiDAR registration. It decouples rotation and translation observability, maps the raw eigenspace into physical motion axes, and stabilizes only the weak directions through a targeted preconditioned solve.

![Overview](https://raw.githubusercontent.com/JokerJohn/DCReg/main/README/image-20250923182814673.png)

## Start Here

- [Algorithm Overview](Algorithm-Overview)
- [Installation and Quick Start](Installation-and-Quick-Start)
- [Experiment Results](Experiment-Results)

## Highlights

- Full DCReg implementation is publicly available on the `main` branch.
- The historical public snapshot is preserved on the `baseline` branch.
- The current release depends on `Eigen + PCL`, with optional `TBB/OpenMP`.
- The default synthetic input is shipped inside the repository.

## Repository Links

- [Project Repository](https://github.com/JokerJohn/DCReg)
- [Paper on arXiv](https://arxiv.org/abs/2509.06285)
- [Video Demo](https://www.bilibili.com/video/BV1jsHQzCEra/?share_source=copy_web)

## Notes

- Runtime code lives under [`DCReg/`](https://github.com/JokerJohn/DCReg/tree/main/DCReg).
- The default sample input is [`DCReg/data/shifted_cylinder/measured_cloud_shifted_cylinder.pcd`](https://github.com/JokerJohn/DCReg/blob/main/DCReg/data/shifted_cylinder/measured_cloud_shifted_cylinder.pcd).
- The tabbed HTML docs remain in [`docs/wiki/index.html`](https://github.com/JokerJohn/DCReg/blob/main/docs/wiki/index.html) for maintainers who want a custom web-style view.
