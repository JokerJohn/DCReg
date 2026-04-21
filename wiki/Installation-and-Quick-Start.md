# Installation and Quick Start

> Language: [English](Installation-and-Quick-Start) | [中文](Installation-and-Quick-Start.zh-CN)

## Dependencies

Tested on Ubuntu 20.04 with C++17.

Required:

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

From the repository root:

```bash
cmake -S DCReg -B DCReg/build
cmake --build DCReg/build -j8
```

## Run

```bash
./DCReg/build/dcreg_minimal_example
./DCReg/build/dcreg_runner
```

## Default Sample Data

The default synthetic input is shipped inside the repository:

- [`DCReg/data/shifted_cylinder/measured_cloud_shifted_cylinder.pcd`](https://github.com/JokerJohn/DCReg/blob/main/DCReg/data/shifted_cylinder/measured_cloud_shifted_cylinder.pcd)

The parking-lot case is still available in code, but it expects your local data path.

## Executables

- `dcreg_minimal_example`: prints the three DCReg modules directly
- `dcreg_runner`: runs the verified simulation pipeline across parameterizations
