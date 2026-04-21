# Installation and Quick Start

## Dependencies On `main`

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

```bash
cd DCReg
mkdir -p build
cd build
cmake ..
cmake --build . -j8
```

## Run

```bash
cd DCReg/build
./dcreg_minimal_example
./dcreg_runner
```

## Default Inputs

The default simulation case is now repo-local:

- `DCReg/dataset/shifted_cylinder/measured_cloud_shifted_cylinder.pcd`

The parking-lot case still exists in code, but it requires your local data path.

## Executables

- `dcreg_minimal_example`: prints the three core DCReg modules
- `dcreg_runner`: runs the verified simulation pipeline and compares parameterizations

