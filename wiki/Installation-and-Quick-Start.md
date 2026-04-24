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

Not required by the C++ core:

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

## Parking-Lot Real-Scene Demo

Build and run the real-scene single-frame-to-map example:

```bash
cmake --build DCReg/build -j8 --target dcreg_parking_lot_example
./DCReg/build/dcreg_parking_lot_example
```

The C++ runner keeps the algorithm dependency surface unchanged: `Eigen + PCL`,
with optional `TBB/OpenMP`. The Open3D viewer is an optional Python-only
visualization helper:

```bash
python3 -m pip install open3d numpy pillow
python3 scripts/visualize_parking_lot_example.py
```

## Default Sample Data

Small demo inputs are bundled under `DCReg/data/`. The complete data package,
including the large parking-lot prior map, is linked in the README. For the
parking-lot demo, place the downloaded map here:

```text
DCReg/data/Parking-Lot-example/prior_map.pcd
```

## Executables

- `dcreg_minimal_example`: prints the three DCReg modules directly
- `dcreg_runner`: runs the verified simulation pipeline across parameterizations
- `dcreg_parking_lot_example`: runs the parking-lot real-scene matching case
