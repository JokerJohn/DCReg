# Experiment Results

> Language: [English](Experiment-Results) | [中文](Experiment-Results.zh-CN)

## Verified Simulation On `main`

The default `shifted_cylinder` experiment was re-validated after the full DCReg release sync:

| Parameterization | Iter | RMSE | Fitness | Trans. Error (m) | Rot. Error (deg) | LinIt | QRfb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Euler | 9 | 0.0314988 | 0.116239 | 0.0258485 | 0.0516469 | 6 | 0 |
| SE3 | 10 | 0.0315708 | 0.116636 | 0.0271197 | 0.0507196 | 6 | 0 |
| SO3 | 10 | 0.0315708 | 0.116636 | 0.0271196 | 0.0507195 | 6 | 0 |
| Quaternion | 10 | 0.0315708 | 0.116636 | 0.0271196 | 0.0507195 | 6 | 0 |

## Representative Module-Level Log

`dcreg_minimal_example` exposes the three core modules:

- Module 1: spectral degeneracy detection
- Module 2: physical-axis characterization
- Module 3: preconditioned linear solve

## Visual Results

### Parking-Lot Real-Scene Demo

The parking-lot example registers one LiDAR frame against a prior map from a
real localization scene. The repository bundles the source frame and initial
pose metadata; the large prior map remains external and is linked from the
README.

![Parking-lot DCReg visualization](https://raw.githubusercontent.com/JokerJohn/DCReg/main/README/parking_lot_dcreg_visualization.png)

Current exported diagnostics for the bundled `pk01-1976` frame:

- Iterations: 5
- Final RMSE: 0.053225
- Final fitness: 0.060365
- Degenerate mask: `000100`, corresponding to a weak physical `x` translation axis
- Visualization: black background, intensity-colored target/source clouds, compact degeneracy card, and highlighted weak axis

### Paper Figures

| Demo | Characterization |
| --- | --- |
| ![PK01](https://raw.githubusercontent.com/JokerJohn/DCReg/main/README/8391c3ce-45dc-4b86-aed7-b496dc33ba87.gif) | ![Characterization](https://raw.githubusercontent.com/JokerJohn/DCReg/main/README/image-20250910213549613.png) |

| Controlled Simulation | Real-World Evaluation |
| --- | --- |
| ![Simulation](https://raw.githubusercontent.com/JokerJohn/DCReg/main/README/image-20250908194819193.png) | ![Real world](https://raw.githubusercontent.com/JokerJohn/DCReg/main/README/image-20250908195036175.png) |

| Runtime | Parameter Study |
| --- | --- |
| ![Runtime](https://raw.githubusercontent.com/JokerJohn/DCReg/main/README/image-20250908195549384.png) | ![Parameter](https://raw.githubusercontent.com/JokerJohn/DCReg/main/README/image-20250913000546827.png) |
