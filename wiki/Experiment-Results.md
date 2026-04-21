# Experiment Results

## Minimal Example

`dcreg_minimal_example` mirrors the three core modules:

- Module 1: spectral detection
- Module 2: physical-axis characterization
- Module 3: preconditioned solve

Representative output:

```text
[Module 1] Spectral degeneracy detection
cond_full: 1122.345689
cond_schur_rot: 36.087707
cond_schur_trans: 742.066396
...
[Module 3] Preconditioned linear solve
pcg_iterations: 6
pcg_relative_residual: 0.000000
qr_fallback: 0
```

## Verified Simulation

The default `shifted_cylinder` experiment on `main` was re-validated after the full release sync:

| Parameterization | Iter | RMSE | Fitness | Trans. Error (m) | Rot. Error (deg) | LinIt | QRfb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Euler | 9 | 0.0314988 | 0.116239 | 0.0258485 | 0.0516469 | 6 | 0 |
| SE3 | 10 | 0.0315708 | 0.116636 | 0.0271197 | 0.0507196 | 6 | 0 |
| SO3 | 10 | 0.0315708 | 0.116636 | 0.0271196 | 0.0507195 | 6 | 0 |
| Quaternion | 10 | 0.0315708 | 0.116636 | 0.0271196 | 0.0507195 | 6 | 0 |

## Visual Results

| Demo | Characterization |
| --- | --- |
| ![PK01](../README/8391c3ce-45dc-4b86-aed7-b496dc33ba87.gif) | ![Case](../README/image-20250910213549613.png) |

| Simulation | Real World |
| --- | --- |
| ![Simulation](../README/image-20250908194819193.png) | ![Real world](../README/image-20250908195036175.png) |

| Runtime | Parameter |
| --- | --- |
| ![Runtime](../README/image-20250908195549384.png) | ![Parameter](../README/image-20250913000546827.png) |

