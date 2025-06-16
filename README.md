
# DCReg

DCReg: Decoupled Characterization for Robust LiDAR Registration.




| FCN-SR                                   | ME-SR(LOAM)                              | ME-TReg                                  | ME-TSVD                                  | Ours                                     |
| ---------------------------------------- | ---------------------------------------- | ---------------------------------------- | ---------------------------------------- | ---------------------------------------- |
| ![image (22)](./README/image%20(22).png) | ![image (25)](./README/image%20(25).png) | ![image (24)](./README/image%20(24).png) | ![image (23)](./README/image%20(23).png) | ![image (21)](./README/image%20(21).png) |

| Baseline Method                                      | Parametrization | Frame            | **Differentiation**                   | Lib   |
| ---------------------------------------------------- | --------------- | ---------------- | ------------------------------------- | ----- |
| [LOAM](https://github.com/laboshinl/loam_velodyne)   | Euler           | Body (右乘更新)  | Jacobian                              | Eigen |
| [ME-SR(LOAM)](https://github.com/JokerJohn/DCReg)    | R3 * SO(3)      | Body             | Jacobian                              | Eigen |
| [SuperLoc](https://github.com/JokerJohn/SuperOdom-M) | **Quaternions** | Body             | Jacobian + **Autodiff**               | Ceres |
| [X-ICP](https://github.com/JokerJohn/XICP-M)         | R3 *  SO(3)     | World (左乘更新) | Jacobian + **Autodiff + NumericDiff** | Ceres |
| [Open3D](https://github.com/isl-org/Open3D)          | SE(3)           | World            | Jacobian                              | Eigen |
| PCL ICP                                              | SE(3)           | World            | Jacobian                              | Eigen |
| Ours                                                 | R3 *  SO(3)     | Body             | Jacobian                              | Eigen |

Different frame and parametrization definition will affect the convergence.  **Autodiff + NumericDiff** are not included in the origin codes of baseline methods.

![image-20250614181622696](./README/image-20250614181622696.png)

![image-20250614181337662](./README/image-20250614181337662.png)



![optimization_landscape_journal](./README/optimization_landscape_journal-1749444065402-27.png)
