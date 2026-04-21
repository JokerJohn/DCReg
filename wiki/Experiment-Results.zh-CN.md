# 实验结果

## 最小示例

`dcreg_minimal_example` 会按照三个核心模块输出：

- 模块 1：谱退化检测
- 模块 2：物理轴退化表征
- 模块 3：预条件求解

代表性输出：

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

## 已验证的默认仿真结果

`main` 分支默认的 `shifted_cylinder` 实验在完整同步后已经重新验证：

| 参数化 | Iter | RMSE | Fitness | 平移误差 (m) | 旋转误差 (deg) | LinIt | QRfb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Euler | 9 | 0.0314988 | 0.116239 | 0.0258485 | 0.0516469 | 6 | 0 |
| SE3 | 10 | 0.0315708 | 0.116636 | 0.0271197 | 0.0507196 | 6 | 0 |
| SO3 | 10 | 0.0315708 | 0.116636 | 0.0271196 | 0.0507195 | 6 | 0 |
| Quaternion | 10 | 0.0315708 | 0.116636 | 0.0271196 | 0.0507195 | 6 | 0 |

## 可视化结果

| 演示 | 表征 |
| --- | --- |
| ![PK01](../README/8391c3ce-45dc-4b86-aed7-b496dc33ba87.gif) | ![Case](../README/image-20250910213549613.png) |

| 仿真 | 真实场景 |
| --- | --- |
| ![Simulation](../README/image-20250908194819193.png) | ![Real world](../README/image-20250908195036175.png) |

| 运行时间 | 参数分析 |
| --- | --- |
| ![Runtime](../README/image-20250908195549384.png) | ![Parameter](../README/image-20250913000546827.png) |

