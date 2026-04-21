# 算法概览

> 语言切换: [English](Algorithm-Overview) | [中文](Algorithm-Overview.zh-CN)

DCReg 的核心由三个模块组成。

## 1. 谱退化检测

从 ICP 法方程 `H = J^T J` 出发，DCReg 分别构造旋转和平移的 Schur 补：

- `S_R = H_RR - H_Rt H_tt^{-1} H_tR`
- `S_t = H_tt - H_tR H_RR^{-1} H_tR`

这样可以去掉耦合项对可观测性分析的遮蔽作用，直接在解耦子空间中分析谱信息。

![Method overview](https://raw.githubusercontent.com/JokerJohn/DCReg/main/README/image-20250923182954540.png)

## 2. 物理轴退化表征

然后把原始 Schur 特征向量映射到物理运动轴：

- 旋转：`roll / pitch / yaw`
- 平移：`x / y / z`

这一阶段会输出：

- 原始 Schur 特征值
- 对齐到物理轴后的特征值
- 退化掩码
- 各轴贡献比例
- 用于预条件器的钳制后特征值

| Schur 检测 | 物理轴表征 |
| --- | --- |
| ![Detection](https://raw.githubusercontent.com/JokerJohn/DCReg/main/README/image-20250923183115035.png) | ![Characterization](https://raw.githubusercontent.com/JokerJohn/DCReg/main/README/image-20250923183019366.png) |

## 3. 预条件线性求解

DCReg 只对弱方向做谱钳制，构造块对角预条件器，再用 PCG 求解法方程。这样可以稳定退化方向，而不会把强观测方向一并过度正则化。

## 代码入口

- [`DCReg/include/dcreg.hpp`](https://github.com/JokerJohn/DCReg/blob/main/DCReg/include/dcreg.hpp)
- [`DCReg/include/utils.hpp`](https://github.com/JokerJohn/DCReg/blob/main/DCReg/include/utils.hpp)
- [`DCReg/src/dcreg_minimal_example.cpp`](https://github.com/JokerJohn/DCReg/blob/main/DCReg/src/dcreg_minimal_example.cpp)
