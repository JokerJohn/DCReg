# 实验结果

> 语言切换: [English](Experiment-Results) | [中文](Experiment-Results.zh-CN)

## `main` 分支已验证的默认仿真结果

完整公开版本同步后，默认的 `shifted_cylinder` 仿真实验已经重新验证：

| 参数化 | Iter | RMSE | Fitness | 平移误差 (m) | 旋转误差 (deg) | LinIt | QRfb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Euler | 9 | 0.0314988 | 0.116239 | 0.0258485 | 0.0516469 | 6 | 0 |
| SE3 | 10 | 0.0315708 | 0.116636 | 0.0271197 | 0.0507196 | 6 | 0 |
| SO3 | 10 | 0.0315708 | 0.116636 | 0.0271196 | 0.0507195 | 6 | 0 |
| Quaternion | 10 | 0.0315708 | 0.116636 | 0.0271196 | 0.0507195 | 6 | 0 |

## 模块级示例日志

`dcreg_minimal_example` 会按照三个核心模块输出：

- 模块 1：谱退化检测
- 模块 2：物理轴退化表征
- 模块 3：预条件线性求解

## 可视化结果

### 停车场真实场景 Demo

停车场案例使用一个真实 LiDAR 单帧与先验地图进行配准。仓库内已包含
source frame 和初始位姿元数据；prior map 体积较大，仍通过 README 中的数据链接下载。

![Parking-lot DCReg visualization](https://raw.githubusercontent.com/JokerJohn/DCReg/main/README/parking_lot_dcreg_visualization.png)

当前 `pk01-1976` 案例导出的诊断结果：

- 迭代次数：5
- 最终 RMSE：0.053225
- 最终 fitness：0.060365
- 退化掩码：`000100`，对应物理 `x` 平移方向弱观测
- 可视化：黑色背景、按 intensity 着色的 target/source 点云、`100 m` 局部 target map、紧凑退化诊断卡片和高亮弱轴

### 论文图示

| 演示 | 表征 |
| --- | --- |
| ![PK01](https://raw.githubusercontent.com/JokerJohn/DCReg/main/README/8391c3ce-45dc-4b86-aed7-b496dc33ba87.gif) | ![Characterization](https://raw.githubusercontent.com/JokerJohn/DCReg/main/README/image-20250910213549613.png) |

| 可控仿真 | 真实场景评测 |
| --- | --- |
| ![Simulation](https://raw.githubusercontent.com/JokerJohn/DCReg/main/README/image-20250908194819193.png) | ![Real world](https://raw.githubusercontent.com/JokerJohn/DCReg/main/README/image-20250908195036175.png) |

| 运行时间 | 参数分析 |
| --- | --- |
| ![Runtime](https://raw.githubusercontent.com/JokerJohn/DCReg/main/README/image-20250908195549384.png) | ![Parameter](https://raw.githubusercontent.com/JokerJohn/DCReg/main/README/image-20250913000546827.png) |
