# DCReg Wiki

> 语言切换: [English](Home) | [中文](Home.zh-CN)

**DCReg** 是一个面向退化 LiDAR 配准的 Schur 补框架。它先解耦旋转和平移的可观测性，再把原始特征空间映射到物理运动轴，最后只对弱约束方向做有针对性的预条件稳定化。

![Overview](https://raw.githubusercontent.com/JokerJohn/DCReg/main/README/image-20250923182814673.png)

## 建议阅读顺序

- [算法概览](Algorithm-Overview.zh-CN)
- [安装与快速开始](Installation-and-Quick-Start.zh-CN)
- [实验结果](Experiment-Results.zh-CN)
- [停车场真实场景 Demo](Installation-and-Quick-Start.zh-CN#停车场真实场景-demo)

## 项目亮点

- 基于 Schur 补的谱退化检测，在分析可观测性之前先消除旋转和平移耦合带来的遮蔽效应。
- 物理轴退化表征把弱方向映射到 `roll/pitch/yaw` 和 `x/y/z`，并给出对齐特征值与贡献比例。
- 有针对性的预条件只稳定弱方向，而不是对整个耦合系统做统一阻尼。
- 轻量依赖栈：`Eigen + PCL`，`TBB/OpenMP` 为可选项。

## 项目入口

- [项目主页](https://github.com/JokerJohn/DCReg)
- [arXiv 论文](https://arxiv.org/abs/2509.06285)
- [视频演示](https://www.bilibili.com/video/BV1jsHQzCEra/?share_source=copy_web)

## 说明

- 运行时代码位于 [`DCReg/`](https://github.com/JokerJohn/DCReg/tree/main/DCReg)。
- 默认示例输入位于 [`DCReg/data/shifted_cylinder/measured_cloud_shifted_cylinder.pcd`](https://github.com/JokerJohn/DCReg/blob/main/DCReg/data/shifted_cylinder/measured_cloud_shifted_cylinder.pcd)。
- 停车场 source frame 已随仓库放在 [`DCReg/dataset/Parking-Lot-example/`](https://github.com/JokerJohn/DCReg/tree/main/DCReg/dataset/Parking-Lot-example)，大尺寸 prior map 请通过 README 中的数据链接下载后再运行真实场景 demo。
- 下一步计划是开源一个基于 DCReg 的定位系统，用来展示它如何整合到更完整的算法管线中。
