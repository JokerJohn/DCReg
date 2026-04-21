# DCReg Wiki

> 语言切换: [English](Home) | [中文](Home.zh-CN)

**DCReg** 是一个面向退化 LiDAR 配准的 Schur 补框架。它先解耦旋转和平移的可观测性，再把原始特征空间映射到物理运动轴，最后只对弱约束方向做有针对性的预条件稳定化。

![Overview](https://raw.githubusercontent.com/JokerJohn/DCReg/main/README/image-20250923182814673.png)

## 建议阅读顺序

- [算法概览](Algorithm-Overview.zh-CN)
- [安装与快速开始](Installation-and-Quick-Start.zh-CN)
- [实验结果](Experiment-Results.zh-CN)

## 项目亮点

- `main` 分支已经完整公开 DCReg 实现。
- `baseline` 分支保留了早期公开的 baseline 代码快照。
- 当前公开版本只依赖 `Eigen + PCL`，`TBB/OpenMP` 为可选项。
- 默认的仿真点云输入已经随仓库一起提供。

## 项目入口

- [项目主页](https://github.com/JokerJohn/DCReg)
- [arXiv 论文](https://arxiv.org/abs/2509.06285)
- [视频演示](https://www.bilibili.com/video/BV1jsHQzCEra/?share_source=copy_web)

## 说明

- 运行时代码位于 [`DCReg/`](https://github.com/JokerJohn/DCReg/tree/main/DCReg)。
- 默认示例输入位于 [`DCReg/data/shifted_cylinder/measured_cloud_shifted_cylinder.pcd`](https://github.com/JokerJohn/DCReg/blob/main/DCReg/data/shifted_cylinder/measured_cloud_shifted_cylinder.pcd)。
- 如果需要维护自定义网页式文档，仓库中仍保留 [`docs/wiki/index.html`](https://github.com/JokerJohn/DCReg/blob/main/docs/wiki/index.html)。
