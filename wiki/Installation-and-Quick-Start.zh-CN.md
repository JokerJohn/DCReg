# 安装与快速开始

> 语言切换: [English](Installation-and-Quick-Start) | [中文](Installation-and-Quick-Start.zh-CN)

## 依赖

测试环境为 Ubuntu 20.04 和 C++17。

必需：

- Eigen3
- PCL

可选：

- TBB
- OpenMP

`main` 分支已不再需要：

- Ceres
- yaml-cpp
- Open3D

## 编译

在仓库根目录下执行：

```bash
cmake -S DCReg -B DCReg/build
cmake --build DCReg/build -j8
```

## 运行

```bash
./DCReg/build/dcreg_minimal_example
./DCReg/build/dcreg_runner
```

## 默认示例数据

默认的仿真输入已经随仓库提供：

- [`DCReg/data/shifted_cylinder/measured_cloud_shifted_cylinder.pcd`](https://github.com/JokerJohn/DCReg/blob/main/DCReg/data/shifted_cylinder/measured_cloud_shifted_cylinder.pcd)

停车场真实数据案例仍然保留在代码中，但需要你自行改成本地数据路径。

## 可执行程序

- `dcreg_minimal_example`：直接输出 DCReg 的三个核心模块
- `dcreg_runner`：运行已验证的默认仿真流程并比较不同参数化结果
