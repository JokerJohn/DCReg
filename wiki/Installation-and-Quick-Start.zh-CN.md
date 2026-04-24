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

这些依赖不再属于 C++ 核心算法：

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

## 停车场真实场景 Demo

编译并运行单帧 LiDAR 到先验地图的真实场景匹配案例：

```bash
cmake --build DCReg/build -j8 --target dcreg_parking_lot_example
./DCReg/build/dcreg_parking_lot_example
```

C++ runner 仍然保持轻量依赖边界：`Eigen + PCL`，`TBB/OpenMP` 可选。
Open3D 只用于独立的 Python 可视化脚本，不影响原始算法编译依赖：

```bash
python3 -m pip install open3d numpy pillow
python3 scripts/visualize_parking_lot_example.py
```

## 默认示例数据

小型 demo 输入放在 `DCReg/data/`。完整数据包，包括停车场大尺寸 prior
map，见 README 中的数据链接。运行停车场 demo 时，把下载的地图放置为：

```text
DCReg/data/Parking-Lot-example/prior_map.pcd
```

## 可执行程序

- `dcreg_minimal_example`：直接输出 DCReg 的三个核心模块
- `dcreg_runner`：运行已验证的默认仿真流程并比较不同参数化结果
- `dcreg_parking_lot_example`：运行停车场真实场景单帧到地图匹配案例
