# 安装与快速开始

## `main` 分支依赖

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

```bash
cd DCReg
mkdir -p build
cd build
cmake ..
cmake --build . -j8
```

## 运行

```bash
cd DCReg/build
./dcreg_minimal_example
./dcreg_runner
```

## 默认输入

默认仿真输入已经放在仓库内：

- `DCReg/dataset/shifted_cylinder/measured_cloud_shifted_cylinder.pcd`

停车场真实数据案例仍然保留在代码中，但需要你改成本地数据路径。

## 可执行程序

- `dcreg_minimal_example`：输出 DCReg 三个核心模块的日志
- `dcreg_runner`：运行完整仿真流程并比较不同参数化结果

