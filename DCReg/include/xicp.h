#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <ceres/ceres.h>

namespace XICP {

// 枚举类型定义
    enum class DegeneracyAwarenessMethod {
        kNone = 0,
        kSolutionRemapping = 1,
        kEqualityConstraints = 2,
        kOptimizedEqualityConstraints = 3,
        kInequalityConstraints = 4
    };

    enum class LocalizabilityCategory {
        kLocalizable = 0,
        kNonLocalizable = 1
    };

    enum class LocalizabilitySamplingType {
        kHighContributionPoints = 0,
        kMixedContributionPoints = 1,
        kUnnecessary = 2,
        kInsufficientPoints = 3
    };

// 退化检测参数
    template<typename T>
    struct DegeneracyDetectionParameters {
        // 阈值参数 - 根据ICP.cpp的典型值设置
        T enoughInformationThreshold = 100.0;      // 足够信息阈值
        T insufficientInformationThreshold = 10.0;  // 不足信息阈值
        T highInformationThreshold = 1000.0;        // 高信息阈值
        T solutionRemappingThreshold = 120.0;       // Solution Remapping条件数阈值
        T point2NormalMinimalAlignmentCosineThreshold = 0.866;  // cos(30°)
        T point2NormalStrongAlignmentCosineThreshold = 0.966;   // cos(15°)
        T inequalityBoundMultiplier = 0.5;

        // 额外的参数（来自ICP.cpp）
        size_t numberOfPoints = 0;
        size_t contributingNumberOfPoints = 0;
        size_t highlyContributingNumberOfPoints = 0;
        size_t highlyContributingNumberOfPoints_trans = 0;
        size_t highlyContributingNumberOfPoints_rot = 0;
        T combinedContribution = 0.0;
        T highContribution = 0.0;

        // 控制参数
        DegeneracyAwarenessMethod degeneracyAwarenessMethod = DegeneracyAwarenessMethod::kNone;
        Eigen::Matrix4d transformationToOptimizationFrame = Eigen::Matrix4d::Identity();
        bool isPrintingEnabled = false;
    };

// 局部化约束结果
    template<typename T>
    struct LocalizabilityConstraints {
        Eigen::Matrix<T, 3, 1> rotationConstraintValues_ = Eigen::Matrix<T, 3, 1>::Zero();
        Eigen::Matrix<T, 3, 1> translationConstraintValues_ = Eigen::Matrix<T, 3, 1>::Zero();
    };

// 局部化分析结果
    template<typename T>
    struct LocalizabilityAnalysisResults {
        // 特征向量
        Eigen::Matrix<T, 3, 3> rotationEigenvectors_ = Eigen::Matrix<T, 3, 3>::Zero();
        Eigen::Matrix<T, 3, 3> translationEigenvectors_ = Eigen::Matrix<T, 3, 3>::Zero();

        // 局部化状态
        Eigen::Matrix<T, 3, 1> localizabilityRpy_ = Eigen::Matrix<T, 3, 1>::Zero();
        Eigen::Matrix<T, 3, 1> localizabilityXyz_ = Eigen::Matrix<T, 3, 1>::Zero();

        // 约束值
        LocalizabilityConstraints<T> localizabilityConstraints_;

        // Solution remapping投影矩阵
        Eigen::Matrix<T, 6, 6> solutionRemappingProjectionMatrix_ = Eigen::Matrix<T, 6, 6>::Identity();
    };

// 添加单维度约束类
    struct SingleDimensionConstraint {
        int dimension;

        SingleDimensionConstraint(int dim);

        template<typename T>
        bool operator()(const T *const x, T *residual) const;
    };

// 修复1: 正确的点到平面代价函数（用于手动雅可比方法）
    struct Point2PlaneLinearCostFunctor {
        Eigen::Matrix<double, 6, 6> A;
        Eigen::Matrix<double, 6, 1> b;

        Point2PlaneLinearCostFunctor(const Eigen::Matrix<double, 6, 6> &A_,
                                     const Eigen::Matrix<double, 6, 1> &b_);

        template<typename T>
        bool operator()(const T *const x, T *residual) const;
    };

// 修复2: 正确的点到平面AutoDiff代价函数
    struct Point2PlaneResidualAutoDiff {
        Eigen::Vector3d src_point;  // 源点（体坐标系）
        Eigen::Vector3d tgt_point;  // 目标点
        Eigen::Vector3d normal;     // 平面法向量

        Point2PlaneResidualAutoDiff(const Eigen::Vector3d &src,
                                    const Eigen::Vector3d &tgt,
                                    const Eigen::Vector3d &n);

        template<typename T>
        bool operator()(const T *const delta, T *residual) const;
    };

// 数值微分版本的代价函数
    struct Point2PlaneResidualNumeric {
        Eigen::Vector3d src_point;
        Eigen::Vector3d tgt_point;
        Eigen::Vector3d normal;

        Point2PlaneResidualNumeric(const Eigen::Vector3d &src,
                                   const Eigen::Vector3d &tgt,
                                   const Eigen::Vector3d &n);

        bool operator()(const double *const delta, double *residual) const;
    };

// 方向约束（用于Ceres优化）
    struct DirectionConstraint {
        DirectionConstraint(const Eigen::VectorXd &direction, double target_value);

        template<typename T>
        bool operator()(const T *const x, T *residual) const;

    private:
        Eigen::VectorXd direction_;
        double target_value_;
    };

// 在x_icp.hpp中添加不等式约束类
    struct InequalityDirectionConstraint {
        InequalityDirectionConstraint(const Eigen::VectorXd &direction, double bound);

        template<typename T>
        bool operator()(const T *const x, T *residual) const;

    private:
        Eigen::VectorXd direction_;
        double bound_;
    };

// XICP核心类
    template<typename T>
    class XICPCore {
    public:
        using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
        using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
        using Matrix6 = Eigen::Matrix<T, 6, 6>;
        using Vector6 = Eigen::Matrix<T, 6, 1>;
        using Matrix3 = Eigen::Matrix<T, 3, 3>;
        using Vector3 = Eigen::Matrix<T, 3, 1>;
        using Matrix4 = Eigen::Matrix<T, 4, 4>;

        XICPCore() = default;

        // 设置参数
        void setParameters(const DegeneracyDetectionParameters<T> &params);

        // 在XICPCore类的public部分添加
        const DegeneracyDetectionParameters<T> &getParameters() const;

        // 主要的退化检测接口
        bool detectDegeneracy(
                const Matrix &sourcePoints,
                const Matrix &targetPoints,
                const Matrix &targetNormals,
                const Matrix6 &hessian,
                LocalizabilityAnalysisResults<T> &results);

        // 获取退化方向
        Matrix6 getDegenerateDirections(const LocalizabilityAnalysisResults<T> &results) const;

        // 获取约束值
        Vector6 getConstraintValues(const LocalizabilityAnalysisResults<T> &results) const;

        // 使用Ceres求解带约束的系统 - 作为TestRunner的成员函数
        void solveDegenerateSystemWithCeres(
                const Eigen::Matrix<double, 6, 6> &hessian,
                const Eigen::Matrix<double, 6, 1> &b,
                const XICP::LocalizabilityAnalysisResults<double> &xicpResults,
                Eigen::Matrix<double, 6, 1> &solution);

        // 使用Ceres AutoDiff求解的替代方案
        void solveDegenerateSystemWithCeresAutoDiff(
                const std::vector <Eigen::Vector3d> &valid_src,
                const std::vector <Eigen::Vector3d> &valid_tgt,
                const std::vector <Eigen::Vector3d> &valid_normals,
                const XICP::LocalizabilityAnalysisResults<double> &xicpResults,
                Eigen::Matrix<double, 6, 1> &solution,
                bool useNumericDiff);

        // 使用增广拉格朗日方法求解带约束的线性系统
        void solveDegenerateSystemWithCeresKKT(
                const Eigen::Matrix<double, 6, 6> &hessian,
                const Eigen::Matrix<double, 6, 1> &b,
                const XICP::LocalizabilityAnalysisResults<double> &xicpResults,
                Eigen::Matrix<double, 6, 1> &solution);

    private:
        DegeneracyDetectionParameters<T> params_;

        // 3x3分块特征分析
        void eigenAnalysis3x3(const Matrix6 &hessian, LocalizabilityAnalysisResults<T> &results);

        // 优化的退化检测方法
        bool detectLocalizabilityOptimized(
                const Matrix &sourcePoints,
                const Matrix &targetNormals,
                const Matrix6 &hessian,
                LocalizabilityAnalysisResults<T> &results);

        // 三元退化检测方法（用于Equality和Inequality约束）
        bool detectLocalizabilityTernary(
                const Matrix &sourcePoints,
                const Matrix &targetPoints,
                const Matrix &targetNormals,
                const Matrix6 &hessian,
                LocalizabilityAnalysisResults<T> &results);

        // Solution Remapping方法
        bool detectLocalizabilitySolutionRemapping(
                const Matrix6 &hessian,
                LocalizabilityAnalysisResults<T> &results);

        // 方向局部化检测函数 - 返回是否可定位
        bool detectDirectionLocalizability(
                const Vector3 &eigenvector,
                const Matrix &alignmentVectors,
                T &combinedContribution,
                T &highContribution);

        // 比较函数用于对齐列表排序
        static bool compareAlignmentList(const std::pair <Eigen::Index, T> &p1,
                                         const std::pair <Eigen::Index, T> &p2);

        // 子空间局部化检测（三元级别）
        void detectSubspaceLocalizabilityTernary(
                const Matrix &sourcePoints,
                const Matrix &targetPoints,
                const Matrix &targetNormals,
                const Matrix &alignmentVectors,
                const Matrix &deltas,
                const Vector3 &eigenvector,
                std::vector <std::pair<Eigen::Index, T>> &alignmentList,
                int index,
                bool isRotationSubspace,
                LocalizabilityAnalysisResults<T> &results);

        // 决定三元局部化级别
        LocalizabilitySamplingType decideLocalizabilityLevelTernary(
                int index, bool isRotationSubspace,
                LocalizabilityAnalysisResults<T> &results);

        // 求解部分约束
        void solvePartialConstraints(
                const Matrix &sourcePoints,
                const Matrix &targetPoints,
                const Matrix &targetNormals,
                const Matrix &deltas,
                const std::vector <std::pair<Eigen::Index, T>> &alignmentList,
                Eigen::Index pointsToSample,
                const Vector3 &eigenvector,
                int index,
                bool isRotationSubspace,
                LocalizabilityAnalysisResults<T> &results);
    };

} // namespace XICP