#ifndef CLOUD_MAP_EVALUATION_SUPERLOC_H
#define CLOUD_MAP_EVALUATION_SUPERLOC_H

#pragma once

// SuperLoc ICP Integration for icp_so3_test_runner
#include <Eigen/Core>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
// 添加TBB支持以提升性能
#include <tbb/parallel_for.h>
#include <tbb/concurrent_vector.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/spin_mutex.h>
#include <atomic>

#include "utils.hpp"

namespace ICPRunner {

// SuperLoc特有的参数化 - 与原始SuperLoc的PoseLocalParameterization保持一致
    class SuperLocPoseParameterization : public ceres::LocalParameterization {
    public:
        virtual ~SuperLocPoseParameterization() {}

        virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;

        virtual bool ComputeJacobian(const double *x, double *jacobian) const;

        virtual int GlobalSize() const { return 7; }

        virtual int LocalSize() const { return 6; }
    };

// SuperLoc点到平面代价函数 - 与原始SuperLoc的SurfNormAnalyticCostFunction对应
    class SuperLocPlaneResidual : public ceres::SizedCostFunction<1, 7> {
    public:
        SuperLocPlaneResidual(const Eigen::Vector3d &src_point,
                              const Eigen::Vector3d &tgt_normal,
                              double d)
                : src_point_(src_point), tgt_normal_(tgt_normal), d_(d) {}

        virtual bool Evaluate(double const *const *parameters,
                              double *residuals,
                              double **jacobians) const;

    private:
        Eigen::Vector3d src_point_;
        Eigen::Vector3d tgt_normal_;
        double d_;
    };

// 使用自动微分的版本，用于对比测试
    struct SuperLocPlaneResidualAuto {
        SuperLocPlaneResidualAuto(const Eigen::Vector3d &src_point,
                                  const Eigen::Vector3d &tgt_normal,
                                  double d)
                : src_point_(src_point), tgt_normal_(tgt_normal), d_(d) {}

        template<typename T>
        bool operator()(const T *const pose, T *residuals) const {
            // 位姿参数: [x, y, z, qx, qy, qz, qw]
            Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(pose);
            Eigen::Quaternion <T> q(pose[6], pose[3], pose[4], pose[5]);

            // 变换源点
            Eigen::Matrix<T, 3, 1> src_pt_T = src_point_.cast<T>();
            Eigen::Matrix<T, 3, 1> p_trans = q * src_pt_T + t;

            // 计算残差
            Eigen::Matrix<T, 3, 1> normal_T = tgt_normal_.cast<T>();
            residuals[0] = normal_T.dot(p_trans) + T(d_);

            return true;
        }

        static ceres::CostFunction *Create(const Eigen::Vector3d &src_point,
                                           const Eigen::Vector3d &tgt_normal,
                                           double d) {
            return new ceres::AutoDiffCostFunction<SuperLocPlaneResidualAuto, 1, 7>(
                    new SuperLocPlaneResidualAuto(src_point, tgt_normal, d));
        }

    private:
        Eigen::Vector3d src_point_;
        Eigen::Vector3d tgt_normal_;
        double d_;
    };

// SuperLoc ICP集成类
    class SuperLocICP {
    public:
        // 特征可观测性枚举 - 与原始SuperLoc保持一致
        enum class FeatureObservability {
            rx_cross = 0,
            neg_rx_cross = 1,
            ry_cross = 2,
            neg_ry_cross = 3,
            rz_cross = 4,
            neg_rz_cross = 5,
            tx_dot = 6,
            ty_dot = 7,
            tz_dot = 8
        };

        struct SuperLocResult {
            bool converged;
            int iterations;
            double final_rmse;
            double final_fitness;

            // 退化信息
            Eigen::Matrix<double, 6, 6> covariance;
            Eigen::Matrix<double, 6, 1> degeneracy_mask;

            // 特征可观测性
            double uncertainty_x, uncertainty_y, uncertainty_z;
            double uncertainty_roll, uncertainty_pitch, uncertainty_yaw;

            // 特征可观测性直方图（用于深入分析）
            std::array<int, 9> feature_histogram;

            // 条件数
            double cond_full;
            double cond_rot;
            double cond_trans;

            bool isDegenerate;
        };

        static void
        runSuperLocICP(const std::string &method_name, Config &config_, TestResult &result, ICPContext &context,
                       const pcl::PointCloud<pcl::PointXYZI>::Ptr &source,
                       const pcl::PointCloud<pcl::PointXYZI>::Ptr &target);

        // 运行SuperLoc ICP
        static bool runSuperLocICPFull(
                const pcl::PointCloud<pcl::PointXYZI>::Ptr &source,
                const pcl::PointCloud<pcl::PointXYZI>::Ptr &target,
                const Eigen::Matrix4d &initial_guess,
                int max_iterations,
                double correspondence_distance,
                ICPContext &context,
                SuperLocResult &result,
                Eigen::Matrix4d &final_transform,
                std::vector <IterationLogData> &iteration_logs,
                double plane_resolution = 0.1  // 新增：平面分辨率参数，用于损失函数
        );


    private:


        // 新增：扩展的对应点查找函数，包含平面参数
        static void findCorrespondencesWithNormals(
                const pcl::PointCloud<pcl::PointXYZI>::Ptr &source,
                const pcl::PointCloud<pcl::PointXYZI>::Ptr &target,
                const Eigen::Matrix4d &transform,
                double max_distance,
                std::vector <std::pair<int, int>> &correspondences,
                std::vector <Eigen::Vector3d> &normals,
                std::vector<double> &plane_d_values,
                std::vector<double> &residual_coefficients,  // 新增：拟合质量系数
                double plane_resolution = 0.1  // 新增：平面分辨率参数
        );

        // 新增：详细的特征可观测性分析 - 与原始SuperLoc的FeatureObservabilityAnalysis对应
        static void analyzeFeatureObservabilityDetailed(
                const pcl::PointCloud<pcl::PointXYZI>::Ptr &source,
                const std::vector <std::pair<int, int>> &correspondences,
                const std::vector <Eigen::Vector3d> &normals,
                const Eigen::Matrix4d &transform,
                std::array<int, 9> &histogram
        );


        // 新增：基于直方图计算不确定性 - 与原始SuperLoc的EstimateLidarUncertainty对应
        static void computeUncertaintiesFromHistogram(
                const std::array<int, 9> &histogram,
                SuperLocResult &result
        );

        // 新增：估计配准误差 - 与原始SuperLoc的EstimateRegistrationError对应
        static void EstimateRegistrationError(
                ceres::Problem &problem,
                const double *pose_parameters,
                SuperLocResult &result
        );

        // 新增：检查退化状态
        static void checkDegeneracy(SuperLocResult &result);


    };

} // namespace ICPRunner

#endif //CLOUD_MAP_EVALUATION_SUPERLOC_H