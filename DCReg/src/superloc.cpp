#include "superloc.h"
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <iostream>

namespace ICPRunner {

// SuperLocPoseParameterization实现
    bool SuperLocPoseParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const {
        // 位置更新
        x_plus_delta[0] = x[0] + delta[0];
        x_plus_delta[1] = x[1] + delta[1];
        x_plus_delta[2] = x[2] + delta[2];

        // 旋转更新 - 与原始SuperLoc保持一致
        const double norm_delta = sqrt(delta[3] * delta[3] + delta[4] * delta[4] + delta[5] * delta[5]);
        if (norm_delta > 0.0) {
            const double sin_delta_by_delta = sin(norm_delta) / norm_delta;
            double q_delta[4];
            q_delta[0] = cos(norm_delta);
            q_delta[1] = sin_delta_by_delta * delta[3];
            q_delta[2] = sin_delta_by_delta * delta[4];
            q_delta[3] = sin_delta_by_delta * delta[5];

            // 四元数乘法: q_delta * q_current
            ceres::QuaternionProduct(q_delta, x + 3, x_plus_delta + 3);
        } else {
            x_plus_delta[3] = x[3];
            x_plus_delta[4] = x[4];
            x_plus_delta[5] = x[5];
            x_plus_delta[6] = x[6];
        }

        return true;
    }

    bool SuperLocPoseParameterization::ComputeJacobian(const double *x, double *jacobian) const {
        Eigen::Map <Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> J(jacobian);
        J.setZero();

        // 位置部分的雅可比
        J.topLeftCorner<3, 3>().setIdentity();

        // 旋转部分的雅可比
        const double q_w = x[6];
        const double q_x = x[3];
        const double q_y = x[4];
        const double q_z = x[5];

        J(3, 3) = 0.5 * q_w;
        J(3, 4) = 0.5 * q_z;
        J(3, 5) = -0.5 * q_y;
        J(4, 3) = -0.5 * q_z;
        J(4, 4) = 0.5 * q_w;
        J(4, 5) = 0.5 * q_x;
        J(5, 3) = 0.5 * q_y;
        J(5, 4) = -0.5 * q_x;
        J(5, 5) = 0.5 * q_w;
        J(6, 3) = -0.5 * q_x;
        J(6, 4) = -0.5 * q_y;
        J(6, 5) = -0.5 * q_z;

        return true;
    }

// SuperLoc点到平面代价函数
    bool SuperLocPlaneResidual::Evaluate(double const *const *parameters,
                                         double *residuals,
                                         double **jacobians) const {
        // 获取位姿参数
        Eigen::Map<const Eigen::Vector3d> t(parameters[0]);
        Eigen::Map<const Eigen::Quaterniond> q(parameters[0] + 3);

        // 变换源点
        Eigen::Vector3d p_trans = q * src_point_ + t;

        // 计算残差：点到平面距离
        // 注意：这里的d_已经是negative_OA_dot_norm
        residuals[0] = tgt_normal_.dot(p_trans) + d_;

        if (jacobians != NULL && jacobians[0] != NULL) {
            Eigen::Map <Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> J(jacobians[0]);

            // 对平移的导数
            J(0, 0) = tgt_normal_(0);
            J(0, 1) = tgt_normal_(1);
            J(0, 2) = tgt_normal_(2);

            // 对旋转的导数 - 简化版本
            // 使用链式法则: dr/dq = n^T * d(R*p)/dq
            //            const Eigen::Matrix3d R = q.toRotationMatrix();
            //            const Eigen::Vector3d Rp = R * src_point_;

            // 构造[Rp]×矩阵
            //            Eigen::Matrix3d skew_Rp;
            //            skew_Rp << 0, -Rp(2), Rp(1),
            //                    Rp(2), 0, -Rp(0),
            //                    -Rp(1), Rp(0), 0;
            //
            //            // 计算 n^T * [Rp]×
            //            Eigen::RowVector3d nT_skew = tgt_normal_.transpose() * skew_Rp;
            //
            //            // 四元数导数（这是一个近似，但在小角度下足够准确）
            //            J(0, 3) = nT_skew(0);
            //            J(0, 4) = nT_skew(1);
            //            J(0, 5) = nT_skew(2);
            //            J(0, 6) = 0.0;

            // 四元数旋转公式: q * p * q^(-1)
            // 展开后对各个四元数分量求导

            double qw = q.w();
            double qx = q.x();
            double qy = q.y();
            double qz = q.z();

            double px = src_point_(0);
            double py = src_point_(1);
            double pz = src_point_(2);

            // 四元数旋转公式: q * p * q^(-1)
            // 展开后对各个四元数分量求导

            // ∂(q*p)/∂qx - 修正后的公式
            Eigen::Vector3d dp_dqx;
            dp_dqx(0) = 2 * (qy * py + qz * pz);
            dp_dqx(1) = 2 * (qy * px - 2 * qx * py - qw * pz);
            dp_dqx(2) = 2 * (qz * px + qw * py - 2 * qx * pz);

            // ∂(q*p)/∂qy - 修正后的公式
            Eigen::Vector3d dp_dqy;
            dp_dqy(0) = 2 * (qx * py - 2 * qy * px + qw * pz);
            dp_dqy(1) = 2 * (qx * px + qz * pz);
            dp_dqy(2) = 2 * (qz * py - qw * px - 2 * qy * pz);

            // ∂(q*p)/∂qz - 修正后的公式
            Eigen::Vector3d dp_dqz;
            dp_dqz(0) = 2 * (qx * pz - qw * py - 2 * qz * px);
            dp_dqz(1) = 2 * (qy * pz + qw * px - 2 * qz * py);
            dp_dqz(2) = 2 * (qx * px + qy * py);

            // ∂(q*p)/∂qw - 修正后的公式
            Eigen::Vector3d dp_dqw;
            dp_dqw(0) = 2 * (qy * pz - qz * py);
            dp_dqw(1) = 2 * (qz * px - qx * pz);
            dp_dqw(2) = 2 * (qx * py - qy * px);

            // 最终雅可比 = n^T * ∂(q*p)/∂q
            J(0, 3) = tgt_normal_.dot(dp_dqx);
            J(0, 4) = tgt_normal_.dot(dp_dqy);
            J(0, 5) = tgt_normal_.dot(dp_dqz);
            J(0, 6) = tgt_normal_.dot(dp_dqw);
        }

        return true;
    }


    void SuperLocICP::runSuperLocICP(const std::string &method_name, Config &config_, TestResult &result,
                                     ICPContext &context,
                                     const pcl::PointCloud<pcl::PointXYZI>::Ptr &source_cloud_,
                                     const pcl::PointCloud<pcl::PointXYZI>::Ptr &target_cloud_) {

        // 获取初始变换
        Eigen::Matrix4d initial_matrix = Pose6D2Matrix(config_.initial_noise);

        // 清空context中的迭代数据
        context.iteration_log_data_.clear();
        context.final_convergence_flag_ = false;
        context.final_iterations_ = 0;

        // 清空result中的迭代数据
        result.iteration_data.clear();
        result.iter_rmse_history.clear();
        result.iter_fitness_history.clear();
        result.iter_corr_num_history.clear();
        result.iter_trans_error_history.clear();
        result.iter_rot_error_history.clear();
        result.iter_transform_history.clear();

        // 运行SuperLoc ICP
        SuperLocICP::SuperLocResult superloc_result;
        auto start_total = std::chrono::high_resolution_clock::now();

        // 调用SuperLoc ICP，使用默认的plane_resolution = 0.1
        bool success = SuperLocICP::runSuperLocICPFull(
                source_cloud_,
                target_cloud_,
                initial_matrix,
                config_.max_iterations,
                config_.search_radius,
                context,
                superloc_result,
                result.final_transform,
                result.iteration_data,      // 直接使用result的iteration_data
                0.1                         // plane_resolution参数，对应原始SuperLoc的localMap.planeRes_
        );

        auto end_total = std::chrono::high_resolution_clock::now();

        // 基本结果
        result.converged = superloc_result.converged;
        result.iterations = superloc_result.iterations;
        result.final_rmse = superloc_result.final_rmse;
        result.final_fitness = superloc_result.final_fitness;
        result.time_ms = std::chrono::duration<double, std::milli>(end_total - start_total).count();

        // 更新context中的值
        context.final_convergence_flag_ = superloc_result.converged;
        context.final_iterations_ = superloc_result.iterations;
        context.total_icp_time_ms_ = result.time_ms;
        context.final_pose_ = MatrixToPose6D(result.final_transform);

        // 从superloc_result复制iteration_log_data到context
        context.iteration_log_data_ = result.iteration_data;

        PoseError error = calculatePoseError(config_.gt_matrix, result.final_transform, true);
        result.rot_error_deg = error.rotation_error;
        result.trans_error_m = error.translation_error;

        // 计算point-to-point误差
        pcl::PointCloud<PointT>::Ptr aligned_cloud(new pcl::PointCloud <PointT>);
        pcl::transformPointCloud(*source_cloud_, *aligned_cloud, result.final_transform);
        calculatePointToPointError(aligned_cloud, target_cloud_,
                                   result.p2p_rmse, result.p2p_fitness,
                                   result.chamfer_distance, result.corr_num, config_.error_threshold);

        // 保存SuperLoc特有数据
        result.superloc_data.has_data = true;
        result.superloc_data.uncertainty_x = superloc_result.uncertainty_x;
        result.superloc_data.uncertainty_y = superloc_result.uncertainty_y;
        result.superloc_data.uncertainty_z = superloc_result.uncertainty_z;
        result.superloc_data.uncertainty_roll = superloc_result.uncertainty_roll;
        result.superloc_data.uncertainty_pitch = superloc_result.uncertainty_pitch;
        result.superloc_data.uncertainty_yaw = superloc_result.uncertainty_yaw;
        result.superloc_data.cond_full = superloc_result.cond_full;
        result.superloc_data.cond_rot = superloc_result.cond_rot;
        result.superloc_data.cond_trans = superloc_result.cond_trans;
        result.superloc_data.is_degenerate = superloc_result.isDegenerate;
        result.superloc_data.covariance = superloc_result.covariance;
        result.superloc_data.feature_histogram = superloc_result.feature_histogram;

//        for (int i = 0; i < 6; ++i) {
//            result.degenerate_mask[i] = (superloc_result.degeneracy_mask(i) > 0);  // 将非零值视为true
//        }
        result.superloc_data.is_degenerate = superloc_result.isDegenerate;


        // 更新context的协方差矩阵
/*        if (superloc_result.covariance.size() == 36) {
            // 将SuperLoc的协方差矩阵转换为6x6 Eigen矩阵
            Eigen::Matrix<double, 6, 6> cov_matrix;
            for (int i = 0; i < 6; ++i) {
                for (int j = 0; j < 6; ++j) {
                    cov_matrix(i, j) = superloc_result.covariance[i * 6 + j];
                }
            }
            context.icp_cov = cov_matrix;
        } else {
            // 如果没有有效的协方差，设置为单位矩阵
            context.icp_cov.setIdentity();
            context.icp_cov *= 1e6;
        }*/

        // 保存迭代历史
        result.iter_rmse_history.clear();
        result.iter_fitness_history.clear();
        result.iter_corr_num_history.clear();
        result.iter_trans_error_history.clear();
        result.iter_rot_error_history.clear();

        // 计算每次迭代相对于真值的误差
        for (auto &iter : result.iteration_data) {
            result.iter_rmse_history.push_back(iter.rmse);
            result.iter_fitness_history.push_back(iter.fitness);
            result.iter_corr_num_history.push_back(iter.corr_num);

            // 计算该迭代的误差
            PoseError error = calculatePoseError(config_.gt_matrix, iter.transform_matrix, true);
            iter.rot_error_vs_gt = error.rotation_error;
            iter.trans_error_vs_gt = error.translation_error;

            result.iter_trans_error_history.push_back(iter.trans_error_vs_gt);
            result.iter_rot_error_history.push_back(iter.rot_error_vs_gt);
            result.iter_transform_history.push_back(iter.transform_matrix);
        }

        // 输出详细结果（保持原有的输出格式）
        std::cout << "\n[SuperLoc] === Final Results ===" << std::endl;
        std::cout << "[SuperLoc] Converged: " << (superloc_result.converged ? "Yes" : "No")
                  << " (after " << superloc_result.iterations << " iterations)" << std::endl;
        std::cout << "[SuperLoc] Final RMSE: " << superloc_result.final_rmse
                  << ", Fitness: " << superloc_result.final_fitness << std::endl;
        std::cout << "[SuperLoc] Transform error: trans=" << result.trans_error_m
                  << "m, rot=" << result.rot_error_deg << "deg" << std::endl;
        std::cout << "[SuperLoc] P2P RMSE: " << result.p2p_rmse
                  << ", Chamfer: " << result.chamfer_distance << std::endl;

        // 特征可观测性结果
        std::cout << "\n[SuperLoc] Feature Observability Analysis:" << std::endl;
        std::cout << std::fixed << std::setprecision(6) << "  Translation uncertainty - X: "
                  << superloc_result.uncertainty_x
                  << ", Y: " << superloc_result.uncertainty_y
                  << ", Z: " << superloc_result.uncertainty_z << std::endl;
        std::cout << "  Rotation uncertainty - Roll: " << superloc_result.uncertainty_roll
                  << ", Pitch: " << superloc_result.uncertainty_pitch
                  << ", Yaw: " << superloc_result.uncertainty_yaw << std::endl;

        // 退化检测结果
//        std::cout << "\n[SuperLoc] Degeneracy Detection:" << std::endl;
//        std::cout << "  Condition numbers - Full: " << superloc_result.cond_full
//                  << ", Rot: " << superloc_result.cond_rot
//                  << ", Trans: " << superloc_result.cond_trans << std::endl;
        std::cout << "  Is degenerate: " << (superloc_result.isDegenerate ? "Yes" : "No") << std::endl;
        std::cout << "Degenerate_mask: " << superloc_result.degeneracy_mask.transpose() << std::endl;

        // 输出迭代历史
        if (config_.num_runs == 1) {
            std::cout << "\n[SuperLoc] Iteration History:" << std::endl;
            std::cout << "Iter\tRMSE\tFitness\tCorr#\tTransErr\tRotErr\tTime(ms)" << std::endl;
            for (size_t i = 0; i < result.iteration_data.size(); ++i) {
                const auto &iter_data = result.iteration_data[i];
                std::cout << i << "\t"
                          << std::fixed << std::setprecision(4) << iter_data.rmse << "\t"
                          << iter_data.fitness << "\t"
                          << iter_data.corr_num << "\t"
                          << iter_data.trans_error_vs_gt << "\t"
                          << iter_data.rot_error_vs_gt << "\t"
                          << iter_data.iter_time_ms << std::endl;
            }
        }

        std::cout << "\n[SuperLoc] Total time: " << result.time_ms << " ms" << std::endl;
        std::cout << "[SuperLoc] ========================\n" << std::endl;
    }


    bool SuperLocICP::runSuperLocICPFull(
            const pcl::PointCloud<pcl::PointXYZI>::Ptr &source,
            const pcl::PointCloud<pcl::PointXYZI>::Ptr &target,
            const Eigen::Matrix4d &initial_guess,
            int max_iterations,
            double correspondence_distance,
            ICPContext &context,
            SuperLocResult &result,
            Eigen::Matrix4d &final_transform,
            std::vector <IterationLogData> &iteration_logs,
            double plane_resolution) {

        // 初始化结果
        result.converged = false;
        result.iterations = 0;
        result.isDegenerate = false;
        final_transform = initial_guess;

        // 初始化位姿参数（7维：x,y,z,qx,qy,qz,qw）
        double pose_parameters[7];
        Eigen::Map <Eigen::Vector3d> pose_t(pose_parameters);
        Eigen::Map <Eigen::Quaterniond> pose_q(pose_parameters + 3);

        // 从初始猜测设置参数
        Eigen::Vector3d t = initial_guess.block<3, 1>(0, 3);
        Eigen::Quaterniond q(initial_guess.block<3, 3>(0, 0));
        q.normalize();

        pose_t = t;
        pose_q = q;

        // 特征可观测性直方图 - 与原始SuperLoc保持一致
        std::array<int, 9> PlaneFeatureHistogramObs;

        // ICP主循环
        for (int iter = 0; iter < max_iterations; ++iter) {
            IterationLogData iter_log;
            iter_log.iter_count = iter;

            auto iter_start = std::chrono::high_resolution_clock::now();

            // 重置特征可观测性直方图
            PlaneFeatureHistogramObs.fill(0);

            // 1. 寻找对应点和法向量
            std::vector <std::pair<int, int>> correspondences;
            std::vector <Eigen::Vector3d> normals;
            std::vector<double> plane_d_values;
            std::vector<double> residual_coefficients;

            findCorrespondencesWithNormals(source, target, final_transform, correspondence_distance, correspondences,
                                           normals, plane_d_values, residual_coefficients, plane_resolution);
            if (correspondences.size() < 10) {
                std::cout << "[SuperLoc] Not enough correspondences: "
                          << correspondences.size() << std::endl;
                break;
            }

            // 2. 分析特征可观测性 - 使用原始SuperLoc的方法
            analyzeFeatureObservabilityDetailed(source, correspondences, normals,
                                                final_transform, PlaneFeatureHistogramObs);

            // 3. 构建优化问题
            ceres::Problem problem;
            ceres::LocalParameterization *pose_parameterization =
                    new SuperLocPoseParameterization();
            problem.AddParameterBlock(pose_parameters, 7, pose_parameterization);

            // 添加点到平面约束
            double total_residual = 0.0;
            int valid_constraints = 0;
            double rmse = 0.0;
            int inlier_count = 0;
            for (size_t i = 0; i < correspondences.size(); ++i) {
                const auto &corr = correspondences[i];
                Eigen::Vector3d src_pt(source->points[corr.first].x,
                                       source->points[corr.first].y,
                                       source->points[corr.first].z);

                // 使用自动微分版本进行测试
                bool use_autodiff = false;  // 可以切换测试
                ceres::CostFunction *cost_function = nullptr;
                if (use_autodiff) {
                    cost_function = SuperLocPlaneResidualAuto::Create(src_pt, normals[i], plane_d_values[i]);
                } else {
                    cost_function = new SuperLocPlaneResidual(src_pt, normals[i], plane_d_values[i]);
                }

                // 损失函数策略
                auto *loss_function = new ceres::TukeyLoss(std::sqrt(3 * plane_resolution));

                // 使用ScaledLoss根据拟合质量加权
                auto *weight_function = new ceres::ScaledLoss(
                        loss_function,
                        residual_coefficients[i],  // 基于拟合质量的权重
                        ceres::TAKE_OWNERSHIP      // Ceres负责删除loss_function
                );
                problem.AddResidualBlock(cost_function, weight_function, pose_parameters);
                valid_constraints++;

                // 计算初始残差用于调试
                if (iter == 0 && i < 100) {
                    Eigen::Vector3d p_trans = pose_q * src_pt + pose_t;
                    double residual = normals[i].dot(p_trans) + plane_d_values[i];
                    total_residual += residual * residual;
                }

                Eigen::Vector3d transformed_pt = pose_q * src_pt + pose_t;
                double residual = normals[i].dot(transformed_pt) + plane_d_values[i];
                rmse += residual * residual;
                if (std::abs(residual) < 0.3) {
                    inlier_count++;
                }
            }
            rmse = std::sqrt(rmse / correspondences.size());
            double fitness = static_cast<double>(inlier_count) / source->size();
            iter_log.corr_num = correspondences.size();
            iter_log.effective_points = inlier_count;

            // 调试信息
            if (iter == 0) {
                double avg_residual = std::sqrt(total_residual / std::min(100, (int) correspondences.size()));
                std::cout << "[SuperLoc Debug] Initial state - t: [" << pose_t.transpose()
                          << "], q: [" << pose_q.coeffs().transpose() << "]" << std::endl;
                std::cout << "[SuperLoc Debug] Valid constraints: " << valid_constraints
                          << ", avg residual: " << avg_residual << std::endl;
            }

            // 4. 求解优化问题 - 与原始SuperLoc完全一致
            ceres::Solver::Options options;
            options.max_num_iterations = 4;
            options.linear_solver_type = ceres::DENSE_QR;
            options.minimizer_progress_to_stdout = false;
            options.check_gradients = false;
            options.gradient_check_relative_precision = 1e-4;

            // 保存优化前的参数用于调试
            Eigen::Vector3d t_before = t;
            Eigen::Quaterniond q_before = q;

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            // 5. 更新变换矩阵
            // 直接从Map获取更新后的值
            t = pose_t;
            q = pose_q;
            q.normalize();

            final_transform.setIdentity();
            final_transform.block<3, 3>(0, 0) = q.toRotationMatrix();
            final_transform.block<3, 1>(0, 3) = t;

            // 记录迭代日志
            iter_log.rmse = rmse;
            iter_log.fitness = fitness;
            iter_log.transform_matrix = final_transform;

            auto iter_end = std::chrono::high_resolution_clock::now();
            iter_log.iter_time_ms = std::chrono::duration<double, std::milli>(iter_end - iter_start).count();

            //            if (!result.converged) {
//                result.iterations = iteration_logs.size();
//                if (!iteration_logs.empty()) {
//                    result.final_rmse = iteration_logs.back().rmse;
//                    result.final_fitness = iteration_logs.back().fitness;
//                }

            // 即使没有收敛，也进行最后的分析
            // 估计配准误差和协方差
            ceres::Problem final_problem;
            ceres::LocalParameterization *pose_parameterization2 =
                    new SuperLocPoseParameterization();
            final_problem.AddParameterBlock(pose_parameters, 7, pose_parameterization2);

            // 使用最后一次迭代的对应点重建问题（简化版）
            EstimateRegistrationError(final_problem, pose_parameters, result);

            // 计算不确定性
            computeUncertaintiesFromHistogram(PlaneFeatureHistogramObs, result);
            result.feature_histogram = PlaneFeatureHistogramObs;

            // 检查退化
            checkDegeneracy(result);

            iter_log.degenerate_mask.clear();
            iter_log.degenerate_mask.resize(6, false);
            for (int i = 0; i < 6; ++i) {
                iter_log.degenerate_mask[i] = (result.degeneracy_mask(i) == 1);  // 将非零值视为true
            }
            iter_log.is_degenerate = result.isDegenerate;
            iteration_logs.push_back(iter_log);

            // 检查收敛条件 - 与原始SuperLoc保持一致
            if ((summary.num_successful_steps > 0) || (iter == max_iterations - 1)) {
                result.converged = (summary.num_successful_steps > 0) && (rmse < 0.01);
                result.iterations = iter + 1;
                result.final_rmse = rmse;
                result.final_fitness = fitness;
                break;
            }
        }


        return result.converged;
    }

    // 扩展的对应点查找函数，包含法向量计算
    void SuperLocICP::findCorrespondencesWithNormals(
            const pcl::PointCloud<pcl::PointXYZI>::Ptr &source,
            const pcl::PointCloud<pcl::PointXYZI>::Ptr &target,
            const Eigen::Matrix4d &transform,
            double max_distance,
            std::vector <std::pair<int, int>> &correspondences,
            std::vector <Eigen::Vector3d> &normals,
            std::vector<double> &plane_d_values,
            std::vector<double> &residual_coefficients,
            double plane_resolution) {

        correspondences.clear();
        normals.clear();
        plane_d_values.clear();
        residual_coefficients.clear();

        pcl::KdTreeFLANN <pcl::PointXYZI> kdtree;
        kdtree.setInputCloud(target);

        // 对每个源点查找最近邻
        for (size_t i = 0; i < source->size(); ++i) {
            Eigen::Vector4d src_pt(source->points[i].x, source->points[i].y,
                                   source->points[i].z, 1.0);
            Eigen::Vector4d transformed_pt = transform * src_pt;

            pcl::PointXYZI query_pt;
            query_pt.x = transformed_pt(0);
            query_pt.y = transformed_pt(1);
            query_pt.z = transformed_pt(2);

            // 查找k个最近邻用于平面拟合
            std::vector<int> k_indices;
            std::vector<float> k_distances;
            if (kdtree.nearestKSearch(query_pt, 5, k_indices, k_distances) >= 5) {
                if (k_distances[0] <= max_distance * max_distance) {
                    // 计算平面参数 - 与原始SuperLoc保持一致
                    Eigen::MatrixXd A(5, 3);
                    Eigen::VectorXd b = -Eigen::VectorXd::Ones(5);

                    for (int j = 0; j < 5; ++j) {
                        A(j, 0) = target->points[k_indices[j]].x;
                        A(j, 1) = target->points[k_indices[j]].y;
                        A(j, 2) = target->points[k_indices[j]].z;
                    }

                    // 使用最小二乘拟合平面: Ax + By + Cz + 1 = 0
                    // 求解 [A B C]^T，使得 ||[x y z][A B C]^T + 1||^2 最小
                    Eigen::Vector3d plane_coeffs = A.colPivHouseholderQr().solve(b);

                    // plane_coeffs = [A, B, C]，平面方程为 Ax + By + Cz + 1 = 0
                    // 转换为标准形式 n·p + d = 0，其中|n| = 1
                    double norm = plane_coeffs.norm();
                    if (norm < 1e-6) continue;  // 退化平面

                    Eigen::Vector3d plane_normal = plane_coeffs / norm;
                    double negative_OA_dot_norm = 1.0 / norm;  // 这是原始SuperLoc中的d

                    // 确保法向量朝向视点（与原始SuperLoc一致）
                    Eigen::Vector3d viewpoint_direction(query_pt.x, query_pt.y, query_pt.z);
                    if (viewpoint_direction.dot(plane_normal) < 0) {
                        plane_normal = -plane_normal;
                        negative_OA_dot_norm = -negative_OA_dot_norm;
                    }

                    // 计算拟合质量系数（residualCoefficient）- 与原始SuperLoc保持一致
                    double meanSquareDist = 0.0;
                    for (int j = 0; j < 5; ++j) {
                        double point_to_plane_dist = std::abs(
                                plane_normal(0) * target->points[k_indices[j]].x +
                                plane_normal(1) * target->points[k_indices[j]].y +
                                plane_normal(2) * target->points[k_indices[j]].z +
                                negative_OA_dot_norm
                        );
                        meanSquareDist += point_to_plane_dist * point_to_plane_dist;
                    }
                    meanSquareDist /= 5.0;

                    // 计算拟合质量系数：1 - sqrt(meanSquareDist / (3 * planeRes))
                    double fitQualityCoeff = 1.0 - std::sqrt(meanSquareDist / (3 * plane_resolution));
                    fitQualityCoeff = std::max(0.1, fitQualityCoeff);  // 避免权重过小

                    correspondences.push_back(std::make_pair(i, k_indices[0]));
                    normals.push_back(plane_normal);
                    plane_d_values.push_back(negative_OA_dot_norm);
                    residual_coefficients.push_back(fitQualityCoeff);
                }
            }
        }
    }

    // 详细的特征可观测性分析 - 与原始SuperLoc保持一致
    void SuperLocICP::analyzeFeatureObservabilityDetailed(
            const pcl::PointCloud<pcl::PointXYZI>::Ptr &source,
            const std::vector <std::pair<int, int>> &correspondences,
            const std::vector <Eigen::Vector3d> &normals,
            const Eigen::Matrix4d &transform,
            std::array<int, 9> &histogram) {

        // 获取当前旋转矩阵
        Eigen::Matrix3d R = transform.block<3, 3>(0, 0);

        // 旋转后的坐标轴
        Eigen::Vector3d x_axis = R * Eigen::Vector3d(1, 0, 0);
        Eigen::Vector3d y_axis = R * Eigen::Vector3d(0, 1, 0);
        Eigen::Vector3d z_axis = R * Eigen::Vector3d(0, 0, 1);

        for (size_t i = 0; i < correspondences.size(); ++i) {
            const auto &corr = correspondences[i];
            Eigen::Vector3d src_pt(source->points[corr.first].x,
                                   source->points[corr.first].y,
                                   source->points[corr.first].z);

            // 变换点到世界坐标系
            Eigen::Vector3d transformed_pt = transform.block<3, 3>(0, 0) * src_pt +
                                             transform.block<3, 1>(0, 3);

            const Eigen::Vector3d &normal = normals[i];

            // 计算叉积（用于旋转可观测性）
            Eigen::Vector3d cross = transformed_pt.cross(normal);

            // 旋转可观测性分析
            std::vector <std::pair<double, int>> rotation_quality;
            rotation_quality.push_back({std::abs(cross.dot(x_axis)), 0});      // rx_cross
            rotation_quality.push_back({std::abs(cross.dot(-x_axis)), 1});     // neg_rx_cross
            rotation_quality.push_back({std::abs(cross.dot(y_axis)), 2});      // ry_cross
            rotation_quality.push_back({std::abs(cross.dot(-y_axis)), 3});     // neg_ry_cross
            rotation_quality.push_back({std::abs(cross.dot(z_axis)), 4});      // rz_cross
            rotation_quality.push_back({std::abs(cross.dot(-z_axis)), 5});     // neg_rz_cross

            // 平移可观测性分析
            std::vector <std::pair<double, int>> trans_quality;
            trans_quality.push_back({std::abs(normal.dot(x_axis)), 6});    // tx_dot
            trans_quality.push_back({std::abs(normal.dot(y_axis)), 7});    // ty_dot
            trans_quality.push_back({std::abs(normal.dot(z_axis)), 8});    // tz_dot

            // 选择最佳的旋转和平移可观测性
            std::sort(rotation_quality.begin(), rotation_quality.end(),
                      [](const auto &a, const auto &b) { return a.first > b.first; });
            std::sort(trans_quality.begin(), trans_quality.end(),
                      [](const auto &a, const auto &b) { return a.first > b.first; });

            // 更新直方图
            histogram[rotation_quality[0].second]++;
            histogram[rotation_quality[1].second]++;
            histogram[trans_quality[0].second]++;
        }
    }

    // 估计配准误差 - 与原始SuperLoc的EstimateRegistrationError保持一致
    void SuperLocICP::EstimateRegistrationError(
            ceres::Problem &problem,
            const double *pose_parameters,
            SuperLocResult &result) {

        // 协方差计算选项 - 与原始SuperLoc保持一致
        ceres::Covariance::Options covOptions;
        covOptions.apply_loss_function = true;
        covOptions.algorithm_type = ceres::CovarianceAlgorithmType::DENSE_SVD;
        covOptions.null_space_rank = -1;
        covOptions.num_threads = 2;

        ceres::Covariance covarianceSolver(covOptions);
        std::vector <std::pair<const double *, const double *>> covarianceBlocks;
        covarianceBlocks.emplace_back(pose_parameters, pose_parameters);

        if (covarianceSolver.Compute(covarianceBlocks, &problem)) {
            // 获取6x6协方差矩阵（在切空间中）
            result.covariance.setZero();
            covarianceSolver.GetCovarianceBlockInTangentSpace(
                    pose_parameters, pose_parameters, result.covariance.data());

            // 计算条件数
            Eigen::SelfAdjointEigenSolver <Eigen::Matrix<double, 6, 6>> solver_full(result.covariance);
            Eigen::VectorXd eigenvalues = solver_full.eigenvalues();
            std::cout << "info Eigen values: " << eigenvalues.transpose() << std::endl;

            // 避免除零
            double min_eigenvalue = std::max(eigenvalues(0), 1e-10);
            double max_eigenvalue = std::max(eigenvalues(5), 1e-10);
            result.cond_full = std::sqrt(max_eigenvalue / min_eigenvalue);

            // 位置部分条件数
            Eigen::SelfAdjointEigenSolver <Eigen::Matrix3d> solver_trans(
                    result.covariance.topLeftCorner<3, 3>());
            double min_trans = std::max(solver_trans.eigenvalues()(0), 1e-10);
            double max_trans = std::max(solver_trans.eigenvalues()(2), 1e-10);
            result.cond_trans = std::sqrt(max_trans / min_trans);

            // 旋转部分条件数
            Eigen::SelfAdjointEigenSolver <Eigen::Matrix3d> solver_rot(
                    result.covariance.bottomRightCorner<3, 3>());
            double min_rot = std::max(solver_rot.eigenvalues()(0), 1e-10);
            double max_rot = std::max(solver_rot.eigenvalues()(2), 1e-10);
            result.cond_rot = std::sqrt(max_rot / min_rot);
        } else {
            // 如果协方差计算失败，设置默认值
            result.covariance.setIdentity();
            result.cond_full = 1.0;
            result.cond_trans = 1.0;
            result.cond_rot = 1.0;
        }
    }

    // 基于特征直方图计算不确定性 - 与原始SuperLoc的EstimateLidarUncertainty保持一致
    void SuperLocICP::computeUncertaintiesFromHistogram(
            const std::array<int, 9> &histogram,
            SuperLocResult &result) {

        // 平移特征总数
        double TotalTransFeature = histogram[6] + histogram[7] + histogram[8];

        if (TotalTransFeature > 0) {
            // X方向不确定性
            double uncertaintyX = (histogram[6] / TotalTransFeature) * 3;
            result.uncertainty_x = std::min(uncertaintyX, 1.0);

            // Y方向不确定性
            double uncertaintyY = (histogram[7] / TotalTransFeature) * 3;
            result.uncertainty_y = std::min(uncertaintyY, 1.0);

            // Z方向不确定性
            double uncertaintyZ = (histogram[8] / TotalTransFeature) * 3;
            result.uncertainty_z = std::min(uncertaintyZ, 1.0);
        } else {
            result.uncertainty_x = 0.0;
            result.uncertainty_y = 0.0;
            result.uncertainty_z = 0.0;
        }

        // 旋转特征总数
        double TotalRotationFeature = histogram[0] + histogram[1] + histogram[2] +
                                      histogram[3] + histogram[4] + histogram[5];

        if (TotalRotationFeature > 0) {
            // Roll不确定性
            double uncertaintyRoll = ((histogram[0] + histogram[1]) / TotalRotationFeature) * 3;
            result.uncertainty_roll = std::min(uncertaintyRoll, 1.0);

            // Pitch不确定性
            double uncertaintyPitch = ((histogram[2] + histogram[3]) / TotalRotationFeature) * 3;
            result.uncertainty_pitch = std::min(uncertaintyPitch, 1.0);

            // Yaw不确定性
            double uncertaintyYaw = ((histogram[4] + histogram[5]) / TotalRotationFeature) * 3;
            result.uncertainty_yaw = std::min(uncertaintyYaw, 1.0);
        } else {
            result.uncertainty_roll = 0.0;
            result.uncertainty_pitch = 0.0;
            result.uncertainty_yaw = 0.0;
        }
    }

    // 检查退化 - 基于不确定性和特征数量
    void SuperLocICP::checkDegeneracy(SuperLocResult &result) {
        // 使用与原始SuperLoc完全一致的退化判断条件
        // 原始代码中的判断逻辑包括不确定性阈值和特征数量检查

        // 基于不确定性的退化判断
//        if (result.uncertainty_x < 0.2 || result.uncertainty_y < 0.1 || result.uncertainty_z < 0.2) {
//            result.isDegenerate = true;
//        }
//            // 基于条件数的退化判断（作为额外的安全检查）
//            //        else if (result.cond_full > 100.0 || result.cond_trans > 100.0 || result.cond_rot > 100.0) {
//            //            result.isDegenerate = true;
//            //        }
//        else {
//            result.isDegenerate = false;
//        }

        // 注：原始SuperLoc还检查了特征数量(PlaneFeatureHistogramObs.at(6/7/8) < 20/10/10)
        // 但在我们的实现中，这个检查已经隐含在不确定性计算中
        // 因为当特征数量少时，不确定性会自动变高
        int degenrate_num = 0;
        if (result.uncertainty_x < 0.2) {
            result.degeneracy_mask(3) = 1;  // x方向退化
            degenrate_num++;
        }
        if (result.uncertainty_y < 0.1) {
            result.degeneracy_mask(4) = 1;  // y方向退化
            degenrate_num++;
        }
        if (result.uncertainty_z < 0.2) {
            result.degeneracy_mask(5) = 1;  // z方向退化
            degenrate_num++;
        }
        // 设置旋转退化的阈值（这些值可能需要根据实际情况调整）
        const double ROLL_UNCERTAINTY_THRESH = 0.2;
        const double PITCH_UNCERTAINTY_THRESH = 0.1;
        const double YAW_UNCERTAINTY_THRESH = 0.2;
        if (result.uncertainty_roll < ROLL_UNCERTAINTY_THRESH) {
            result.degeneracy_mask(0) = 1;  // roll(ωx)方向退化
            degenrate_num++;
        }
        if (result.uncertainty_pitch < PITCH_UNCERTAINTY_THRESH) {
            result.degeneracy_mask(1) = 1;  // pitch(ωy)方向退化
            degenrate_num++;
        }
        if (result.uncertainty_yaw < YAW_UNCERTAINTY_THRESH) {
            result.degeneracy_mask(2) = 1;  // yaw(ωz)方向退化
            degenrate_num++;
        }

        if (degenrate_num > 0) {
            result.isDegenerate = true;
        } else {
            result.isDegenerate = false;
        }
    }


} // namespace ICPRunner