//
// Created by xchu on 12/6/2025.
//

#ifndef CLOUD_MAP_EVALUATION_UTILS_H
#define CLOUD_MAP_EVALUATION_UTILS_H


#include <fstream>
#include <chrono>
#include <map>
#include <vector>
#include <string>
#include <filesystem>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/angles.h> // for pcl::rad2deg, pcl::deg2rad
#include <pcl/io/pcd_io.h>
#include <pcl/common/distances.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/normal_3d.h>


// *** 新增: Open3D 包含 ***
#include <open3d/Open3D.h> // 包含 Open3D 主头文件
#include <open3d/pipelines/registration/Registration.h> // 包含 ICP 相关头文件


// Define PointT if not defined elsewhere
// typedef pcl::PointXYZI PointT; // Example: Use PointXYZI
// Or use the one from the original code if it's different and accessible
using PointT = pcl::PointXYZI; // Using PointXYZ as a common default
using namespace Eigen;
namespace fs = std::filesystem;

namespace ICPRunner {

    // Pose Representation (same as before)
    struct Pose6D {
        double roll = 0.0, pitch = 0.0, yaw = 0.0;
        double x = 0.0, y = 0.0, z = 0.0;

        Pose6D operator+(const Pose6D &other) const {
            Pose6D result;
            result.roll = roll + other.roll;
            result.pitch = pitch + other.pitch;
            result.yaw = yaw + other.yaw;
            result.x = x + other.x;
            result.y = y + other.y;
            result.z = z + other.z;
            return result;
        }
    };

    struct PoseError {
        double translation_error;  // in meters
        double rotation_error;     // in radians or degrees based on use_degrees flag

        PoseError() : translation_error(0.0), rotation_error(0.0) {}

        // Utility function for printing pose errors
        void printPoseError() const {
            bool is_degrees = true;
            std::cout << "Translation error: " << this->translation_error << " m, ";
            std::cout << "Rotation error: " << this->rotation_error
                      << (is_degrees ? " deg" : " rad") << std::endl;
        }
    };

    // Tunable Parameters - will be updated from config
    struct ICPParameters {
        double DEGENERACY_THRES_COND = 10.0;
        double DEGENERACY_THRES_EIG = 120.0;
        double KAPPA_TARGET = 1.0;
        double PCG_TOLERANCE = 1e-6;
        int PCG_MAX_ITER = 10;
        double ADAPTIVE_REG_ALPHA = 10.0;
        double STD_REG_GAMMA = 0.01;
        double LOAM_EIGEN_THRESH = 120.0;
        double TSVD_SINGULAR_THRESH = 120.0;

        // X-ICP specific parameters
        double XICP_ENOUGH_INFO_THRESHOLD = 100.0;
        double XICP_INSUFFICIENT_INFO_THRESHOLD = 10.0;
        double XICP_HIGH_INFO_THRESHOLD = 1000.0;
        double XICP_SOLUTION_REMAPPING_THRESHOLD = 120.0;
        double XICP_MINIMAL_ALIGNMENT_ANGLE = 30.0;  // degrees
        double XICP_STRONG_ALIGNMENT_ANGLE = 15.0;   // degrees
        double XICP_INEQUALITY_BOUND_MULTIPLIER = 1.0;

        bool XICP_DEBUG = false;
    };

    // Degeneracy Detection Methods
    enum class DetectionMethod {
        NONE_DETE, SCHUR_CONDITION_NUMBER, FULL_EVD_MIN_EIGENVALUE,
        EVD_SUB_CONDITION, FULL_SVD_CONDITION, O3D, SUPERLOC,
        XICP_OPTIMIZED_EQUALITY,     // kOptimizedEqualityConstraints
        XICP_INEQUALITY,            // kInequalityConstraints
        XICP_EQUALITY,              // kEqualityConstraints
        XICP_SOLUTION_REMAPPING     // kSolutionRemapping
    };

// Degeneracy Handling Methods
    enum class HandlingMethod {
        NONE_HAND, STANDARD_REGULARIZATION, ADAPTIVE_REGULARIZATION,
        PRECONDITIONED_CG, SOLUTION_REMAPPING, TRUNCATED_SVD, O3D, SUPERLOC,
        XICP_CONSTRAINT,            // XICP约束处理
        XICP_PROJECTION            // XICP投影处理
    };


    enum class ICPEngine {
        CUSTOM_EULER,    // 自定义欧拉角实现
        CUSTOM_SO3,      // 自定义SO(3)实现
        OPEN3D           // Open3D实现
    };


    // Configuration structure
    struct Config {
        // Test configuration
        int num_runs = 1;
        bool save_pcd = true;
        bool save_error_pcd = true;
        bool visualize = false;

        double CONVERGENCE_THRESH_ROT = 1e-5;
        double CONVERGENCE_THRESH_TRANS = 1e-3;


        // File paths
        std::string folder_path;
        std::string source_pcd;
        std::string target_pcd;
        std::string output_folder;

        // ICP parameters
        double search_radius = 1.0;
        int max_iterations = 30;
        int normal_nn = 5;
        double error_threshold = 0.05; // for visualization

        // Initial noise
        Pose6D initial_noise;
        Pose6D gt_pose;

        // will rest if you set gt_pose in the yaml file
        Eigen::Matrix4d gt_matrix = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d initial_matrix = Eigen::Matrix4d::Identity();

        // ICP parameter values
        ICPParameters icp_params;

        // Test methods
        std::map <std::string, std::pair<std::string, std::string>> test_methods;

        // Use SO(3) parameterization
        bool use_so3_parameterization = true;
    };

    // --- CSV Logging Data Structure ---
    struct IterationLogData {
        int iter_count = 0;
        int effective_points = 0; // point-to-plane corress
        double rmse = 0.0;
        double fitness = 0.0;
        int corr_num = 0;        // point-to-point corress

        double iter_time_ms = 0.0;
        double cond_schur_rot = NAN;
        double cond_schur_trans = NAN;
        double cond_diag_rot = NAN;
        double cond_diag_trans = NAN;
        double cond_full_evd_sub_rot = NAN;
        double cond_full_evd_sub_trans = NAN;
        double cond_full_svd = NAN;
        double cond_full = NAN;  // 直接从H = J^T * J计算的条件数
        Eigen::Vector3d lambda_schur_rot = Eigen::Vector3d::Constant(NAN);
        Eigen::Vector3d lambda_schur_trans = Eigen::Vector3d::Constant(NAN);
        Eigen::Matrix<double, 6, 1> eigenvalues_full = Eigen::Matrix<double, 6, 1>::Constant(NAN);
        Eigen::Matrix<double, 6, 1> singular_values_full = Eigen::Matrix<double, 6, 1>::Constant(NAN);
        Eigen::Matrix<double, 6, 1> update_dx = Eigen::Matrix<double, 6, 1>::Constant(NAN);
        bool is_degenerate = false;
        std::vector<bool> degenerate_mask = std::vector<bool>(6, false);

        Eigen::Matrix4d transform_matrix = Eigen::Matrix4d::Identity(); // 添加变换矩阵
        double trans_error_vs_gt = 0.0;  // 相对于真值的平移误差
        double rot_error_vs_gt = 0.0;    // 相对于真值的旋转误差
        Eigen::Vector3d trans_eigenvalues = Eigen::Vector3d::Zero();
        Eigen::Vector3d rot_eigenvalues = Eigen::Vector3d::Zero();


        // 新增：对角块特征值
        Eigen::Vector3d lambda_diag_rot = Eigen::Vector3d::Constant(NAN);
        Eigen::Vector3d lambda_diag_trans = Eigen::Vector3d::Constant(NAN);

        // 新增：预处理矩阵
        Eigen::Matrix<double, 6, 6> P_preconditioner = Eigen::Matrix<double, 6, 6>::Identity();
        Eigen::Matrix<double, 6, 6> W_adaptive = Eigen::Matrix<double, 6, 6>::Zero();


        // 新增：对于Schur+PCG方法的特殊信息
        Eigen::Matrix3d aligned_V_rot = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d aligned_V_trans = Eigen::Matrix3d::Identity();
        std::vector<int> rot_indices = {0, 1, 2};
        std::vector<int> trans_indices = {0, 1, 2};

        // 新增：梯度信息（-J^T * r）
        Eigen::Matrix<double, 6, 1> gradient = Eigen::Matrix<double, 6, 1>::Constant(NAN);

        // 新增：目标函数值（0.5 * r^T * r）
        double objective_value = NAN;

        // 默认构造函数
        IterationLogData() = default;

        // 拷贝构造函数
        IterationLogData(const IterationLogData& other) = default;

        // 移动构造函数
        IterationLogData(IterationLogData&& other) noexcept = default;

        // 拷贝赋值运算符
        IterationLogData& operator=(const IterationLogData& other) = default;

        // 移动赋值运算符
        IterationLogData& operator=(IterationLogData&& other) noexcept = default;

        // 析构函数
        ~IterationLogData() = default;

        // 深拷贝方法（可选，但推荐）
        IterationLogData deepCopy() const {
            return *this;  // 由于所有成员都支持深拷贝，直接返回副本即可
        }

    };


    // Test result structure
    struct TestResult {
        std::string method_name;
        bool converged = false;
        int iterations = 0;
        double time_ms = 0.0;
        double trans_error_m = 0.0;
        double rot_error_deg = 0.0;
        double final_rmse = 0.0;
        double final_fitness = 0.0;
        double p2p_rmse = 0.0;
        double p2p_fitness = 0.0;
        double chamfer_distance = 0.0;
        int corr_num = 0;
        Eigen::Matrix4d final_transform = Eigen::Matrix4d::Identity();

        // Degeneracy analysis (for single run)
        std::vector<double> condition_numbers;
        std::vector<double> eigenvalues;
        std::vector<double> singular_values;
        std::vector<bool> degenerate_mask;

        // Iteration history for plotting
        std::vector<double> iter_rmse_history;
        std::vector<double> iter_fitness_history;
        std::vector<int> iter_corr_num_history;
        std::vector<double> iter_trans_error_history;
        std::vector<double> iter_rot_error_history;

        // 新增：保存每次迭代的完整数据
        std::vector <IterationLogData> iteration_data;

        // 新增：保存每次迭代的变换矩阵
        std::vector <Eigen::Matrix4d> iter_transform_history;

        // SuperLoc特有的数据
        struct SuperLocData {
            bool has_data = false;
            double uncertainty_x = 0.0;
            double uncertainty_y = 0.0;
            double uncertainty_z = 0.0;
            double uncertainty_roll = 0.0;
            double uncertainty_pitch = 0.0;
            double uncertainty_yaw = 0.0;
            double cond_full = 0.0;
            double cond_rot = 0.0;
            double cond_trans = 0.0;
            bool is_degenerate = false;
            Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Identity();
            std::array<int, 9> feature_histogram = {0, 0, 0, 0, 0, 0, 0, 0, 0};  // 新增：特征可观测性直方图
        } superloc_data;
    };

// Statistics for multiple runs
    struct MethodStatistics {
        std::string method_name;
        int total_runs = 0;
        int converged_runs = 0;
        int corr_num = 0;

        // Mean values
        double mean_trans_error = 0.0;
        double mean_rot_error = 0.0;
        double mean_time_ms = 0.0;
        double mean_iterations = 0.0;
        double mean_rmse = 0.0;
        double mean_fitness = 0.0;
        double mean_p2p_rmse = 0.0;
        double mean_p2p_fitness = 0.0;
        double mean_chamfer = 0.0;

        // Standard deviations
        double std_trans_error = 0.0;
        double std_rot_error = 0.0;
        double std_time_ms = 0.0;

        // Min/Max values
        double min_trans_error = std::numeric_limits<double>::max();
        double max_trans_error = 0.0;
        double min_rot_error = std::numeric_limits<double>::max();
        double max_rot_error = 0.0;

        // Success rate
        double success_rate = 0.0;
    };


    // --- ICP State Context ---
    struct ICPContext {
        // Pointers to clouds used internally
        pcl::PointCloud<PointT>::Ptr laserCloudEffective;
        pcl::PointCloud<PointT>::Ptr coeffSel; // Stores weighted normal (xyz) and residual (intensity)

        // 添加目标点云法向量存储
        pcl::PointCloud<pcl::Normal>::Ptr targetNormals;
        pcl::PointCloud<PointT>::Ptr targetCloud;  // 保存目标点云引用

        // Internal state vectors (resized based on input cloud)
        std::vector <PointT> laserCloudOriSurfVec;
        std::vector <PointT> coeffSelSurfVec;
        std::vector <uint8_t> laserCloudOriSurfFlag; // Use uint8_t for bool efficiency

        // KdTree for the target map
        pcl::KdTreeFLANN<PointT>::Ptr kdtreeSurfFromMap;

        // Result Storage
        Eigen::Matrix<double, 6, 6> icp_cov; // Final computed covariance
        std::vector <IterationLogData> iteration_log_data_; // Log data per iteration
        double total_icp_time_ms_ = 0.0;
        Pose6D final_pose_; // The final optimized pose
        bool final_convergence_flag_ = false;
        int final_iterations_ = 0; // Store the number of iterations performed

        // Constructor to initialize pointers and default covariance
        ICPContext() :
                laserCloudEffective(new pcl::PointCloud<PointT>()),
                coeffSel(new pcl::PointCloud<PointT>()),
                kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointT>()),
                targetNormals(new pcl::PointCloud<pcl::Normal>()),
                targetCloud(new pcl::PointCloud<PointT>()),
                icp_cov(Eigen::Matrix<double, 6, 6>::Identity()) {
            icp_cov *= 1e6; // Default high covariance
        }

        // Prevent copying (pointers would be shallow copied)
        ICPContext(const ICPContext &) = delete;

        ICPContext &operator=(const ICPContext &) = delete;

        // Setup method (optional, can be done in runSingleICPTest)
        //        void setTargetCloud(pcl::PointCloud<PointT>::Ptr target_cloud_ptr) {
        //            if (!target_cloud_ptr || target_cloud_ptr->empty()) {
        //                std::cerr << "[ICPContext::setTargetCloud] Error: Target cloud is null or empty." << std::endl;
        //                return;
        //            }
        //            kdtreeSurfFromMap->setInputCloud(target_cloud_ptr);
        //            std::cout << "[ICPContext::setTargetCloud] KdTree built for target cloud with "
        //                      << target_cloud_ptr->size() << " points." << std::endl;
        //        }

        // 修改setTargetCloud，同时计算法向量
        void setTargetCloud(pcl::PointCloud<PointT>::Ptr target_cloud_ptr, int normal_nn) {
            if (!target_cloud_ptr || target_cloud_ptr->empty()) {
                std::cerr << "[ICPContext::setTargetCloud] Error: Target cloud is null or empty." << std::endl;
                return;
            }

            // 保存目标点云
            targetCloud = target_cloud_ptr;

            // 设置KdTree
            kdtreeSurfFromMap->setInputCloud(target_cloud_ptr);

            // 预计算法向量
            pcl::NormalEstimation <PointT, pcl::Normal> ne;
            ne.setInputCloud(target_cloud_ptr);
            typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
            ne.setSearchMethod(tree);
            ne.setKSearch(normal_nn);  // 使用10个最近邻
            ne.compute(*targetNormals);

            // 确保法向量方向一致（可选）
            //            for (size_t i = 0; i < targetNormals->size(); ++i) {
            //                if (targetNormals->points[i].normal_z < 0) {
            //                    targetNormals->points[i].normal_x *= -1;
            //                    targetNormals->points[i].normal_y *= -1;
            //                    targetNormals->points[i].normal_z *= -1;
            //                }
            //            }

            std::cout << "[ICPContext::setTargetCloud] KdTree built and normals computed for target cloud with "
                      << target_cloud_ptr->size() << " points." << std::endl;
        }
    };

    struct DegeneracyAnalysisResult {
        bool isDegenerate = false;
        std::vector<bool> degenerate_mask;
        double cond_schur_rot = NAN, cond_schur_trans = NAN;
        double cond_diag_rot = NAN, cond_diag_trans = NAN;
        double cond_full = NAN;
        double cond_full_sub_rot = NAN, cond_full_sub_trans = NAN;
        Eigen::Matrix<double, 6, 1> eigenvalues_full;
        Eigen::Matrix<double, 6, 1> singular_values;
        Eigen::Vector3d lambda_schur_rot = Eigen::Vector3d::Constant(NAN);
        Eigen::Vector3d lambda_schur_trans = Eigen::Vector3d::Constant(NAN);
        Eigen::Vector3d lambda_sub_rot = Eigen::Vector3d::Constant(NAN);
        Eigen::Vector3d lambda_sub_trans = Eigen::Vector3d::Constant(NAN);
        Eigen::Matrix3d aligned_V_rot = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d aligned_V_trans = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d schur_V_rot = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d schur_V_trans = Eigen::Matrix3d::Identity();
        std::vector<int> rot_indices = {0, 1, 2};
        std::vector<int> trans_indices = {0, 1, 2};
        Eigen::Matrix<double, 6, 6> W_adaptive = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 6> P_preconditioner = Eigen::Matrix<double, 6, 6>::Identity();
    };


    // Pose6D -> Matrix4d (保持不变，因为实现已经正确)
    inline Eigen::Matrix4d Pose6D2Matrix(const Pose6D &p) {
        Translation3d tf_trans(p.x, p.y, p.z);
        AngleAxisd rot_x(p.roll, Vector3d::UnitX());
        AngleAxisd rot_y(p.pitch, Vector3d::UnitY());
        AngleAxisd rot_z(p.yaw, Vector3d::UnitZ());
        // 顺序: Z * Y * X (内旋 XYZ 或 外旋 ZYX)
        Matrix4d mat = (tf_trans * rot_z * rot_y * rot_x).matrix();
        return mat;
    }

    inline Pose6D MatrixToPose6D(const Eigen::Matrix4d &matrix) {
        Pose6D p;
        p.x = matrix(0, 3);
        p.y = matrix(1, 3);
        p.z = matrix(2, 3);

        Eigen::Matrix3d rot_matrix = matrix.block<3, 3>(0, 0);
        Eigen::Quaterniond q(rot_matrix);

        double siny_cosp = 2.0 * (q.w() * q.z() + q.x() * q.y());
        double cosy_cosp = 1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());
        p.yaw = std::atan2(siny_cosp, cosy_cosp);

        double sinp = 2.0 * (q.w() * q.y() - q.z() * q.x());
        if (std::abs(sinp) >= 1.0)
            p.pitch = std::copysign(M_PI / 2.0, sinp);
        else
            p.pitch = std::asin(sinp);

        double sinr_cosp = 2.0 * (q.w() * q.x() + q.y() * q.z());
        double cosr_cosp = 1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y());
        p.roll = std::atan2(sinr_cosp, cosr_cosp);

        return p;
    }


    /**
     * Calculate pose error between ground truth and estimated transformation
     *
     * @param gt_matrix Ground truth transformation matrix (4x4)
     * @param final_matrix Estimated/final transformation matrix (4x4)
     * @param use_degrees If true, return rotation error in degrees; otherwise in radians
     * @return PoseError struct containing translation and rotation errors
     */
    inline PoseError calculatePoseError(const Eigen::Matrix4d &gt_matrix,
                                        const Eigen::Matrix4d &final_matrix,
                                        bool use_degrees = true) {
        PoseError error;

        // Calculate relative error transformation
        // error_matrix represents how much final_matrix deviates from gt_matrix
        Eigen::Matrix4d error_matrix = gt_matrix.inverse() * final_matrix;



        // Extract translation error (Euclidean norm of translation vector)
        error.translation_error = error_matrix.block<3, 1>(0, 3).norm();

        // Extract rotation error
        Eigen::Matrix3d R_error = error_matrix.block<3, 3>(0, 0);

        // Method 1: Using rotation matrix trace (more numerically stable)
//        double trace = R_error.trace();
//        // Clamp to avoid numerical issues with acos
//        double cos_angle = std::max(-1.0, std::min(1.0, (trace - 1.0) / 2.0));
//        double angle_rad = std::acos(cos_angle);

        // Alternative Method 2: Using AngleAxis (commented out)
        Eigen::AngleAxisd angle_axis(R_error);
        double angle_rad = std::abs(angle_axis.angle());


        // Convert to degrees if requested
        if (use_degrees) {
            error.rotation_error = angle_rad * 180.0 / M_PI;
        } else {
            error.rotation_error = angle_rad;
        }
        // std::cout << "trans error: " << error.translation_error << " " << error.rotation_error << std::endl;


        return error;
    }

    // Calculate point-to-point error metrics
    inline void calculatePointToPointError(const pcl::PointCloud<PointT>::Ptr &aligned_cloud,
                                           const pcl::PointCloud<PointT>::Ptr &target_cloud,
                                           double &rmse, double &fitness, double &chamfer,
                                           int &valid_correspondences,
                                           double &error_threshold) {
        pcl::KdTreeFLANN <PointT> kdtree;
        kdtree.setInputCloud(target_cloud);

        double sum_sq_error = 0.0;
        valid_correspondences = 0;
        fitness = 0.0;
        double sum_forward = 0.0;

        // Forward direction: aligned -> target
        for (size_t i = 0; i < aligned_cloud->points.size(); ++i) {
            std::vector<int> indices(1);
            std::vector<float> sq_distances(1);

            if (kdtree.nearestKSearch(aligned_cloud->points[i], 1, indices, sq_distances) > 0) {
                double dist = std::sqrt(sq_distances[0]);
                sum_forward += dist;

                if (dist < error_threshold) {
                    sum_sq_error += sq_distances[0];
                    valid_correspondences++;
                }
            }
        }

        // RMSE
        rmse = std::sqrt(sum_sq_error / aligned_cloud->points.size());
        //        rmse = std::sqrt(sum_sq_error / valid_correspondences);

        // Fitness (percentage of points within threshold)
        fitness = static_cast<double>(valid_correspondences) / aligned_cloud->points.size();

        // Chamfer distance (symmetric)
        kdtree.setInputCloud(aligned_cloud);
        double sum_backward = 0.0;

        for (size_t i = 0; i < target_cloud->points.size(); ++i) {
            std::vector<int> indices(1);
            std::vector<float> sq_distances(1);

            if (kdtree.nearestKSearch(target_cloud->points[i], 1, indices, sq_distances) > 0) {
                sum_backward += std::sqrt(sq_distances[0]);
            }
        }

        chamfer = (sum_forward / aligned_cloud->points.size() +
                   sum_backward / target_cloud->points.size()) / 2.0;
    }

    // 新的jet colormap函数
    inline pcl::PointXYZRGB getJetColorForError(double error, double max_threshold) {
        pcl::PointXYZRGB point;
        // 将误差归一化到 [0, 1] 范围
        double normalized_error = std::min(error / max_threshold, 1.0);
        // Jet colormap: 蓝色(0) -> 青色(0.25) -> 绿色(0.5) -> 黄色(0.75) -> 红色(1.0)
        double r, g, b;
        if (normalized_error < 0.25) {
            // 蓝色到青色
            double t = normalized_error / 0.25;
            r = 0.0;
            g = t;
            b = 1.0;
        } else if (normalized_error < 0.5) {
            // 青色到绿色
            double t = (normalized_error - 0.25) / 0.25;
            r = 0.0;
            g = 1.0;
            b = 1.0 - t;
        } else if (normalized_error < 0.75) {
            // 绿色到黄色
            double t = (normalized_error - 0.5) / 0.25;
            r = t;
            g = 1.0;
            b = 0.0;
        } else {
            // 黄色到红色
            double t = (normalized_error - 0.75) / 0.25;
            r = 1.0;
            g = 1.0 - t;
            b = 0.0;
        }
        point.r = static_cast<uint8_t>(255 * r);
        point.g = static_cast<uint8_t>(255 * g);
        point.b = static_cast<uint8_t>(255 * b);
        return point;
    }


    inline void pointBodyToGlobal(const PointT &pi, PointT &po, const Eigen::Matrix4d &T) {
        Eigen::Vector3d point_in(pi.x, pi.y, pi.z);
        Eigen::Vector3d point_out = T.block<3, 3>(0, 0) * point_in + T.block<3, 1>(0, 3);
        po.x = point_out.x();
        po.y = point_out.y();
        po.z = point_out.z();
    }

    // Open3D转换函数
    inline std::shared_ptr <open3d::geometry::PointCloud> PclToO3d(const pcl::PointCloud <PointT> &pcl_cloud) {
        auto o3d_cloud = std::make_shared<open3d::geometry::PointCloud>();
        o3d_cloud->points_.reserve(pcl_cloud.points.size());
        for (const auto &pcl_point : pcl_cloud.points) {
            if (!std::isfinite(pcl_point.x) || !std::isfinite(pcl_point.y) || !std::isfinite(pcl_point.z)) {
                continue; // 跳过无效点
            }
            o3d_cloud->points_.emplace_back(pcl_point.x, pcl_point.y, pcl_point.z);
        }
        return o3d_cloud;
    }

    inline pcl::PointCloud<PointT>::Ptr O3dToPcl(const open3d::geometry::PointCloud &o3d_cloud) {
        pcl::PointCloud<PointT>::Ptr pcl_cloud(new pcl::PointCloud <PointT>);
        pcl_cloud->points.reserve(o3d_cloud.points_.size());
        for (const auto &o3d_point : o3d_cloud.points_) {
            PointT pcl_point;
            pcl_point.x = static_cast<float>(o3d_point.x());
            pcl_point.y = static_cast<float>(o3d_point.y());
            pcl_point.z = static_cast<float>(o3d_point.z());
            pcl_point.intensity = 0.0f; // 默认强度
            pcl_cloud->points.push_back(pcl_point);
        }
        pcl_cloud->width = pcl_cloud->points.size();
        pcl_cloud->height = 1;
        pcl_cloud->is_dense = false;
        return pcl_cloud;
    }

    class TicToc {
    public:
        TicToc() { tic(); }

        void tic() { start = std::chrono::system_clock::now(); }

        double toc() {
            end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            return elapsed_seconds.count() * 1000; // return ms
        }

    private:
        std::chrono::time_point <std::chrono::system_clock> start, end;
    };

}

#endif //CLOUD_MAP_EVALUATION_UTILS_H
