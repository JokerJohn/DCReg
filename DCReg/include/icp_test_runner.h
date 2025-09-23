#ifndef ICP_TEST_RUNNER_HPP
#define ICP_TEST_RUNNER_HPP

#pragma once
#include <yaml-cpp/yaml.h>
#include <iomanip>
#include <sstream>
#include <filesystem>

#include "dcreg.hpp"
#include "superloc.h"
#include "xicp.h"
#include "math_utils.hpp"
#include "hessian_computer.h"

namespace ICPRunner {

    // Configuration loading function
    bool loadConfig(const std::string &filename, Config &config);

    // Main test runner class
    class TestRunner {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        TestRunner(const Config &config);

        // Run all configured test methods
        bool runAllTests();

        // Run a single method multiple times
        bool runMethod(const std::string &method_name,
                       DetectionMethod detection,
                       HandlingMethod handling);

        // Save results
        void saveStatistics();

        void saveDetailedResults();

        // Print current parameters
        void printCurrentParameters(const std::string &method_name,
                                    DetectionMethod detection,
                                    HandlingMethod handling);

    private:
        Config config_;
        std::map <std::string, MethodStatistics> statistics_;
        std::map <std::string, std::vector<TestResult>> detailed_results_;

        // Point clouds
        pcl::PointCloud<PointT>::Ptr source_cloud_;
        pcl::PointCloud<PointT>::Ptr target_cloud_;

        std::shared_ptr <DCReg> dcreg_;


        // Helper functions
        bool loadPointClouds();

        TestResult runSingleTest(const std::string &method_name,
                                 DetectionMethod detection,
                                 HandlingMethod handling);


        // Original Euler-based ICP follow LOAM: https://github.com/laboshinl/loam_velodyne
        // the results of baseline in DCReg paper are based on this implementation, since we need to keep consistent with the original LOAM
        // some minor changes:
        // 1. we only use point-to-plane error
        // 2. we handle degeneracy across the iterations instead of only in the first iteration
        // 3. we set a threshold to filter correspondences  
        bool Point2PlaneICP(
                pcl::PointCloud<PointT>::Ptr measure_cloud,
                pcl::PointCloud<PointT>::Ptr target_cloud,
                const Pose6D &initial_pose,
                double SEARCH_RADIUS,
                DetectionMethod detection_method,
                HandlingMethod handling_method,
                int MAX_ITERATIONS,
                ICPContext &context,
                TestResult &result,
                Pose6D &output_pose
        );

                // SO(3)-based Point-to-Plane ICP Implementation with Weight Derivative
        // actually compared to the original implement of LOAM: https://github.com/laboshinl/loam_velodyne
        // we do several changes in this function:
        // 1. we only use point-to-plane error instead of point-to-line + point-to-plane error
        // 2. we use SO3 for rotation representation instead of Euler angles
        // 3. we handle degeneracy across the iterations instead of only in the first iteration
        // 4. we fixed the bugs of OpenMP parallelization
        bool Point2PlaneICP_SO3_OpenMP(
                pcl::PointCloud<PointT>::Ptr measure_cloud,
                pcl::PointCloud<PointT>::Ptr target_cloud,
                const MathUtils::SE3State &initial_state,
                double SEARCH_RADIUS,
                DetectionMethod detection_method,
                HandlingMethod handling_method,
                int MAX_ITERATIONS,
                ICPContext &context,
                TestResult &result,
                MathUtils::SE3State &output_state);


        /**
         * @brief SO(3)-based Point-to-Plane ICP Implementation with TBB and X-ICP
         *  we follow the implementation of XICP: https://ieeexplore.ieee.org/document/10328716
         *  xicp codes: https://github.com/leggedrobotics/perfectlyconstrained
         */
        bool Point2PlaneICP_SO3_tbb_XICP(
                pcl::PointCloud<PointT>::Ptr measure_cloud,
                pcl::PointCloud<PointT>::Ptr target_cloud,
                const MathUtils::SE3State &initial_state,
                double SEARCH_RADIUS,
                DetectionMethod detection_method,
                HandlingMethod handling_method,
                int MAX_ITERATIONS,
                ICPContext &context,
                TestResult &result,
                MathUtils::SE3State &output_state);


        // Open3D ICP实现
        bool runOpen3DICP(const std::string &method_name, TestResult &result);


        void visualizeResults(const pcl::PointCloud<PointT>::Ptr &aligned_cloud,
                              const pcl::PointCloud<PointT>::Ptr &target_cloud,
                              const std::string &method_name,
                              const TestResult &result);

        // Point cloud error calculation with color coding
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr createErrorPointCloud(
                const pcl::PointCloud<PointT>::Ptr &aligned_cloud,
                const pcl::PointCloud<PointT>::Ptr &target_cloud,
                double max_error_threshold);


        // Visualization functions
        void saveAlignedClouds(const pcl::PointCloud<PointT>::Ptr &aligned_cloud,
                               const pcl::PointCloud<PointT>::Ptr &target_cloud,
                               const std::string &filename);

        void saveErrorPointCloud(const pcl::PointCloud<PointT>::Ptr &aligned_cloud,
                                 const pcl::PointCloud<PointT>::Ptr &target_cloud,
                                 const std::string &filename);

        // Statistics calculation
        void updateStatistics(const std::string &method_name, const TestResult &result);

        void finalizeStatistics();

        // String conversion helpers
        DetectionMethod stringToDetectionMethod(const std::string &str);

        HandlingMethod stringToHandlingMethod(const std::string &str);
    };


} // namespace ICPRunner

#endif // ICP_TEST_RUNNER_HPP