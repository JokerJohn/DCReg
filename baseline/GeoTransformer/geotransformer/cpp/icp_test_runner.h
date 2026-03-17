#ifndef ICP_TEST_RUNNER_HPP
#define ICP_TEST_RUNNER_HPP

#include <yaml-cpp/yaml.h>


#include "superloc.h"
#include "xicp.h"
#include "dcreg.hpp"
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


        // Original Euler-based ICP
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

        bool Point2PlaneICP_SO3_tbb(
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


        // 集成到您的Point2PlaneICP_SO3_tbb函数中
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

        // SO(3)-based Point-to-Plane ICP Implementation with Weight Derivative
        bool Point2PlaneICP_SO3(
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