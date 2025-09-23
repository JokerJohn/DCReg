#include "icp_test_runner.h"


namespace fs = std::filesystem;

int main(int argc, char **argv) {

    //    std::string config_file = "../config/icp_pk01.yaml"; // real-world data config in planner degenerate case
    // std::string config_file = "../config/icp.yaml";
    // std::string config_file = "../config/icp_iter.yaml";
    std::string config_file = "../config/icp.yaml";



    ICPRunner::Config config;
    if (!ICPRunner::loadConfig(config_file, config)) {
        std::cerr << "Failed to load configuration from: " << config_file << std::endl;
        return 1;
    }

    // Create output directory
    fs::create_directories(config.output_folder);

    // Initialize test runner
    ICPRunner::TestRunner runner(config);

    // Run all configured methods
    std::cout << "\n========================================" << std::endl;
    std::cout << "Starting ICP Test Suite" << std::endl;
    std::cout << "Number of methods: " << config.test_methods.size() << std::endl;
    std::cout << "Number of runs per method: " << config.num_runs << std::endl;
    std::cout << "========================================\n" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Run tests
    if (!runner.runAllTests()) {
        std::cerr << "Test execution failed!" << std::endl;
        return 1;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    std::cout << "\n========================================" << std::endl;
    std::cout << "All tests completed successfully!" << std::endl;
    std::cout << "Total time: " << duration.count() << " seconds" << std::endl;
    std::cout << "Results saved to: " << config.output_folder << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}