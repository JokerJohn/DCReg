#include "dcreg.hpp"

#include <array>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

using dcreg::Parameterization;
using dcreg::PointCloud;
using dcreg::PointCloudPtr;
using dcreg::RegistrationResult;
using dcreg::RunOptions;
using dcreg::TestCase;

void PrintComparisonHeader() {
  std::cout << std::left << std::setw(14) << "Param"
            << std::setw(12) << "Status"
            << std::setw(10) << "Iter"
            << std::setw(14) << "RMSE"
            << std::setw(14) << "Fitness"
            << std::setw(14) << "Trans(m)"
            << std::setw(14) << "Rot(deg)"
            << std::setw(10) << "LinIt"
            << std::setw(10) << "QRfb"
            << std::setw(14) << "Time(ms)"
            << '\n';
  std::cout << std::string(112, '-') << '\n';
}

void PrintComparisonRow(const Parameterization parameterization,
                        const RegistrationResult& result) {
  const std::string status = result.success ? "ok" : "failed";
  int linear_solver_iterations = 0;
  int qr_fallback = 0;
  if (!result.history.empty()) {
    linear_solver_iterations = result.history.back().linear_solver_iterations;
    qr_fallback = result.history.back().used_qr_fallback ? 1 : 0;
  }
  std::cout << std::left << std::setw(14) << dcreg::ToString(parameterization)
            << std::setw(12) << status
            << std::setw(10) << result.iterations
            << std::setw(14) << result.rmse
            << std::setw(14) << result.fitness
            << std::setw(14) << result.pose_error.translation_m
            << std::setw(14) << result.pose_error.rotation_deg
            << std::setw(10) << linear_solver_iterations
            << std::setw(10) << qr_fallback
            << std::setw(14) << result.time_ms
            << '\n';
}

int RunParameterizationComparison(const TestCase& test_case,
                                  const RunOptions& base_options) {
  PointCloudPtr source(new PointCloud);
  PointCloudPtr target(new PointCloud);

  std::string error_message;
  const std::string source_path = test_case.folder_path + test_case.source_pcd;
  const std::string target_path = test_case.folder_path + test_case.target_pcd;
  if (!dcreg::LoadPointCloud(source_path, source, &error_message)) {
    std::cerr << error_message << '\n';
    return 1;
  }
  if (!dcreg::LoadPointCloud(target_path, target, &error_message)) {
    std::cerr << error_message << '\n';
    return 1;
  }

  constexpr std::array<Parameterization, 4> kParameterizations = {
      Parameterization::kEuler,
      Parameterization::kSE3,
      Parameterization::kSO3,
      Parameterization::kQuaternion,
  };

  std::array<RegistrationResult, kParameterizations.size()> results;
  int success_count = 0;

  std::cout << "Test case: " << test_case.name
            << " | Algorithm: " << dcreg::ToString(base_options.algorithm)
            << " | Parallel: " << dcreg::ToString(base_options.parallel_mode)
            << '\n';
  PrintComparisonHeader();

  for (std::size_t index = 0; index < kParameterizations.size(); ++index) {
    RunOptions options = base_options;
    options.parameterization = kParameterizations[index];
    results[index] = dcreg::RunRegistration(*source, *target, test_case, options);
    PrintComparisonRow(kParameterizations[index], results[index]);
    if (results[index].success) {
      ++success_count;
    }
  }

  for (std::size_t index = 0; index < kParameterizations.size(); ++index) {
    if (results[index].success || results[index].failure_reason.empty()) {
      continue;
    }
    std::cout << dcreg::ToString(kParameterizations[index]) << ": "
              << results[index].failure_reason << '\n';
  }

  std::cout << "Summary: " << success_count << "/" << kParameterizations.size()
            << " parameterizations converged\n";
  return 0;
}

}  // namespace

int main() {
  // Test-case entry:
  // 1. Switch among the built-in presets here.
  // 2. Adjust algorithm / parameterization / backend here.
  dcreg::TestCase test_case = dcreg::kShiftedCylinderCase;
  // test_case = dcreg::kShiftedCylinderLongRunCase;
  // test_case = dcreg::kParkingLotPk01Case;

  dcreg::RunOptions options;
  options.algorithm = dcreg::Algorithm::kDCReg;
  // options.algorithm = dcreg::Algorithm::kNone;
  options.parameterization = dcreg::Parameterization::kSO3;
  // options.parameterization = dcreg::Parameterization::kEuler;
  // options.parameterization = dcreg::Parameterization::kSE3;
  // options.parameterization = dcreg::Parameterization::kQuaternion;
  options.parallel_mode = dcreg::ParallelMode::kTbb;

  constexpr bool kRunParameterizationComparison = true;
  if (kRunParameterizationComparison) {
    return RunParameterizationComparison(test_case, options);
  }

  const dcreg::RegistrationResult result =
      dcreg::RunRegistration(test_case, options);
  dcreg::PrintSummary(test_case, options, result);
  return result.success ? 0 : 1;
}
