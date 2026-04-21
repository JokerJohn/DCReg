#include "dcreg.hpp"

#include <array>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

dcreg::LinearSystem BuildSyntheticWeakAxisSystem() {
  dcreg::LinearSystem system;

  dcreg::Matrix6d basis = dcreg::Matrix6d::Identity();
  basis(0, 2) = 0.35;
  basis(1, 5) = 0.15;
  basis(2, 5) = 0.45;
  basis(3, 2) = 0.20;
  basis(4, 5) = 0.25;

  dcreg::Vector6d stiffness;
  stiffness << 6.0, 4.5, 1.1, 5.0, 3.5, 0.18;

  const dcreg::Matrix6d jacobian = basis * stiffness.asDiagonal();
  system.hessian = jacobian.transpose() * jacobian;
  system.rhs << 0.8, -1.1, 0.6, -0.5, 0.3, 1.4;
  return system;
}

std::string MaskString(const std::array<bool, 6>& mask) {
  std::string text;
  text.reserve(mask.size());
  for (const bool value : mask) {
    text.push_back(value ? '1' : '0');
  }
  return text;
}

void PrintContributionMatrix(const std::string& title,
                             const std::array<const char*, 3>& row_labels,
                             const std::array<const char*, 3>& col_labels,
                             const Eigen::Matrix3d& ratios) {
  const std::ios::fmtflags original_flags = std::cout.flags();
  std::cout << title << '\n';
  std::cout << "                " << std::right << std::setw(12) << col_labels[0]
            << std::setw(12) << col_labels[1] << std::setw(12) << col_labels[2]
            << '\n';
  for (int row = 0; row < 3; ++row) {
    std::cout << std::left << std::setw(16) << row_labels[row];
    for (int col = 0; col < 3; ++col) {
      std::cout << std::right << std::setw(12) << ratios(row, col);
    }
    std::cout << '\n';
  }
  std::cout.flags(original_flags);
}

}  // namespace

int main() {
  dcreg::SolverParameters params;
  params.degeneracy_condition_threshold = 10.0;
  params.kappa_target = 10.0;
  params.pcg_tolerance = 1e-10;
  params.pcg_max_iterations = 20;

  const dcreg::LinearSystem system = BuildSyntheticWeakAxisSystem();
  const dcreg::DegeneracyDetection detection =
      dcreg::DetectDegeneracy(system.hessian, params);
  const dcreg::DegeneracyCharacterization characterization =
      dcreg::CharacterizeDegeneracy(detection, params);
  const dcreg::LinearSolveReport preconditioned =
      dcreg::SolvePreconditionedUpdate(system, params, characterization);

  const std::array<const char*, 3> kRotAxisLabels = {"roll", "pitch", "yaw"};
  const std::array<const char*, 3> kTransAxisLabels = {"x", "y", "z"};
  const std::array<const char*, 3> kRotModeLabels = {"mode_r", "mode_p",
                                                     "mode_y"};
  const std::array<const char*, 3> kTransModeLabels = {"mode_x", "mode_y",
                                                       "mode_z"};

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Synthetic DCReg example\n";
  std::cout << "[Module 1] Spectral degeneracy detection\n";
  std::cout << "cond_full: " << detection.cond_full << '\n';
  std::cout << "cond_schur_rot: " << characterization.cond_schur_rot << '\n';
  std::cout << "cond_schur_trans: " << characterization.cond_schur_trans << '\n';
  std::cout << '\n';

  std::cout << "[Module 2] Physical-axis degeneracy characterization\n";
  std::cout << "degenerate_mask: " << MaskString(characterization.degenerate_mask)
            << '\n';
  std::cout << "raw_lambda_rot: "
            << detection.lambda_schur_rot.transpose() << '\n';
  std::cout << "raw_lambda_trans: "
            << detection.lambda_schur_trans.transpose() << '\n';
  std::cout << "aligned_lambda_rpy: "
            << characterization.aligned_lambda_schur_rot.transpose() << '\n';
  std::cout << "aligned_lambda_xyz: "
            << characterization.aligned_lambda_schur_trans.transpose() << '\n';
  PrintContributionMatrix("rot_axis_contribution_ratio(rows=rpy, cols=aligned_rpy):",
                          kRotAxisLabels, kRotModeLabels,
                          characterization.rot_axis_contribution_ratio);
  PrintContributionMatrix(
      "trans_axis_contribution_ratio(rows=xyz, cols=aligned_xyz):",
      kTransAxisLabels, kTransModeLabels,
      characterization.trans_axis_contribution_ratio);
  std::cout << "clamped_lambda_rpy: "
            << characterization.clamped_lambda_schur_rot.transpose() << '\n';
  std::cout << "clamped_lambda_xyz: "
            << characterization.clamped_lambda_schur_trans.transpose() << '\n';
  std::cout << '\n';

  std::cout << "[Module 3] Preconditioned linear solve\n";
  std::cout << "preconditioned_delta: "
            << preconditioned.delta.transpose() << '\n';
  std::cout << "pcg_iterations: " << preconditioned.iterations << '\n';
  std::cout << "pcg_relative_residual: "
            << preconditioned.relative_residual << '\n';
  std::cout << "qr_fallback: " << (preconditioned.used_qr_fallback ? 1 : 0)
            << '\n';
  return 0;
}
