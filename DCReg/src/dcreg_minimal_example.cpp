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

// ===== BEGIN CHANGE: clearer spectral-mode contribution log =====
void PrintContributionModes(const std::string& title,
                            const std::string& mode_prefix,
                            const std::array<const char*, 3>& axis_labels,
                            const Eigen::Matrix3d& ratios) {
  std::cout << title << '\n';
  for (int mode = 0; mode < 3; ++mode) {
    std::cout << "  " << mode_prefix << mode << " = ";
    for (int axis = 0; axis < 3; ++axis) {
      if (axis > 0) {
        std::cout << " + ";
      }
      std::cout << ratios(axis, mode) << "*" << axis_labels[axis];
    }
    std::cout << '\n';
  }
}
// ===== END CHANGE: clearer spectral-mode contribution log =====

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
  PrintContributionModes(
      "rot_axis_contribution_ratio(each r_i as physical-axis mixture):", "r",
      kRotAxisLabels, characterization.rot_axis_contribution_ratio);
  PrintContributionModes(
      "trans_axis_contribution_ratio(each t_i as physical-axis mixture):", "t",
      kTransAxisLabels,
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
