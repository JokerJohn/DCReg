#ifndef DCREG_DCREG_HPP_
#define DCREG_DCREG_HPP_

#include "utils.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <array>
#include <stdexcept>

namespace dcreg {

struct Correspondence {
  Eigen::Vector3d point_body = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal_world = Eigen::Vector3d::Zero();
  double residual = 0.0;
  double weight = 0.0;
  double weight_derivative = 0.0;
};

struct LinearSystem {
  Matrix6d hessian = Matrix6d::Zero();
  Vector6d rhs = Vector6d::Zero();
};

struct DegeneracyDetection {
  bool factorization_ok = false;
  double cond_full = std::numeric_limits<double>::quiet_NaN();
  double cond_schur_rot = std::numeric_limits<double>::quiet_NaN();
  double cond_schur_trans = std::numeric_limits<double>::quiet_NaN();
  Eigen::Vector3d lambda_schur_rot = Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN());
  Eigen::Vector3d lambda_schur_trans = Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN());
  Eigen::Matrix3d raw_rot_basis = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d raw_trans_basis = Eigen::Matrix3d::Identity();
};

struct DegeneracyCharacterization {
  bool factorization_ok = false;
  bool is_degenerate = false;
  double cond_full = std::numeric_limits<double>::quiet_NaN();
  double cond_schur_rot = std::numeric_limits<double>::quiet_NaN();
  double cond_schur_trans = std::numeric_limits<double>::quiet_NaN();
  std::array<bool, 6> degenerate_mask = {false, false, false, false, false,
                                         false};
  Eigen::Vector3d lambda_schur_rot = Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN());
  Eigen::Vector3d lambda_schur_trans = Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN());
  Eigen::Vector3d aligned_lambda_schur_rot = Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN());
  Eigen::Vector3d aligned_lambda_schur_trans = Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN());
  Eigen::Vector3d clamped_lambda_schur_rot = Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN());
  Eigen::Vector3d clamped_lambda_schur_trans = Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN());
  Eigen::Matrix3d aligned_rot_basis = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d aligned_trans_basis = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d rot_axis_contribution_ratio = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d trans_axis_contribution_ratio = Eigen::Matrix3d::Zero();
  std::array<int, 3> rot_indices = {0, 1, 2};
  std::array<int, 3> trans_indices = {0, 1, 2};

  // Left preconditioner used by the DCReg PCG solve:
  //   z_k = P r_k,  P ~= H^{-1}.
  //
  // Each 3x3 block is assembled from the Schur-complement eigenbasis after
  // selective eigenvalue clamping.
  Matrix6d preconditioner = Matrix6d::Identity();
};

struct LinearSolveReport {
  Vector6d delta = Vector6d::Zero();
  int iterations = 0;
  double relative_residual = std::numeric_limits<double>::quiet_NaN();
  bool converged = false;
  bool used_preconditioned_solver = false;
  bool used_qr_fallback = false;
};

inline constexpr double kAxisOrthogonalityWarningThreshold = 1e-8;

inline double MaxOrthogonalityError(const Eigen::Matrix3d& basis) {
  return (basis.transpose() * basis - Eigen::Matrix3d::Identity())
      .cwiseAbs()
      .maxCoeff();
}

// Disabled verification helpers kept as a paper-reference snapshot.
// They were used to confirm that the mapped Schur eigenbasis is already
// orthonormal up to machine precision on the checked datasets, so the runtime
// path no longer enables them by default.
#if 0
inline bool ApplyGramSchmidtOrthonormalization(Eigen::Matrix3d* basis) {
  for (int column = 0; column < 3; ++column) {
    for (int previous = 0; previous < column; ++previous) {
      basis->col(column) -= basis->col(column).dot(basis->col(previous)) *
                            basis->col(previous);
    }
    const double norm = basis->col(column).norm();
    if (norm < 1e-8) {
      return false;
    }
    basis->col(column) /= norm;
  }
  return true;
}

inline void LogAxisAlignmentDiagnostics(
    const char* label, const Eigen::Matrix3d& basis_before_orthogonalization,
    const Eigen::Matrix3d& basis_after_orthogonalization,
    const std::array<int, 3>& original_indices) {
  const Eigen::Matrix3d basis_delta =
      basis_after_orthogonalization - basis_before_orthogonalization;
  const Eigen::Vector3d column_delta_norms(
      basis_delta.col(0).norm(), basis_delta.col(1).norm(),
      basis_delta.col(2).norm());
  std::cout << "[DCReg][AxisAlignment][" << label << "] indices=["
            << original_indices[0] << ' ' << original_indices[1] << ' '
            << original_indices[2] << "] orthogonality_error_before="
            << MaxOrthogonalityError(basis_before_orthogonalization)
            << " orthogonality_error_after="
            << MaxOrthogonalityError(basis_after_orthogonalization)
            << " delta_fro=" << basis_delta.norm()
            << " column_delta_norms=" << column_delta_norms.transpose() << '\n'
            << "mapped_basis_runtime=\n"
            << basis_before_orthogonalization << '\n'
            << "basis_if_gram_schmidt_enabled=\n"
            << basis_after_orthogonalization << '\n'
            << "basis_delta_if_gram_schmidt_enabled=\n"
            << basis_delta << '\n';
}
#endif

// Core module 2 helper: map the raw Schur eigenvectors into physical axes.
//
// After the Schur EVD in Eq. (19), the eigenvectors are only defined up to
// sign and permutation. This helper implements the physical-axis matching step
// from Algorithm 2 by selecting the eigenvector with the largest absolute inner
// product against each canonical axis and then fixing its sign.
inline bool AlignEigenBasisToAxes(const Eigen::Matrix3d& raw_basis,
                                  const std::array<Eigen::Vector3d, 3>& refs,
                                  const char* label,
                                  Eigen::Matrix3d* aligned_basis,
                                  std::array<int, 3>* original_indices) {
  aligned_basis->setZero();
  original_indices->fill(-1);

  std::array<bool, 3> used = {false, false, false};
  for (int axis = 0; axis < 3; ++axis) {
    double best_score = -1.0;
    int best_index = -1;
    for (int candidate = 0; candidate < 3; ++candidate) {
      if (used[candidate]) {
        continue;
      }
      // Eq. (36)-(38): use the maximum absolute inner product with the
      // canonical axis to decide which Schur eigenvector represents that axis.
      const double score = std::abs(refs[axis].dot(raw_basis.col(candidate)));
      if (score > best_score) {
        best_score = score;
        best_index = candidate;
      }
    }
    if (best_index < 0) {
      return false;
    }
    used[best_index] = true;
    (*original_indices)[axis] = best_index;
    Eigen::Vector3d aligned_column = raw_basis.col(best_index);
    if (refs[axis].dot(aligned_column) < 0.0) {
      aligned_column = -aligned_column;
    }
    (*aligned_basis).col(axis) = aligned_column;
  }
  const double orthogonality_error = MaxOrthogonalityError(*aligned_basis);

  // The paper lists an additional Gram-Schmidt cleanup after axis matching.
  // The checked datasets keep the mapped basis orthogonal up to machine
  // precision, so the runtime path leaves that step disabled and only warns if
  // a future dataset violates this assumption.
  // if (!ApplyGramSchmidtOrthonormalization(aligned_basis)) {
  //   return false;
  // }
  if (orthogonality_error > kAxisOrthogonalityWarningThreshold) {
    std::cerr << "[DCReg][Warning][AxisAlignment][" << label
              << "] mapped basis is not sufficiently orthogonal after axis "
                 "permutation/sign selection (error=" << orthogonality_error
              << "). Re-enable the commented paper step if this happens on a "
                 "real dataset.\n";
  }
  return true;
}

// Core module 1: detect degeneracy from the Schur complements.
//
// Starting from H = J^T J, DCReg forms the rotation and translation Schur
// complements in Eq. (18), diagonalizes them as Eq. (19), and reports the raw
// Schur spectrum plus the coarse condition numbers from Eq. (20). This is the
// spectral observability test used by Algorithm 1.
inline DegeneracyDetection DetectDegeneracy(const Matrix6d& hessian,
                                            const SolverParameters& params) {
  DegeneracyDetection detection;
  const Eigen::JacobiSVD<Matrix6d> full_svd(
      hessian, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const auto singular_values = full_svd.singularValues();
  detection.cond_full =
      singular_values(5) > 1e-12
          ? singular_values(0) / singular_values(5)
          : std::numeric_limits<double>::infinity();

  const Eigen::Matrix3d h_rr = hessian.block<3, 3>(0, 0);
  const Eigen::Matrix3d h_rt = hessian.block<3, 3>(0, 3);
  const Eigen::Matrix3d h_tr = hessian.block<3, 3>(3, 0);
  const Eigen::Matrix3d h_tt = hessian.block<3, 3>(3, 3);

  const Eigen::FullPivLU<Eigen::Matrix3d> lu_rr(h_rr);
  const Eigen::FullPivLU<Eigen::Matrix3d> lu_tt(h_tt);
  if (!lu_rr.isInvertible() || !lu_tt.isInvertible()) {
    return detection;
  }

  // Eq. (18): decouple rotation and translation observability.
  const Eigen::Matrix3d schur_rot = h_rr - h_rt * lu_tt.inverse() * h_tr;
  const Eigen::Matrix3d schur_trans = h_tt - h_tr * lu_rr.inverse() * h_rt;

  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> rot_solver(schur_rot);
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> trans_solver(schur_trans);
  if (rot_solver.info() != Eigen::Success ||
      trans_solver.info() != Eigen::Success) {
    return detection;
  }

  detection.factorization_ok = true;
  detection.lambda_schur_rot = rot_solver.eigenvalues();
  detection.lambda_schur_trans = trans_solver.eigenvalues();
  detection.raw_rot_basis = rot_solver.eigenvectors();
  detection.raw_trans_basis = trans_solver.eigenvectors();
  detection.cond_schur_rot =
      detection.lambda_schur_rot.maxCoeff() /
      std::max(detection.lambda_schur_rot.minCoeff(), 1e-12);
  detection.cond_schur_trans =
      detection.lambda_schur_trans.maxCoeff() /
      std::max(detection.lambda_schur_trans.minCoeff(), 1e-12);
  return detection;
}

// Core module 2: characterize degeneracy in physical coordinates.
//
// Algorithm 2 turns the raw Schur eigenvectors into physically meaningful
// [roll, pitch, yaw, x, y, z] directions, tests each aligned direction with the
// threshold in Eq. (21), and then clamps only the weak eigenvalues according to
// Eq. (46) before building the block-diagonal preconditioner in Eqs. (43)-(44).
inline DegeneracyCharacterization CharacterizeDegeneracy(
    const DegeneracyDetection& detection, const SolverParameters& params) {
  DegeneracyCharacterization characterization;
  characterization.factorization_ok = detection.factorization_ok;
  characterization.cond_full = detection.cond_full;
  characterization.cond_schur_rot = detection.cond_schur_rot;
  characterization.cond_schur_trans = detection.cond_schur_trans;
  characterization.lambda_schur_rot = detection.lambda_schur_rot;
  characterization.lambda_schur_trans = detection.lambda_schur_trans;

  if (!detection.factorization_ok) {
    characterization.is_degenerate = true;
    characterization.degenerate_mask = {true, true, true, true, true, true};
    return characterization;
  }

  const std::array<Eigen::Vector3d, 3> refs = {
      Eigen::Vector3d::UnitX(),
      Eigen::Vector3d::UnitY(),
      Eigen::Vector3d::UnitZ(),
  };
  if (!AlignEigenBasisToAxes(detection.raw_rot_basis, refs, "rot",
                             &characterization.aligned_rot_basis,
                             &characterization.rot_indices)) {
    characterization.aligned_rot_basis = detection.raw_rot_basis;
    characterization.rot_indices = {0, 1, 2};
  }
  if (!AlignEigenBasisToAxes(detection.raw_trans_basis, refs, "trans",
                             &characterization.aligned_trans_basis,
                             &characterization.trans_indices)) {
    characterization.aligned_trans_basis = detection.raw_trans_basis;
    characterization.trans_indices = {0, 1, 2};
  }

  // Each column is an aligned Schur eigenvector expressed in physical
  // coordinates. Squared coefficients provide an intuitive per-axis
  // contribution ratio whose column sum is one.
  characterization.rot_axis_contribution_ratio =
      characterization.aligned_rot_basis.cwiseProduct(
          characterization.aligned_rot_basis);
  characterization.trans_axis_contribution_ratio =
      characterization.aligned_trans_basis.cwiseProduct(
          characterization.aligned_trans_basis);

  for (int axis = 0; axis < 3; ++axis) {
    characterization.aligned_lambda_schur_rot(axis) =
        detection.lambda_schur_rot(characterization.rot_indices[axis]);
    characterization.aligned_lambda_schur_trans(axis) =
        detection.lambda_schur_trans(characterization.trans_indices[axis]);
  }

  characterization.clamped_lambda_schur_rot =
      characterization.aligned_lambda_schur_rot;
  characterization.clamped_lambda_schur_trans =
      characterization.aligned_lambda_schur_trans;

  const double rot_max = characterization.aligned_lambda_schur_rot.maxCoeff();
  const double trans_max =
      characterization.aligned_lambda_schur_trans.maxCoeff();
  const double min_rot = std::max(rot_max / params.kappa_target, 1e-9);
  const double min_trans = std::max(trans_max / params.kappa_target, 1e-9);
  for (int axis = 0; axis < 3; ++axis) {
    const double rot_condition =
        rot_max /
        std::max(characterization.aligned_lambda_schur_rot(axis), 1e-12);
    if (rot_condition > params.degeneracy_condition_threshold) {
      characterization.is_degenerate = true;
      characterization.degenerate_mask[axis] = true;
      characterization.clamped_lambda_schur_rot(axis) = min_rot;
    }

    const double trans_condition =
        trans_max /
        std::max(characterization.aligned_lambda_schur_trans(axis), 1e-12);
    if (trans_condition > params.degeneracy_condition_threshold) {
      characterization.is_degenerate = true;
      characterization.degenerate_mask[axis + 3] = true;
      characterization.clamped_lambda_schur_trans(axis) = min_trans;
    }
  }

  characterization.preconditioner.setZero();
  characterization.preconditioner.block<3, 3>(0, 0) =
      characterization.aligned_rot_basis *
      characterization.clamped_lambda_schur_rot.cwiseMax(1e-9)
          .cwiseInverse()
          .asDiagonal() *
      characterization.aligned_rot_basis.transpose();
  characterization.preconditioner.block<3, 3>(3, 3) =
      characterization.aligned_trans_basis *
      characterization.clamped_lambda_schur_trans.cwiseMax(1e-9)
          .cwiseInverse()
          .asDiagonal() *
      characterization.aligned_trans_basis.transpose();
  return characterization;
}

// Solves the raw ICP normal equation H delta = b with a direct QR factorization.
//
// This is the baseline update used by the None branch. It intentionally does
// not inspect degeneracy, apply Schur analysis, or build a DCReg
// preconditioner.
inline Vector6d SolveRawNormalEquation(const LinearSystem& system) {
  return system.hessian.colPivHouseholderQr().solve(system.rhs);
}

// Core module 3: solve the preconditioned normal equation.
//
// DCReg uses the characterized weak directions to build P and then applies PCG
// to the preconditioned system in Eqs. (41)-(42). If the Schur factorization
// or the PCG iteration becomes numerically unreliable, the solver falls back to
// the dense QR update so the registration loop still receives a finite step.
inline LinearSolveReport SolvePreconditionedUpdate(
    const LinearSystem& system, const SolverParameters& params,
    const DegeneracyCharacterization& characterization) {
  LinearSolveReport report;
  report.used_preconditioned_solver = true;

  if (!characterization.factorization_ok ||
      !characterization.preconditioner.allFinite()) {
    report.delta = SolveRawNormalEquation(system);
    report.converged = report.delta.allFinite();
    report.used_qr_fallback = true;
    return report;
  }

  const Matrix6d hessian =
      0.5 * (system.hessian + system.hessian.transpose());
  const double rhs_norm = system.rhs.norm();
  if (rhs_norm < 1e-12) {
    report.converged = true;
    report.relative_residual = 0.0;
    return report;
  }

  Vector6d delta = Vector6d::Zero();
  Vector6d residual = system.rhs;
  Vector6d z = characterization.preconditioner * residual;
  if (!z.allFinite()) {
    report.delta = SolveRawNormalEquation(system);
    report.converged = report.delta.allFinite();
    report.used_qr_fallback = true;
    return report;
  }

  Vector6d direction = z;
  double rz_old = residual.dot(z);
  if (!std::isfinite(rz_old) || std::abs(rz_old) < 1e-20) {
    report.delta = SolveRawNormalEquation(system);
    report.converged = report.delta.allFinite();
    report.used_qr_fallback = true;
    return report;
  }

  const double target_residual =
      params.pcg_tolerance * std::max(1.0, rhs_norm);
  for (int iteration = 0; iteration < std::max(params.pcg_max_iterations, 6);
       ++iteration) {
    const Vector6d hessian_direction = hessian * direction;
    const double denom = direction.dot(hessian_direction);
    if (!std::isfinite(denom) || std::abs(denom) < 1e-20) {
      break;
    }

    // Algorithm 3: alpha_k = (r_k^T z_k) / (p_k^T H p_k).
    const double alpha = rz_old / denom;
    if (!std::isfinite(alpha)) {
      break;
    }

    delta += alpha * direction;
    residual -= alpha * hessian_direction;
    report.iterations = iteration + 1;
    report.relative_residual =
        residual.norm() / std::max(1.0, rhs_norm);
    if (!delta.allFinite() || !residual.allFinite()) {
      break;
    }
    if (residual.norm() <= target_residual) {
      report.delta = delta;
      report.converged = true;
      return report;
    }

    const Vector6d z_next = characterization.preconditioner * residual;
    const double rz_new = residual.dot(z_next);
    if (!z_next.allFinite() || !std::isfinite(rz_new) ||
        std::abs(rz_old) < 1e-20) {
      break;
    }

    const double beta = rz_new / rz_old;
    if (!std::isfinite(beta)) {
      break;
    }

    direction = z_next + beta * direction;
    z = z_next;
    rz_old = rz_new;
  }

  if (delta.allFinite() && residual.allFinite()) {
    report.delta = delta;
    report.converged = report.relative_residual <= params.pcg_tolerance;
    if (report.converged) {
      return report;
    }
  }

  report.delta = SolveRawNormalEquation(system);
  report.converged = report.delta.allFinite();
  report.used_qr_fallback = true;
  report.relative_residual = std::numeric_limits<double>::quiet_NaN();
  return report;
}

inline LinearSystem ReduceNormalEquations(
    const Eigen::Matrix<double, Eigen::Dynamic, 6>& jacobians,
    const Eigen::VectorXd& weighted_residuals, ParallelMode mode) {
  LinearSystem system;
  const int row_count = static_cast<int>(jacobians.rows());

#if DCREG_USE_TBB
  if (mode == ParallelMode::kTbb) {
    struct TbbReducer {
      const Eigen::Matrix<double, Eigen::Dynamic, 6>& jacobians;
      const Eigen::VectorXd& residuals;
      Matrix6d hessian = Matrix6d::Zero();
      Vector6d rhs = Vector6d::Zero();

      TbbReducer(const Eigen::Matrix<double, Eigen::Dynamic, 6>& jacobians_in,
                 const Eigen::VectorXd& residuals_in)
          : jacobians(jacobians_in), residuals(residuals_in) {}

      TbbReducer(TbbReducer& other, tbb::split)
          : jacobians(other.jacobians), residuals(other.residuals) {}

      void operator()(const tbb::blocked_range<int>& range) {
        for (int row = range.begin(); row != range.end(); ++row) {
          const Eigen::Matrix<double, 1, 6> jacobian_row = jacobians.row(row);
          hessian.noalias() += jacobian_row.transpose() * jacobian_row;
          rhs.noalias() += jacobian_row.transpose() * residuals(row);
        }
      }

      void join(const TbbReducer& other) {
        hessian += other.hessian;
        rhs += other.rhs;
      }
    };

    TbbReducer reducer(jacobians, weighted_residuals);
    tbb::parallel_reduce(tbb::blocked_range<int>(0, row_count, 128), reducer);
    system.hessian = reducer.hessian;
    system.rhs = reducer.rhs;
    return system;
  }
#endif

#ifdef _OPENMP
  if (mode == ParallelMode::kOpenMP) {
#pragma omp parallel
    {
      Matrix6d local_hessian = Matrix6d::Zero();
      Vector6d local_rhs = Vector6d::Zero();
#pragma omp for nowait schedule(static)
      for (int row = 0; row < row_count; ++row) {
        const Eigen::Matrix<double, 1, 6> jacobian_row = jacobians.row(row);
        local_hessian.noalias() += jacobian_row.transpose() * jacobian_row;
        local_rhs.noalias() += jacobian_row.transpose() * weighted_residuals(row);
      }
#pragma omp critical
      {
        system.hessian += local_hessian;
        system.rhs += local_rhs;
      }
    }
    return system;
  }
#endif

  system.hessian.noalias() = jacobians.transpose() * jacobians;
  system.rhs.noalias() = jacobians.transpose() * weighted_residuals;
  return system;
}

inline bool FitPlaneFromNeighbors(
    const PointCloud& target, const std::vector<int>& neighbor_indices,
    double max_plane_thickness, Eigen::Vector3d* normal_world,
    double* plane_offset) {
  Eigen::Matrix<double, Eigen::Dynamic, 3> a_matrix(neighbor_indices.size(), 3);
  Eigen::VectorXd b_vector =
      Eigen::VectorXd::Constant(neighbor_indices.size(), -1.0);
  for (int row = 0; row < static_cast<int>(neighbor_indices.size()); ++row) {
    const PointT& point = target.points[neighbor_indices[row]];
    a_matrix.row(row) =
        Eigen::Vector3d(point.x, point.y, point.z).transpose();
  }

  const Eigen::Vector3d plane_coefficients =
      a_matrix.colPivHouseholderQr().solve(b_vector);
  const double coeff_norm = plane_coefficients.norm();
  if (coeff_norm < 1e-6) {
    return false;
  }

  *normal_world = plane_coefficients / coeff_norm;
  *plane_offset = 1.0 / coeff_norm;

  double max_residual = 0.0;
  for (const int index : neighbor_indices) {
    const PointT& point = target.points[index];
    const Eigen::Vector3d xyz(point.x, point.y, point.z);
    max_residual = std::max(max_residual,
                            std::abs(normal_world->dot(xyz) + *plane_offset));
  }
  return max_residual < max_plane_thickness;
}

// Correspondence construction.
//
// For each transformed source point, we fit a local plane from its k-nearest
// target neighbors by solving
//   [x_j y_j z_j] [a b c]^T = -1.
// After normalization, the plane is written as
//   n^T x + d = 0.
//
// The scalar residual is
//   r_i = n_i^T x_i + d_i,
// and the robust weight follows the original implementation,
//   w_i = max(0, 1 - alpha |r_i|).
inline std::vector<Correspondence> CollectCorrespondences(
    const PointCloud& source, const PointCloud& target,
    pcl::KdTreeFLANN<PointT>* target_kdtree, const Eigen::Matrix4d& transform,
    const SolverParameters& params, ParallelMode mode, int* near_neighbor_count,
    double* residual_square_sum) {
  std::vector<Correspondence> candidates(source.size());
  std::vector<std::uint8_t> is_valid(source.size(), 0);

  const double radius_sq = params.search_radius * params.search_radius;
  const int k = params.plane_fit_neighbors;
  ParallelFor(0, static_cast<int>(source.size()), mode, [&](int index) {
    const PointT& source_point = source.points[index];
    PointT transformed_point;
    const Eigen::Vector3d world = TransformPoint(transform, source_point);
    transformed_point.x = static_cast<float>(world.x());
    transformed_point.y = static_cast<float>(world.y());
    transformed_point.z = static_cast<float>(world.z());

    std::vector<int> neighbor_indices(k);
    std::vector<float> neighbor_distances(k);
    // Keep only neighborhoods with exactly k supports inside the search radius.
    if (target_kdtree->nearestKSearch(transformed_point, k, neighbor_indices,
                                      neighbor_distances) != k ||
        neighbor_distances.back() >= radius_sq) {
      return;
    }

    Eigen::Vector3d normal_world;
    double plane_offset = 0.0;
    if (!FitPlaneFromNeighbors(target, neighbor_indices, params.max_plane_thickness,
                               &normal_world, &plane_offset)) {
      return;
    }

    // r_i = n_i^T x_i + d_i, with x_i already transformed into the target frame.
    const double residual = normal_world.dot(world) + plane_offset;
    // Original robust weight: w_i = max(0, 1 - alpha |r_i|).
    const double weight =
        std::max(0.0, 1.0 - params.weight_slope * std::abs(residual));
    if (weight <= params.min_weight) {
      return;
    }

    Correspondence corr;
    corr.point_body = Eigen::Vector3d(source_point.x, source_point.y, source_point.z);
    corr.normal_world = normal_world;
    corr.residual = residual;
    corr.weight = weight;
    if (params.use_weight_derivative && weight < 1.0 && weight > 0.0) {
      // d w_i / d r_i for the piecewise-linear robust weight. This term is
      // needed because the linear system is built from e_i = w_i(r_i) r_i
      // rather than from a constant-weight residual.
      corr.weight_derivative =
          -params.weight_slope * (residual >= 0.0 ? 1.0 : -1.0);
    }
    candidates[index] = corr;
    is_valid[index] = 1;
  });

  std::vector<Correspondence> correspondences;
  correspondences.reserve(source.size());
  *near_neighbor_count = 0;
  *residual_square_sum = 0.0;
  for (std::size_t index = 0; index < source.size(); ++index) {
    if (!is_valid[index]) {
      continue;
    }
    ++(*near_neighbor_count);
    *residual_square_sum +=
        candidates[index].residual * candidates[index].residual;
    correspondences.push_back(candidates[index]);
  }
  return correspondences;
}

template <typename JacobianFn>
inline LinearSystem BuildWeightedLinearSystem(
    const std::vector<Correspondence>& correspondences, ParallelMode mode,
    const JacobianFn& jacobian_of) {
  Eigen::Matrix<double, Eigen::Dynamic, 6> jacobians(correspondences.size(), 6);
  Eigen::VectorXd weighted_residuals(correspondences.size());

  ParallelFor(0, static_cast<int>(correspondences.size()), mode, [&](int index) {
    const Correspondence& corr = correspondences[index];
    const Eigen::Matrix<double, 1, 6> residual_jacobian = jacobian_of(corr);
    // J_i = d(w_i r_i) / dxi = (w_i + r_i d w_i / d r_i) d r_i / dxi.
    const Eigen::Matrix<double, 1, 6> full_jacobian =
        corr.weight * residual_jacobian +
        corr.residual * corr.weight_derivative * residual_jacobian;
    jacobians.row(index) = full_jacobian;
    // b_i = -J_i^T e_i with e_i = w_i r_i, accumulated later as J^T b.
    weighted_residuals(index) = -corr.weight * corr.residual;
  });

  return ReduceNormalEquations(jacobians, weighted_residuals, mode);
}

// SE(3) linearization with right-invariant perturbation.
inline LinearSystem BuildSe3LinearSystem(
    const std::vector<Correspondence>& correspondences, const Se3State& state,
    ParallelMode mode) {
  return BuildWeightedLinearSystem(
      correspondences, mode, [&](const Correspondence& corr) {
        return ComputeSe3PointToPlaneJacobian(corr.point_body, corr.normal_world,
                                              state.rotation);
      });
}

// SO(3)+R^3 linearization.
//
// The weighted objective is
//   E = 1/2 sum_i (w_i(r_i) r_i)^2.
// Therefore the row Jacobian is
//   J_i = d(w_i r_i) / dxi
//       = (w_i + r_i dw_i / dr_i) dr_i / dxi.
//
// In the historical DCReg SO(3) implementation, the translational increment is
// still applied in the local body frame through t <- t + R v, so the last
// three Jacobian columns remain n_i^T R.
inline LinearSystem BuildSo3LinearSystem(
    const std::vector<Correspondence>& correspondences, const So3State& state,
    ParallelMode mode) {
  return BuildWeightedLinearSystem(
      correspondences, mode, [&](const Correspondence& corr) {
        return ComputeSo3PointToPlaneJacobian(corr.point_body, corr.normal_world,
                                              state.rotation);
      });
}

// Quaternion linearization.
//
// This path shares the same residual model as the original SO(3) branch; only
// the rotation state/update representation differs.
inline LinearSystem BuildQuaternionLinearSystem(
    const std::vector<Correspondence>& correspondences,
    const QuaternionState& state, ParallelMode mode) {
  const Eigen::Matrix3d rotation = state.rotation.toRotationMatrix();
  return BuildWeightedLinearSystem(
      correspondences, mode, [&](const Correspondence& corr) {
        return ComputeSo3PointToPlaneJacobian(corr.point_body, corr.normal_world,
                                              rotation);
      });
}

// Euler-angle linearization.
//
// This path now differentiates the current Z-Y-X Euler pose model exactly, so
// any remaining performance gap relative to SE3 / SO3 / quaternion comes from
// the parameterization itself rather than from a mismatched Jacobian. The main
// residual risk is the usual Euler-angle chart sensitivity near pitch
// singularities, not an inconsistency with PoseToMatrix(). The pose-dependent
// derivative matrices are cached once per iteration and then reused for every
// correspondence.
inline LinearSystem BuildEulerLinearSystem(
    const std::vector<Correspondence>& correspondences, const Pose6D& pose,
    ParallelMode mode) {
  const EulerLinearizationCache cache = MakeEulerLinearizationCache(pose);
  return BuildWeightedLinearSystem(
      correspondences, mode, [&](const Correspondence& corr) {
        return ComputeEulerPointToPlaneJacobian(corr.point_body, corr.normal_world,
                                                cache);
      });
}

template <typename State, typename TransformFn, typename BuildSystemFn,
          typename UpdateFn>
inline RegistrationResult RunPlainRegistration(
    const PointCloud& source, const PointCloud& target,
    const TestCase& test_case, const RunOptions& options, State state,
    const TransformFn& transform_of, const BuildSystemFn& build_system,
    const UpdateFn& update_state) {
  RegistrationResult result;
  result.resolved_parallel_mode = ResolveParallelMode(options.parallel_mode);

  pcl::KdTreeFLANN<PointT> target_kdtree;
  target_kdtree.setInputCloud(target.makeShared());

  const Eigen::Matrix4d gt = PoseToMatrix(test_case.ground_truth_pose);
  const SolverParameters& params = test_case.params;

  Stopwatch total_timer;
  for (int iteration = 0; iteration < params.max_iterations; ++iteration) {
    int near_neighbor_count = 0;
    double residual_square_sum = 0.0;
    const Eigen::Matrix4d current_transform = transform_of(state);
    const std::vector<Correspondence> correspondences =
        CollectCorrespondences(source, target, &target_kdtree, current_transform,
                               params, result.resolved_parallel_mode,
                               &near_neighbor_count,
                               &residual_square_sum);

    if (static_cast<int>(correspondences.size()) < kMinCorrespondences) {
      result.failure_reason = "too few correspondences";
      result.iterations = iteration;
      result.time_ms = total_timer.ElapsedMilliseconds();
      result.transform = current_transform;
      result.pose_error = CalculatePoseError(gt, result.transform);
      return result;
    }

    const LinearSystem system = build_system(correspondences, state,
                                             result.resolved_parallel_mode);
    const Vector6d delta = SolveRawNormalEquation(system);
    if (!delta.allFinite()) {
      result.failure_reason = "solver produced non-finite update";
      result.iterations = iteration;
      result.time_ms = total_timer.ElapsedMilliseconds();
      result.transform = current_transform;
      result.pose_error = CalculatePoseError(gt, result.transform);
      return result;
    }

    update_state(&state, delta);
    result.transform = transform_of(state);
    result.rmse = std::sqrt(residual_square_sum / correspondences.size());
    result.fitness =
        static_cast<double>(near_neighbor_count) / static_cast<double>(source.size());

    IterationSummary summary;
    summary.iteration = iteration;
    summary.correspondence_count = static_cast<int>(correspondences.size());
    summary.rmse = result.rmse;
    summary.fitness = result.fitness;
    summary.rotation_step_norm = delta.head<3>().norm();
    summary.translation_step_norm = delta.tail<3>().norm();
    result.history.push_back(summary);

    result.iterations = iteration + 1;
    if (summary.rotation_step_norm < params.convergence_rot &&
        summary.translation_step_norm < params.convergence_trans) {
      result.converged = true;
      break;
    }
  }

  result.success = result.converged;
  result.time_ms = total_timer.ElapsedMilliseconds();
  result.pose_error = CalculatePoseError(gt, result.transform);
  if (!result.converged && result.failure_reason.empty()) {
    result.failure_reason = "reached max_iterations without convergence";
  }
  return result;
}

template <typename State, typename TransformFn, typename BuildSystemFn,
          typename UpdateFn>
inline RegistrationResult RunDcRegRegistration(
    const PointCloud& source, const PointCloud& target,
    const TestCase& test_case, const RunOptions& options, State state,
    const TransformFn& transform_of, const BuildSystemFn& build_system,
    const UpdateFn& update_state) {
  RegistrationResult result;
  result.resolved_parallel_mode = ResolveParallelMode(options.parallel_mode);

  pcl::KdTreeFLANN<PointT> target_kdtree;
  target_kdtree.setInputCloud(target.makeShared());

  const Eigen::Matrix4d gt = PoseToMatrix(test_case.ground_truth_pose);
  const SolverParameters& params = test_case.params;

  Stopwatch total_timer;
  for (int iteration = 0; iteration < params.max_iterations; ++iteration) {
    int near_neighbor_count = 0;
    double residual_square_sum = 0.0;
    const Eigen::Matrix4d current_transform = transform_of(state);
    const std::vector<Correspondence> correspondences =
        CollectCorrespondences(source, target, &target_kdtree, current_transform,
                               params, result.resolved_parallel_mode,
                               &near_neighbor_count,
                               &residual_square_sum);

    if (static_cast<int>(correspondences.size()) < kMinCorrespondences) {
      result.failure_reason = "too few correspondences";
      result.iterations = iteration;
      result.time_ms = total_timer.ElapsedMilliseconds();
      result.transform = current_transform;
      result.pose_error = CalculatePoseError(gt, result.transform);
      return result;
    }

    const LinearSystem system = build_system(correspondences, state,
                                             result.resolved_parallel_mode);
    const DegeneracyDetection detection = DetectDegeneracy(system.hessian, params);
    const DegeneracyCharacterization characterization =
        CharacterizeDegeneracy(detection, params);
    const LinearSolveReport solve_report =
        SolvePreconditionedUpdate(system, params, characterization);
    const Vector6d& delta = solve_report.delta;
    if (!delta.allFinite()) {
      result.failure_reason = "solver produced non-finite update";
      result.iterations = iteration;
      result.time_ms = total_timer.ElapsedMilliseconds();
      result.transform = current_transform;
      result.pose_error = CalculatePoseError(gt, result.transform);
      return result;
    }

    update_state(&state, delta);
    result.transform = transform_of(state);
    result.rmse = std::sqrt(residual_square_sum / correspondences.size());
    result.fitness =
        static_cast<double>(near_neighbor_count) / static_cast<double>(source.size());

    IterationSummary summary;
    summary.iteration = iteration;
    summary.correspondence_count = static_cast<int>(correspondences.size());
    summary.rmse = result.rmse;
    summary.fitness = result.fitness;
    summary.rotation_step_norm = delta.head<3>().norm();
    summary.translation_step_norm = delta.tail<3>().norm();
    summary.linear_solver_iterations = solve_report.iterations;
    summary.linear_solver_relative_residual = solve_report.relative_residual;
    summary.used_preconditioned_solver = solve_report.used_preconditioned_solver;
    summary.used_qr_fallback = solve_report.used_qr_fallback;
    summary.is_degenerate = characterization.is_degenerate;
    summary.degenerate_mask = characterization.degenerate_mask;
    summary.schur_cond_rot = characterization.cond_schur_rot;
    summary.schur_cond_trans = characterization.cond_schur_trans;
    result.history.push_back(summary);

    result.iterations = iteration + 1;
    if (summary.rotation_step_norm < params.convergence_rot &&
        summary.translation_step_norm < params.convergence_trans) {
      result.converged = true;
      break;
    }
  }

  result.success = result.converged;
  result.time_ms = total_timer.ElapsedMilliseconds();
  result.pose_error = CalculatePoseError(gt, result.transform);
  if (!result.converged && result.failure_reason.empty()) {
    result.failure_reason = "reached max_iterations without convergence";
  }
  return result;
}

template <typename State, typename TransformFn, typename BuildSystemFn,
          typename UpdateFn>
inline RegistrationResult RunParameterizedRegistration(
    const PointCloud& source, const PointCloud& target,
    const TestCase& test_case, const RunOptions& options, State state,
    const TransformFn& transform_of, const BuildSystemFn& build_system,
    const UpdateFn& update_state) {
  if (options.algorithm == Algorithm::kNone) {
    return RunPlainRegistration(source, target, test_case, options, state,
                                transform_of, build_system, update_state);
  }
  return RunDcRegRegistration(source, target, test_case, options, state,
                              transform_of, build_system, update_state);
}

inline RegistrationResult RunSe3Registration(const PointCloud& source,
                                             const PointCloud& target,
                                             const TestCase& test_case,
                                             const RunOptions& options) {
  return RunParameterizedRegistration(
      source, target, test_case, options, ToSe3State(test_case.initial_pose),
      [](const Se3State& state) { return state.Matrix(); },
      [](const std::vector<Correspondence>& correspondences,
         const Se3State& state, ParallelMode mode) {
        return BuildSe3LinearSystem(correspondences, state, mode);
      },
      [](Se3State* state, const Vector6d& delta) { *state = state->BoxPlus(delta); });
}

inline RegistrationResult RunSo3Registration(const PointCloud& source,
                                             const PointCloud& target,
                                             const TestCase& test_case,
                                             const RunOptions& options) {
  return RunParameterizedRegistration(
      source, target, test_case, options, ToSo3State(test_case.initial_pose),
      [](const So3State& state) { return state.Matrix(); },
      [](const std::vector<Correspondence>& correspondences,
         const So3State& state, ParallelMode mode) {
        return BuildSo3LinearSystem(correspondences, state, mode);
      },
      [](So3State* state, const Vector6d& delta) { *state = state->BoxPlus(delta); });
}

inline RegistrationResult RunQuaternionRegistration(
    const PointCloud& source, const PointCloud& target,
    const TestCase& test_case, const RunOptions& options) {
  return RunParameterizedRegistration(
      source, target, test_case, options,
      ToQuaternionState(test_case.initial_pose),
      [](const QuaternionState& state) { return state.Matrix(); },
      [](const std::vector<Correspondence>& correspondences,
         const QuaternionState& state, ParallelMode mode) {
        return BuildQuaternionLinearSystem(correspondences, state, mode);
      },
      [](QuaternionState* state, const Vector6d& delta) {
        *state = state->BoxPlus(delta);
      });
}

inline RegistrationResult RunEulerRegistration(const PointCloud& source,
                                               const PointCloud& target,
                                               const TestCase& test_case,
                                               const RunOptions& options) {
  return RunParameterizedRegistration(
      source, target, test_case, options, test_case.initial_pose,
      [](const Pose6D& pose) { return PoseToMatrix(pose); },
      [](const std::vector<Correspondence>& correspondences,
         const Pose6D& pose, ParallelMode mode) {
        return BuildEulerLinearSystem(correspondences, pose, mode);
      },
      [](Pose6D* pose, const Vector6d& delta) {
        pose->roll += delta(0);
        pose->pitch += delta(1);
        pose->yaw += delta(2);
        pose->x += delta(3);
        pose->y += delta(4);
        pose->z += delta(5);
      });
}

inline RegistrationResult RunRegistration(const PointCloud& source,
                                          const PointCloud& target,
                                          const TestCase& test_case,
                                          const RunOptions& options) {
  switch (options.parameterization) {
    case Parameterization::kEuler:
      return RunEulerRegistration(source, target, test_case, options);
    case Parameterization::kSE3:
      return RunSe3Registration(source, target, test_case, options);
    case Parameterization::kSO3:
      return RunSo3Registration(source, target, test_case, options);
    case Parameterization::kQuaternion:
      return RunQuaternionRegistration(source, target, test_case, options);
  }
  return RunSo3Registration(source, target, test_case, options);
}

inline RegistrationResult RunRegistration(const TestCase& test_case,
                                          const RunOptions& options) {
  PointCloudPtr source(new PointCloud);
  PointCloudPtr target(new PointCloud);
  RegistrationResult result;

  std::string error_message;
  const std::string source_path = test_case.folder_path + test_case.source_pcd;
  const std::string target_path = test_case.folder_path + test_case.target_pcd;
  if (!LoadPointCloud(source_path, source, &error_message)) {
    result.failure_reason = error_message;
    return result;
  }
  if (!LoadPointCloud(target_path, target, &error_message)) {
    result.failure_reason = error_message;
    return result;
  }
  return RunRegistration(*source, *target, test_case, options);
}

inline void PrintSummary(const TestCase& test_case, const RunOptions& options,
                         const RegistrationResult& result) {
  std::cout << "Test case: " << test_case.name << '\n';
  std::cout << "Algorithm: " << ToString(options.algorithm)
            << " | Parameterization: " << ToString(options.parameterization)
            << " | Parallel: " << ToString(result.resolved_parallel_mode) << '\n';
  if (!result.success) {
    std::cout << "Status: failed";
    if (!result.failure_reason.empty()) {
      std::cout << " (" << result.failure_reason << ")";
    }
    std::cout << '\n';
    return;
  }

  std::cout << "Status: converged in " << result.iterations << " iterations\n";
  std::cout << "RMSE: " << result.rmse << " | Fitness: " << result.fitness
            << " | Time(ms): " << result.time_ms << '\n';
  std::cout << "Pose error: translation=" << result.pose_error.translation_m
            << " m, rotation=" << result.pose_error.rotation_deg << " deg\n";
  if (!result.history.empty() && options.algorithm == Algorithm::kDCReg) {
    const IterationSummary& last = result.history.back();
    std::cout << "DCReg solver: pcg_iterations="
              << last.linear_solver_iterations
              << ", relative_residual=" << last.linear_solver_relative_residual
              << ", qr_fallback=" << (last.used_qr_fallback ? 1 : 0) << '\n';
    std::cout << "Observability: schur_rot=" << last.schur_cond_rot
              << ", schur_trans=" << last.schur_cond_trans << ", mask=";
    for (const bool value : last.degenerate_mask) {
      std::cout << (value ? '1' : '0');
    }
    std::cout << '\n';
  }
  std::cout << "Final transform:\n" << result.transform << '\n';
}

}  // namespace dcreg

#endif  // DCREG_DCREG_HPP_
