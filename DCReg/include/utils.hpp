#ifndef DCREG_UTILS_HPP_
#define DCREG_UTILS_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#if defined(DCREG_HAS_TBB) && __has_include(<tbb/blocked_range.h>) && \
    __has_include(<tbb/parallel_for.h>) && __has_include(<tbb/parallel_reduce.h>)
#define DCREG_USE_TBB 1
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#else
#define DCREG_USE_TBB 0
#endif

namespace dcreg {

using PointT = pcl::PointXYZ;
using PointCloud = pcl::PointCloud<PointT>;
using PointCloudPtr = PointCloud::Ptr;
using PointCloudConstPtr = PointCloud::ConstPtr;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;

constexpr int kMinCorrespondences = 10;

#ifndef DCREG_SOURCE_DIR
#define DCREG_SOURCE_DIR "."
#endif

enum class Algorithm {
  kNone,
  kDCReg,
};

enum class Parameterization {
  kEuler,
  kSE3,
  kSO3,
  kQuaternion,
};

enum class ParallelMode {
  kAuto,
  kSerial,
  kOpenMP,
  kTbb,
};

struct Pose6D {
  double roll = 0.0;
  double pitch = 0.0;
  double yaw = 0.0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

struct PoseError {
  double translation_m = 0.0;
  double rotation_deg = 0.0;
};

struct SolverParameters {
  double search_radius = 1.0;
  int max_iterations = 30;
  double convergence_rot = 1e-5;
  double convergence_trans = 1e-3;

  // DCReg keeps only the Schur-based condition threshold and PCG settings.
  double degeneracy_condition_threshold = 10.0;
  double kappa_target = 10.0;
  double pcg_tolerance = 1e-6;
  int pcg_max_iterations = 10;

  // Point-to-plane construction constants kept consistent with the original code.
  int plane_fit_neighbors = 5;
  double max_plane_thickness = 0.2;
  double weight_slope = 0.9;
  double min_weight = 0.1;
  bool use_weight_derivative = true;
};

struct TestCase {
  std::string name;
  std::string folder_path;
  std::string source_pcd;
  std::string target_pcd;
  Pose6D initial_pose;
  Pose6D ground_truth_pose;
  bool has_ground_truth_pose = true;
  SolverParameters params;
};

struct RunOptions {
  Algorithm algorithm = Algorithm::kDCReg;
  Parameterization parameterization = Parameterization::kSE3;
  ParallelMode parallel_mode = ParallelMode::kAuto;
  bool verbose = true;
};

struct IterationSummary {
  int iteration = 0;
  int correspondence_count = 0;
  double rmse = std::numeric_limits<double>::quiet_NaN();
  double fitness = 0.0;
  double rotation_step_norm = 0.0;
  double translation_step_norm = 0.0;
  int linear_solver_iterations = 0;
  double linear_solver_relative_residual =
      std::numeric_limits<double>::quiet_NaN();
  bool used_preconditioned_solver = false;
  bool used_qr_fallback = false;
  bool is_degenerate = false;
  std::array<bool, 6> degenerate_mask = {false, false, false, false, false, false};
  double schur_cond_rot = std::numeric_limits<double>::quiet_NaN();
  double schur_cond_trans = std::numeric_limits<double>::quiet_NaN();
};

struct RegistrationResult {
  bool success = false;
  bool converged = false;
  bool has_pose_error = false;
  int iterations = 0;
  double time_ms = 0.0;
  double rmse = std::numeric_limits<double>::quiet_NaN();
  double fitness = 0.0;
  PoseError pose_error;
  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  ParallelMode resolved_parallel_mode = ParallelMode::kSerial;
  std::vector<IterationSummary> history;
  std::string failure_reason;
};

class Stopwatch {
 public:
  Stopwatch() { Reset(); }

  void Reset() { start_ = std::chrono::steady_clock::now(); }

  double ElapsedMilliseconds() const {
    const auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(now - start_).count();
  }

 private:
  std::chrono::steady_clock::time_point start_;
};

inline double DegToRad(double value_deg) { return value_deg * M_PI / 180.0; }

inline double RadToDeg(double value_rad) { return value_rad * 180.0 / M_PI; }

inline Pose6D PoseFromDegrees(double x, double y, double z, double roll_deg,
                              double pitch_deg, double yaw_deg) {
  Pose6D pose;
  pose.x = x;
  pose.y = y;
  pose.z = z;
  pose.roll = DegToRad(roll_deg);
  pose.pitch = DegToRad(pitch_deg);
  pose.yaw = DegToRad(yaw_deg);
  return pose;
}

inline Pose6D PoseFromRadians(double x, double y, double z, double roll,
                              double pitch, double yaw) {
  Pose6D pose;
  pose.x = x;
  pose.y = y;
  pose.z = z;
  pose.roll = roll;
  pose.pitch = pitch;
  pose.yaw = yaw;
  return pose;
}

inline Eigen::Matrix4d PoseToMatrix(const Pose6D& pose) {
  const Eigen::Translation3d translation(pose.x, pose.y, pose.z);
  const Eigen::AngleAxisd rot_x(pose.roll, Eigen::Vector3d::UnitX());
  const Eigen::AngleAxisd rot_y(pose.pitch, Eigen::Vector3d::UnitY());
  const Eigen::AngleAxisd rot_z(pose.yaw, Eigen::Vector3d::UnitZ());
  return (translation * rot_z * rot_y * rot_x).matrix();
}

inline Pose6D MatrixToPose(const Eigen::Matrix4d& transform) {
  Pose6D pose;
  pose.x = transform(0, 3);
  pose.y = transform(1, 3);
  pose.z = transform(2, 3);

  const Eigen::Matrix3d rotation = transform.block<3, 3>(0, 0);
  const Eigen::Quaterniond q(rotation);

  const double siny_cosp = 2.0 * (q.w() * q.z() + q.x() * q.y());
  const double cosy_cosp = 1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());
  pose.yaw = std::atan2(siny_cosp, cosy_cosp);

  const double sinp = 2.0 * (q.w() * q.y() - q.z() * q.x());
  pose.pitch = std::abs(sinp) >= 1.0 ? std::copysign(M_PI / 2.0, sinp)
                                     : std::asin(sinp);

  const double sinr_cosp = 2.0 * (q.w() * q.x() + q.y() * q.z());
  const double cosr_cosp = 1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y());
  pose.roll = std::atan2(sinr_cosp, cosr_cosp);
  return pose;
}

inline PoseError CalculatePoseError(const Eigen::Matrix4d& gt,
                                    const Eigen::Matrix4d& estimate) {
  PoseError error;
  const Eigen::Matrix4d relative = gt.inverse() * estimate;
  error.translation_m = relative.block<3, 1>(0, 3).norm();
  const Eigen::AngleAxisd angle_axis(relative.block<3, 3>(0, 0));
  error.rotation_deg = RadToDeg(std::abs(angle_axis.angle()));
  return error;
}

inline Eigen::Matrix3d Skew(const Eigen::Vector3d& vector) {
  Eigen::Matrix3d skew;
  skew << 0.0, -vector.z(), vector.y(), vector.z(), 0.0, -vector.x(),
      -vector.y(), vector.x(), 0.0;
  return skew;
}

inline Eigen::Matrix3d So3Exp(const Eigen::Vector3d& omega) {
  const double theta = omega.norm();
  if (theta < 1e-10) {
    return Eigen::Matrix3d::Identity() + Skew(omega);
  }

  const Eigen::Vector3d axis = omega / theta;
  const Eigen::Matrix3d hat = Skew(axis);
  return Eigen::Matrix3d::Identity() + std::sin(theta) * hat +
         (1.0 - std::cos(theta)) * hat * hat;
}

inline Eigen::Matrix3d So3LeftJacobian(const Eigen::Vector3d& omega) {
  const double theta = omega.norm();
  if (theta < 1e-10) {
    return Eigen::Matrix3d::Identity() + 0.5 * Skew(omega);
  }

  const Eigen::Matrix3d omega_hat = Skew(omega);
  const double theta2 = theta * theta;
  const double theta3 = theta2 * theta;
  return Eigen::Matrix3d::Identity() +
         ((1.0 - std::cos(theta)) / theta2) * omega_hat +
         ((theta - std::sin(theta)) / theta3) * omega_hat * omega_hat;
}

inline Eigen::Matrix4d MakeTransform(const Eigen::Matrix3d& rotation,
                                     const Eigen::Vector3d& translation) {
  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  transform.block<3, 3>(0, 0) = rotation;
  transform.block<3, 1>(0, 3) = translation;
  return transform;
}

struct Se3State {
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();

  Eigen::Matrix4d Matrix() const {
    return MakeTransform(rotation, translation);
  }

  // Right-invariant SE(3) update:
  //   T <- T Exp([omega, rho]^).
  // The translational increment is mapped by the SE(3) V(omega) matrix, which
  // distinguishes this path from the lighter SO(3)+R^3 parameterization below.
  Se3State BoxPlus(const Vector6d& delta) const {
    Se3State next = *this;
    next.rotation = rotation * So3Exp(delta.head<3>());
    next.translation =
        translation + rotation * (So3LeftJacobian(delta.head<3>()) *
                                  delta.tail<3>());
    return next;
  }
};

struct So3State {
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();

  Eigen::Matrix4d Matrix() const {
    return MakeTransform(rotation, translation);
  }

  // Original project SO(3) update:
  //   R <- R Exp(omega),  t <- t + R v.
  // This keeps the lighter SO(3) rotation update while expressing the
  // translational increment in the local body frame, which is the convention
  // used by the historical DCReg runner and logs.
  So3State BoxPlus(const Vector6d& delta) const {
    So3State next = *this;
    next.rotation = rotation * So3Exp(delta.head<3>());
    next.translation = translation + rotation * delta.tail<3>();
    return next;
  }
};

struct QuaternionState {
  Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();

  Eigen::Matrix4d Matrix() const {
    return MakeTransform(rotation.toRotationMatrix(), translation);
  }

  // Quaternion manifold update:
  //   q <- q * delta_q(omega),  t <- t + R(q) v.
  // We keep the same local-frame translational increment as the original SO(3)
  // branch so the only difference is the rotation parameterization itself.
  QuaternionState BoxPlus(const Vector6d& delta) const {
    QuaternionState next = *this;
    next.rotation =
        Eigen::Quaterniond(rotation.toRotationMatrix() * So3Exp(delta.head<3>()));
    next.rotation.normalize();
    next.translation = translation + rotation.toRotationMatrix() * delta.tail<3>();
    return next;
  }
};

inline Se3State ToSe3State(const Pose6D& pose) {
  const Eigen::Matrix4d transform = PoseToMatrix(pose);
  Se3State state;
  state.rotation = transform.block<3, 3>(0, 0);
  state.translation = transform.block<3, 1>(0, 3);
  return state;
}

inline So3State ToSo3State(const Pose6D& pose) {
  const Eigen::Matrix4d transform = PoseToMatrix(pose);
  So3State state;
  state.rotation = transform.block<3, 3>(0, 0);
  state.translation = transform.block<3, 1>(0, 3);
  return state;
}

inline QuaternionState ToQuaternionState(const Pose6D& pose) {
  const Eigen::Matrix4d transform = PoseToMatrix(pose);
  QuaternionState state;
  state.rotation = Eigen::Quaterniond(transform.block<3, 3>(0, 0));
  state.rotation.normalize();
  state.translation = transform.block<3, 1>(0, 3);
  return state;
}

inline Eigen::Vector3d TransformPoint(const Eigen::Matrix4d& transform,
                                      const PointT& point) {
  const Eigen::Vector3d input(point.x, point.y, point.z);
  return transform.block<3, 3>(0, 0) * input + transform.block<3, 1>(0, 3);
}

// Point-to-plane residual used by the right-invariant SE(3) formulation:
//   r_i = n_i^T (R p_i + t - q_i).
//
// With a right-multiplicative perturbation
//   T <- T * Exp([omega, v]),
// the first-order derivative becomes
//   dr_i / d[omega, v] =
//   [ -n_i^T R [p_i]_x, n_i^T R ].
//
// This is the Jacobian of the geometric residual r_i only. The weighted
// residual e_i = w_i(r_i) r_i is assembled later in BuildSo3LinearSystem().
inline Eigen::Matrix<double, 1, 6> ComputeSe3PointToPlaneJacobian(
    const Eigen::Vector3d& point_body, const Eigen::Vector3d& normal_world,
    const Eigen::Matrix3d& rotation_world_body) {
  Eigen::Matrix<double, 1, 6> jacobian;
  // Rotation block: -n_i^T R [p_i]_x.
  jacobian.block<1, 3>(0, 0) =
      -normal_world.transpose() * rotation_world_body * Skew(point_body);
  // Translation block: n_i^T R because the perturbation translation lives in
  // the local body frame under the right-invariant SE(3) update.
  jacobian.block<1, 3>(0, 3) = normal_world.transpose() * rotation_world_body;
  return jacobian;
}

// Point-to-plane residual used by the original SO(3) and quaternion
// formulations:
//   r_i = n_i^T (R p_i + t - q_i),
// with update
//   R <- R Exp(omega),  t <- t + R v.
//
// The translational increment is expressed in the local body frame, so the
// first-order derivative with respect to v is n_i^T R. This matches the
// custom SO(3) implementation used by the original DCReg project.
inline Eigen::Matrix<double, 1, 6> ComputeSo3PointToPlaneJacobian(
    const Eigen::Vector3d& point_body, const Eigen::Vector3d& normal_world,
    const Eigen::Matrix3d& rotation_world_body) {
  Eigen::Matrix<double, 1, 6> jacobian;
  // Rotation block: same first-order term as the SE(3) manifold case.
  jacobian.block<1, 3>(0, 0) =
      -normal_world.transpose() * rotation_world_body * Skew(point_body);
  // Translation block: n_i^T R because the increment v is applied through
  // t <- t + R v in the historical SO(3)+R^3 implementation.
  jacobian.block<1, 3>(0, 3) = normal_world.transpose() * rotation_world_body;
  return jacobian;
}

struct EulerLinearizationCache {
  Eigen::Matrix3d d_rotation_droll = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d d_rotation_dpitch = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d d_rotation_dyaw = Eigen::Matrix3d::Zero();
};

// Precomputes the Euler rotation derivatives for one Gauss-Newton iteration.
//
// For R(roll, pitch, yaw) = Rz(yaw) Ry(pitch) Rx(roll), the cached matrices are
//   dR / droll  = Rz Ry (dRx / droll),
//   dR / dpitch = Rz (dRy / dpitch) Rx,
//   dR / dyaw   = (dRz / dyaw) Ry Rx.
//
// BuildEulerLinearSystem() reuses this cache for every correspondence so the
// per-point loop does not repeatedly construct Rx / Ry / Rz and their
// derivatives.
inline EulerLinearizationCache MakeEulerLinearizationCache(
    const Pose6D& pose) {
  const double sr = std::sin(pose.roll);
  const double cr = std::cos(pose.roll);
  const double sp = std::sin(pose.pitch);
  const double cp = std::cos(pose.pitch);
  const double sy = std::sin(pose.yaw);
  const double cy = std::cos(pose.yaw);

  const Eigen::Matrix3d rx =
      (Eigen::Matrix3d() << 1.0, 0.0, 0.0, 0.0, cr, -sr, 0.0, sr, cr)
          .finished();
  const Eigen::Matrix3d ry =
      (Eigen::Matrix3d() << cp, 0.0, sp, 0.0, 1.0, 0.0, -sp, 0.0, cp)
          .finished();
  const Eigen::Matrix3d rz =
      (Eigen::Matrix3d() << cy, -sy, 0.0, sy, cy, 0.0, 0.0, 0.0, 1.0)
          .finished();

  const Eigen::Matrix3d drx =
      (Eigen::Matrix3d() << 0.0, 0.0, 0.0, 0.0, -sr, -cr, 0.0, cr, -sr)
          .finished();
  const Eigen::Matrix3d dry =
      (Eigen::Matrix3d() << -sp, 0.0, cp, 0.0, 0.0, 0.0, -cp, 0.0, -sp)
          .finished();
  const Eigen::Matrix3d drz =
      (Eigen::Matrix3d() << -sy, -cy, 0.0, cy, -sy, 0.0, 0.0, 0.0, 0.0)
          .finished();

  EulerLinearizationCache cache;
  cache.d_rotation_droll = rz * ry * drx;
  cache.d_rotation_dpitch = rz * dry * rx;
  cache.d_rotation_dyaw = drz * ry * rx;
  return cache;
}

// Euler-angle Jacobian under the same pose convention as PoseToMatrix():
//   R(roll, pitch, yaw) = Rz(yaw) Ry(pitch) Rx(roll).
//
// For the point-to-plane residual
//   r_i = n_i^T (R p_i + t - q_i),
// the Euler derivatives are
//   dr_i / droll  = n_i^T (dR / droll) p_i,
//   dr_i / dpitch = n_i^T (dR / dpitch) p_i,
//   dr_i / dyaw   = n_i^T (dR / dyaw) p_i.
//
// Unlike the SE(3) and SO(3) manifold updates, this parameterization operates
// directly on roll / pitch / yaw coordinates, so it remains more sensitive to
// chart singularities and angle coupling when pitch approaches +/- pi/2.
inline Eigen::Matrix<double, 1, 6> ComputeEulerPointToPlaneJacobian(
    const Eigen::Vector3d& point_body, const Eigen::Vector3d& normal_world,
    const EulerLinearizationCache& cache) {
  Eigen::Matrix<double, 1, 6> jacobian;
  // Rotation blocks: n_i^T (dR / d angle) p_i for roll / pitch / yaw.
  jacobian(0) = normal_world.dot(cache.d_rotation_droll * point_body);
  jacobian(1) = normal_world.dot(cache.d_rotation_dpitch * point_body);
  jacobian(2) = normal_world.dot(cache.d_rotation_dyaw * point_body);
  // Translation blocks: direct derivatives of n_i^T t with respect to x, y, z.
  jacobian(3) = normal_world.x();
  jacobian(4) = normal_world.y();
  jacobian(5) = normal_world.z();
  return jacobian;
}

inline bool LoadPointCloud(const std::string& path, PointCloudPtr cloud,
                           std::string* error_message) {
  if (!cloud) {
    if (error_message != nullptr) {
      *error_message = "cloud pointer is null";
    }
    return false;
  }
  if (pcl::io::loadPCDFile<PointT>(path, *cloud) != 0) {
    if (error_message != nullptr) {
      *error_message = "failed to load PCD: " + path;
    }
    return false;
  }
  if (cloud->empty()) {
    if (error_message != nullptr) {
      *error_message = "loaded PCD is empty: " + path;
    }
    return false;
  }
  return true;
}

inline bool HasOpenMp() {
#ifdef _OPENMP
  return true;
#else
  return false;
#endif
}

inline bool HasTbb() { return DCREG_USE_TBB == 1; }

inline ParallelMode ResolveParallelMode(ParallelMode mode) {
  if (mode == ParallelMode::kAuto) {
    if (HasTbb()) {
      return ParallelMode::kTbb;
    }
    if (HasOpenMp()) {
      return ParallelMode::kOpenMP;
    }
    return ParallelMode::kSerial;
  }
  if (mode == ParallelMode::kTbb && !HasTbb()) {
    return HasOpenMp() ? ParallelMode::kOpenMP : ParallelMode::kSerial;
  }
  if (mode == ParallelMode::kOpenMP && !HasOpenMp()) {
    return HasTbb() ? ParallelMode::kTbb : ParallelMode::kSerial;
  }
  return mode;
}

inline const char* ToString(Algorithm algorithm) {
  switch (algorithm) {
    case Algorithm::kNone:
      return "None";
    case Algorithm::kDCReg:
      return "DCReg";
  }
  return "Unknown";
}

inline const char* ToString(Parameterization parameterization) {
  switch (parameterization) {
    case Parameterization::kEuler:
      return "Euler";
    case Parameterization::kSE3:
      return "SE3";
    case Parameterization::kSO3:
      return "SO3";
    case Parameterization::kQuaternion:
      return "Quaternion";
  }
  return "Unknown";
}

inline const char* ToString(ParallelMode mode) {
  switch (mode) {
    case ParallelMode::kAuto:
      return "Auto";
    case ParallelMode::kSerial:
      return "Serial";
    case ParallelMode::kOpenMP:
      return "OpenMP";
    case ParallelMode::kTbb:
      return "TBB";
  }
  return "Unknown";
}

template <typename Fn>
inline void ParallelFor(int begin, int end, ParallelMode mode, const Fn& fn) {
  if (begin >= end) {
    return;
  }

#if DCREG_USE_TBB
  if (mode == ParallelMode::kTbb) {
    tbb::parallel_for(
        tbb::blocked_range<int>(begin, end),
        [&](const tbb::blocked_range<int>& range) {
          for (int index = range.begin(); index != range.end(); ++index) {
            fn(index);
          }
        });
    return;
  }
#endif

#ifdef _OPENMP
  if (mode == ParallelMode::kOpenMP) {
#pragma omp parallel for schedule(static)
    for (int index = begin; index < end; ++index) {
      fn(index);
    }
    return;
  }
#endif

  for (int index = begin; index < end; ++index) {
    fn(index);
  }
}

inline std::string RepoPath(const std::string& relative_path) {
  const std::filesystem::path root(DCREG_SOURCE_DIR);
  return (root / relative_path).lexically_normal().string();
}

inline const TestCase kShiftedCylinderCase = [] {
  TestCase test_case;
  test_case.name = "shifted_cylinder";
  test_case.folder_path = RepoPath("data/shifted_cylinder/");
  test_case.source_pcd = "measured_cloud_shifted_cylinder.pcd";
  test_case.target_pcd = "measured_cloud_shifted_cylinder.pcd";
  test_case.initial_pose = PoseFromDegrees(0.2, 0.8, 0.5, 0.1, 0.1, 2.0);
  test_case.ground_truth_pose = PoseFromDegrees(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  return test_case;
}();

inline const TestCase kShiftedCylinderLongRunCase = [] {
  TestCase test_case = kShiftedCylinderCase;
  test_case.name = "shifted_cylinder_long_run";
  test_case.params.max_iterations = 5000;
  test_case.params.convergence_trans = 1e-10;
  test_case.params.convergence_rot = 1e-12;
  return test_case;
}();

inline const TestCase kParkingLotPk01Case = [] {
  TestCase test_case;
  test_case.name = "parking_lot_pk01";
  test_case.folder_path = RepoPath("dataset/Parking-Lot-example/");
  test_case.source_pcd = "parkinglot_raw_1976_frame.pcd";
  test_case.target_pcd = "prior_map.pcd";
  test_case.initial_pose = PoseFromRadians(
      -85.258712146960, -464.541500829005, -1.092934540703, 0.019508452196,
      0.220824202961, 2.841990977838);
  test_case.has_ground_truth_pose = false;
  test_case.params.search_radius = 0.5;
  test_case.params.max_iterations = 30;
  test_case.params.convergence_trans = 1e-3;
  test_case.params.convergence_rot = 1e-5;
  test_case.params.plane_fit_neighbors = 5;
  return test_case;
}();

}  // namespace dcreg

#endif  // DCREG_UTILS_HPP_
