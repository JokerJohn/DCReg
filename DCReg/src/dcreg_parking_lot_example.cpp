#include "dcreg.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using dcreg::PointCloud;
using dcreg::PointCloudPtr;
using dcreg::RunOptions;
using dcreg::TestCase;
using IntensityPoint = pcl::PointXYZI;
using IntensityCloud = pcl::PointCloud<IntensityPoint>;
using IntensityCloudPtr = IntensityCloud::Ptr;

// Real-scene sample:
//   source: one parking-lot LiDAR frame, `parkinglot_raw_1976_frame.pcd`
//   target: the prior map used by localization, `prior_map.pcd`
// The initial pose is the `predict` pose from `parkinglot_raw_1976_info.txt`;
// this scene has no independent ground-truth pose.
constexpr float kSourceRangeCropM = 10.0f;
constexpr float kSourceLeafSizeM = 0.1f;
constexpr float kTargetVisualizationRadiusM = 100.0f;
constexpr float kTargetVisualizationLeafSizeM = 0.5f;
constexpr const char* kDefaultVisualizationDirectory =
    "data/Parking-Lot-example/visualization";
constexpr std::array<const char*, 3> kRotAxisLabels = {"roll", "pitch", "yaw"};
constexpr std::array<const char*, 3> kTransAxisLabels = {"x", "y", "z"};

struct ParsedInput {
  TestCase test_case;
  bool using_default_paths = true;
  bool export_visualization = true;
  std::string export_directory =
      dcreg::RepoPath(kDefaultVisualizationDirectory);
};

struct PreparedClouds {
  PointCloudPtr source{new PointCloud};
  PointCloudPtr target{new PointCloud};
  std::size_t raw_source_point_count = 0;
  std::size_t cropped_source_point_count = 0;
  std::size_t raw_target_point_count = 0;
};

struct InitialAnalysis {
  int correspondence_count = 0;
  double initial_rmse = 0.0;
  dcreg::DegeneracyDetection detection;
  dcreg::DegeneracyCharacterization characterization;
  dcreg::LinearSolveReport solve_report;
};

struct VoxelKey {
  std::int64_t x = 0;
  std::int64_t y = 0;
  std::int64_t z = 0;

  bool operator==(const VoxelKey& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct VoxelKeyHash {
  std::size_t operator()(const VoxelKey& key) const {
    std::size_t seed = std::hash<std::int64_t>{}(key.x);
    seed ^= std::hash<std::int64_t>{}(key.y) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2);
    seed ^= std::hash<std::int64_t>{}(key.z) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2);
    return seed;
  }
};

struct VoxelAccumulator {
  Eigen::Vector3d point_sum = Eigen::Vector3d::Zero();
  double intensity_sum = 0.0;
  int count = 0;
};

std::string JoinPath(const TestCase& test_case, const std::string& filename) {
  return test_case.folder_path + filename;
}

std::string JsonMask(const std::array<bool, 6>& mask) {
  std::string text = "[";
  for (std::size_t i = 0; i < mask.size(); ++i) {
    text += mask[i] ? "true" : "false";
    if (i + 1 != mask.size()) {
      text += ", ";
    }
  }
  text += "]";
  return text;
}

// Keeps the real-scene example self-contained without changing the shared
// presets in utils.hpp.
void ApplyExampleOverrides(TestCase* test_case) {
  test_case->params.search_radius = 0.5;
  test_case->params.convergence_trans = 1e-2;
  test_case->params.convergence_rot = 1e-3;
}

ParsedInput ParseInput(int argc, char** argv) {
  ParsedInput input;
  input.test_case = dcreg::kParkingLotPk01Case;

  std::vector<std::string> positional_args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--export_vis") {
      if (i + 1 >= argc) {
        throw std::invalid_argument("--export_vis requires a directory");
      }
      input.export_visualization = true;
      input.export_directory = argv[++i];
      continue;
    }
    positional_args.push_back(arg);
  }

  if (positional_args.empty()) {
    return input;
  }
  if (positional_args.size() != 2) {
    throw std::invalid_argument("expected zero or two positional arguments");
  }

  input.using_default_paths = false;
  input.test_case.name = "parking_lot_custom";
  input.test_case.folder_path.clear();
  input.test_case.source_pcd = positional_args[0];
  input.test_case.target_pcd = positional_args[1];
  return input;
}

void PrintUsage(const char* program, const TestCase& default_case) {
  std::cout << "Usage:\n"
            << "  " << program << " [--export_vis <dir>]\n"
            << "  " << program
            << " <source_pcd> <target_pcd> [--export_vis <dir>]\n\n"
            << "Default visualization export:\n"
            << "  " << dcreg::RepoPath(kDefaultVisualizationDirectory) << "\n\n"
            << "Default real-scene sample paths:\n"
            << "  source: " << JoinPath(default_case, default_case.source_pcd)
            << '\n'
            << "  target: " << JoinPath(default_case, default_case.target_pcd)
            << '\n';
}

void PrintDefaultPathHint() {
  std::cerr << "The default parking-lot sample expects the bundled source frame "
            << "and a downloaded prior_map.pcd under "
            << dcreg::RepoPath("data/Parking-Lot-example/") << ".\n";
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

std::string MaskString(const std::array<bool, 6>& mask) {
  std::string text;
  text.reserve(mask.size());
  for (bool value : mask) {
    text.push_back(value ? '1' : '0');
  }
  return text;
}

template <typename Derived>
void WriteJsonVector(std::ostream& stream, const Eigen::MatrixBase<Derived>& vector) {
  stream << '[';
  for (int i = 0; i < vector.size(); ++i) {
    stream << vector.derived()(i);
    if (i + 1 != vector.size()) {
      stream << ", ";
    }
  }
  stream << ']';
}

template <typename Derived>
void WriteJsonMatrix(std::ostream& stream, const Eigen::MatrixBase<Derived>& matrix) {
  stream << '[';
  for (int row = 0; row < matrix.rows(); ++row) {
    stream << '[';
    for (int col = 0; col < matrix.cols(); ++col) {
      stream << matrix(row, col);
      if (col + 1 != matrix.cols()) {
        stream << ", ";
      }
    }
    stream << ']';
    if (row + 1 != matrix.rows()) {
      stream << ", ";
    }
  }
  stream << ']';
}

// Applies simple voxel-centroid downsampling without adding more PCL
// dependencies to this example file.
void DownsamplePointCloudInPlace(PointCloudPtr cloud, float leaf_size_m) {
  std::unordered_map<VoxelKey, VoxelAccumulator, VoxelKeyHash> voxel_map;
  voxel_map.reserve(std::max<std::size_t>(1, cloud->size() / 8));

  const double leaf = static_cast<double>(leaf_size_m);
  for (const dcreg::PointT& point : cloud->points) {
    const VoxelKey key = {
        static_cast<std::int64_t>(std::floor(point.x / leaf)),
        static_cast<std::int64_t>(std::floor(point.y / leaf)),
        static_cast<std::int64_t>(std::floor(point.z / leaf)),
    };
    VoxelAccumulator& accumulator = voxel_map[key];
    accumulator.point_sum += Eigen::Vector3d(point.x, point.y, point.z);
    ++accumulator.count;
  }

  PointCloud filtered;
  filtered.reserve(voxel_map.size());
  for (const auto& [key, accumulator] : voxel_map) {
    (void)key;
    const Eigen::Vector3d centroid =
        accumulator.point_sum / static_cast<double>(accumulator.count);
    filtered.emplace_back(static_cast<float>(centroid.x()),
                          static_cast<float>(centroid.y()),
                          static_cast<float>(centroid.z()));
  }
  filtered.width = static_cast<std::uint32_t>(filtered.size());
  filtered.height = 1;
  filtered.is_dense = true;
  cloud->swap(filtered);
}

// ===== BEGIN CHANGE: intensity-preserving visualization clouds =====
bool LoadIntensityCloud(const std::string& path, IntensityCloudPtr cloud,
                        std::string* error_message) {
  if (pcl::io::loadPCDFile<IntensityPoint>(path, *cloud) < 0) {
    *error_message = "failed to load intensity PCD: " + path;
    return false;
  }
  return true;
}

void CropIntensitySourceByRangeInPlace(IntensityCloudPtr source,
                                       float max_range_m) {
  IntensityCloud cropped;
  cropped.reserve(source->size());

  const double max_range_sq =
      static_cast<double>(max_range_m) * static_cast<double>(max_range_m);
  for (const IntensityPoint& point : source->points) {
    const double range_sq = static_cast<double>(point.x) * point.x +
                            static_cast<double>(point.y) * point.y +
                            static_cast<double>(point.z) * point.z;
    if (range_sq <= max_range_sq) {
      cropped.push_back(point);
    }
  }

  cropped.width = static_cast<std::uint32_t>(cropped.size());
  cropped.height = 1;
  cropped.is_dense = source->is_dense;
  source->swap(cropped);
}

void DownsampleIntensityCloudInPlace(IntensityCloudPtr cloud, float leaf_size_m,
                                     bool use_ground_plane_grid = false) {
  std::unordered_map<VoxelKey, VoxelAccumulator, VoxelKeyHash> voxel_map;
  voxel_map.reserve(std::max<std::size_t>(1, cloud->size() / 8));

  const double leaf = static_cast<double>(leaf_size_m);
  for (const IntensityPoint& point : cloud->points) {
    const VoxelKey key = {
        static_cast<std::int64_t>(std::floor(point.x / leaf)),
        static_cast<std::int64_t>(std::floor(point.y / leaf)),
        use_ground_plane_grid
            ? 0
            : static_cast<std::int64_t>(std::floor(point.z / leaf)),
    };
    VoxelAccumulator& accumulator = voxel_map[key];
    accumulator.point_sum += Eigen::Vector3d(point.x, point.y, point.z);
    accumulator.intensity_sum += point.intensity;
    ++accumulator.count;
  }

  IntensityCloud filtered;
  filtered.reserve(voxel_map.size());
  for (const auto& [key, accumulator] : voxel_map) {
    (void)key;
    const Eigen::Vector3d centroid =
        accumulator.point_sum / static_cast<double>(accumulator.count);
    IntensityPoint point;
    point.x = static_cast<float>(centroid.x());
    point.y = static_cast<float>(centroid.y());
    point.z = static_cast<float>(centroid.z());
    point.intensity =
        static_cast<float>(accumulator.intensity_sum / accumulator.count);
    filtered.push_back(point);
  }
  filtered.width = static_cast<std::uint32_t>(filtered.size());
  filtered.height = 1;
  filtered.is_dense = true;
  cloud->swap(filtered);
}

void CropIntensityTargetAroundCenterInPlace(IntensityCloudPtr target,
                                            const Eigen::Vector3d& center_world,
                                            float radius_m) {
  IntensityCloud cropped;
  cropped.reserve(target->size());

  const double radius_sq =
      static_cast<double>(radius_m) * static_cast<double>(radius_m);
  for (const IntensityPoint& point : target->points) {
    const double dx = static_cast<double>(point.x) - center_world.x();
    const double dy = static_cast<double>(point.y) - center_world.y();
    if (dx * dx + dy * dy <= radius_sq) {
      cropped.push_back(point);
    }
  }

  cropped.width = static_cast<std::uint32_t>(cropped.size());
  cropped.height = 1;
  cropped.is_dense = target->is_dense;
  target->swap(cropped);
}

IntensityCloud TransformIntensityCloud(const IntensityCloud& source,
                                       const Eigen::Matrix4d& transform) {
  IntensityCloud transformed;
  transformed.reserve(source.size());
  for (const IntensityPoint& point : source.points) {
    const Eigen::Vector3d input(point.x, point.y, point.z);
    const Eigen::Vector3d xyz =
        transform.block<3, 3>(0, 0) * input + transform.block<3, 1>(0, 3);
    IntensityPoint transformed_point;
    transformed_point.x = static_cast<float>(xyz.x());
    transformed_point.y = static_cast<float>(xyz.y());
    transformed_point.z = static_cast<float>(xyz.z());
    transformed_point.intensity = point.intensity;
    transformed.push_back(transformed_point);
  }
  transformed.width = static_cast<std::uint32_t>(transformed.size());
  transformed.height = 1;
  transformed.is_dense = source.is_dense;
  return transformed;
}

bool PrepareVisualizationClouds(const TestCase& test_case,
                                const Eigen::Matrix4d& center_transform,
                                IntensityCloudPtr source,
                                IntensityCloudPtr target,
                                std::string* error_message) {
  if (!LoadIntensityCloud(JoinPath(test_case, test_case.source_pcd), source,
                          error_message)) {
    return false;
  }
  if (!LoadIntensityCloud(JoinPath(test_case, test_case.target_pcd), target,
                          error_message)) {
    return false;
  }

  CropIntensitySourceByRangeInPlace(source, kSourceRangeCropM);
  DownsampleIntensityCloudInPlace(source, kSourceLeafSizeM);
  CropIntensityTargetAroundCenterInPlace(
      target, center_transform.block<3, 1>(0, 3), kTargetVisualizationRadiusM);
  DownsampleIntensityCloudInPlace(target, kTargetVisualizationLeafSizeM,
                                  true);
  if (source->empty() || target->empty()) {
    *error_message = "visualization cloud is empty after preprocessing";
    return false;
  }
  return true;
}
// ===== END CHANGE: intensity-preserving visualization clouds =====

void CropSourceByRangeInPlace(PointCloudPtr source, float max_range_m) {
  PointCloud cropped;
  cropped.reserve(source->size());

  const double max_range_sq =
      static_cast<double>(max_range_m) * static_cast<double>(max_range_m);
  for (const dcreg::PointT& point : source->points) {
    const double range_sq = static_cast<double>(point.x) * point.x +
                            static_cast<double>(point.y) * point.y +
                            static_cast<double>(point.z) * point.z;
    if (range_sq <= max_range_sq) {
      cropped.push_back(point);
    }
  }

  cropped.width = static_cast<std::uint32_t>(cropped.size());
  cropped.height = 1;
  cropped.is_dense = source->is_dense;
  source->swap(cropped);
}

bool PrepareClouds(const TestCase& test_case, PreparedClouds* clouds,
                   std::string* error_message) {
  if (!dcreg::LoadPointCloud(JoinPath(test_case, test_case.source_pcd),
                             clouds->source, error_message)) {
    return false;
  }
  if (!dcreg::LoadPointCloud(JoinPath(test_case, test_case.target_pcd),
                             clouds->target, error_message)) {
    return false;
  }

  clouds->raw_source_point_count = clouds->source->size();
  clouds->raw_target_point_count = clouds->target->size();
  CropSourceByRangeInPlace(clouds->source, kSourceRangeCropM);
  clouds->cropped_source_point_count = clouds->source->size();
  DownsamplePointCloudInPlace(clouds->source, kSourceLeafSizeM);

  if (clouds->source->empty()) {
    *error_message = "source cloud is empty after range crop and voxel filtering";
    return false;
  }
  if (clouds->target->empty()) {
    *error_message = "target prior map is empty";
    return false;
  }
  return true;
}

// Runs the paper-style three-module analysis on the initial pose before the
// full iterative registration starts.
InitialAnalysis AnalyzeInitialPose(const TestCase& test_case,
                                   const RunOptions& options,
                                   const PreparedClouds& clouds) {
  const dcreg::ParallelMode resolved_mode =
      dcreg::ResolveParallelMode(options.parallel_mode);
  pcl::KdTreeFLANN<dcreg::PointT> target_kdtree;
  target_kdtree.setInputCloud(clouds.target);

  const dcreg::So3State state = dcreg::ToSo3State(test_case.initial_pose);
  int near_neighbor_count = 0;
  double residual_square_sum = 0.0;
  const std::vector<dcreg::Correspondence> correspondences =
      dcreg::CollectCorrespondences(*clouds.source, *clouds.target, &target_kdtree,
                                    state.Matrix(), test_case.params,
                                    resolved_mode, &near_neighbor_count,
                                    &residual_square_sum);
  if (static_cast<int>(correspondences.size()) < dcreg::kMinCorrespondences) {
    throw std::runtime_error("too few correspondences at the initial pose");
  }

  InitialAnalysis analysis;
  analysis.correspondence_count = static_cast<int>(correspondences.size());
  analysis.initial_rmse =
      std::sqrt(residual_square_sum / correspondences.size());

  const dcreg::LinearSystem system =
      dcreg::BuildSo3LinearSystem(correspondences, state, resolved_mode);
  analysis.detection = dcreg::DetectDegeneracy(system.hessian, test_case.params);
  analysis.characterization =
      dcreg::CharacterizeDegeneracy(analysis.detection, test_case.params);
  analysis.solve_report = dcreg::SolvePreconditionedUpdate(
      system, test_case.params, analysis.characterization);
  return analysis;
}

void PrintExampleHeader(const TestCase& test_case, const PreparedClouds& clouds,
                        const InitialAnalysis& analysis) {
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Parking-lot DCReg example\n";
  std::cout << "source_pcd: " << JoinPath(test_case, test_case.source_pcd)
            << '\n';
  std::cout << "target_pcd: " << JoinPath(test_case, test_case.target_pcd)
            << '\n';
  std::cout << "source_note: single parking-lot LiDAR frame in the sensor frame\n";
  std::cout << "target_note: prior localization map in the target/world frame\n";
  std::cout << "source_points_raw: " << clouds.raw_source_point_count
            << " | source_points_cropped: "
            << clouds.cropped_source_point_count
            << " | source_points_downsampled: " << clouds.source->size()
            << " | target_points_raw: " << clouds.raw_target_point_count
            << " | target_points_used: " << clouds.target->size() << '\n';
  std::cout << "source_range_crop_m: " << kSourceRangeCropM
            << " | source_voxel_leaf_size_m: " << kSourceLeafSizeM
            << " | target_preprocessing: unchanged\n";
  std::cout << "search_radius: " << test_case.params.search_radius
            << " | convergence_trans: " << test_case.params.convergence_trans
            << " | convergence_rot: " << test_case.params.convergence_rot
            << '\n';
  std::cout << "initial_correspondences: " << analysis.correspondence_count
            << " | initial_rmse: " << analysis.initial_rmse << '\n';
  if (!test_case.has_ground_truth_pose) {
    std::cout
        << "initial_pose_error: unavailable (no ground-truth pose for this case)\n";
  }
  std::cout << '\n';
}

void PrintInitialModules(const InitialAnalysis& analysis) {
  std::cout << "[Module 1] Spectral degeneracy detection\n";
  std::cout << "cond_full: " << analysis.detection.cond_full << '\n';
  std::cout << "cond_schur_rot: " << analysis.characterization.cond_schur_rot
            << '\n';
  std::cout << "cond_schur_trans: "
            << analysis.characterization.cond_schur_trans << "\n\n";

  std::cout << "[Module 2] Physical-axis degeneracy characterization\n";
  std::cout << "degenerate_mask: "
            << MaskString(analysis.characterization.degenerate_mask) << '\n';
  std::cout << "raw_lambda_rot: "
            << analysis.detection.lambda_schur_rot.transpose() << '\n';
  std::cout << "raw_lambda_trans: "
            << analysis.detection.lambda_schur_trans.transpose() << '\n';
  std::cout << "aligned_lambda_rpy: "
            << analysis.characterization.aligned_lambda_schur_rot.transpose()
            << '\n';
  std::cout << "aligned_lambda_xyz: "
            << analysis.characterization.aligned_lambda_schur_trans.transpose()
            << '\n';
  PrintContributionModes(
      "rot_axis_contribution_ratio(each r_i as physical-axis mixture):", "r",
      kRotAxisLabels,
      analysis.characterization.rot_axis_contribution_ratio);
  PrintContributionModes(
      "trans_axis_contribution_ratio(each t_i as physical-axis mixture):", "t",
      kTransAxisLabels,
      analysis.characterization.trans_axis_contribution_ratio);
  std::cout << "clamped_lambda_rpy: "
            << analysis.characterization.clamped_lambda_schur_rot.transpose()
            << '\n';
  std::cout << "clamped_lambda_xyz: "
            << analysis.characterization.clamped_lambda_schur_trans.transpose()
            << "\n\n";

  std::cout << "[Module 3] Preconditioned linear solve\n";
  std::cout << "preconditioned_delta: "
            << analysis.solve_report.delta.transpose() << '\n';
  std::cout << "pcg_iterations: " << analysis.solve_report.iterations << '\n';
  std::cout << "pcg_relative_residual: "
            << analysis.solve_report.relative_residual << '\n';
  std::cout << "qr_fallback: "
            << (analysis.solve_report.used_qr_fallback ? 1 : 0) << "\n\n";
}

void RunExample(const TestCase& test_case, const RunOptions& options,
                const PreparedClouds& clouds, const std::string& export_directory) {
  const InitialAnalysis analysis = AnalyzeInitialPose(test_case, options, clouds);
  PrintExampleHeader(test_case, clouds, analysis);
  PrintInitialModules(analysis);

  std::cout << "[Registration] Full real-scene matching\n";
  const dcreg::RegistrationResult result =
      dcreg::RunRegistration(*clouds.source, *clouds.target, test_case, options);
  dcreg::PrintSummary(test_case, options, result);

  if (!export_directory.empty()) {
    const std::filesystem::path output_dir(export_directory);
    std::filesystem::create_directories(output_dir);

    const Eigen::Matrix4d initial_transform =
        dcreg::PoseToMatrix(test_case.initial_pose);
    // ===== BEGIN CHANGE: export local intensity target map =====
    IntensityCloudPtr visualization_source(new IntensityCloud);
    IntensityCloudPtr visualization_target(new IntensityCloud);
    std::string visualization_error;
    if (!PrepareVisualizationClouds(test_case, result.transform,
                                    visualization_source,
                                    visualization_target,
                                    &visualization_error)) {
      throw std::runtime_error(visualization_error);
    }

    const IntensityCloud source_initial_intensity =
        TransformIntensityCloud(*visualization_source, initial_transform);
    const IntensityCloud source_registered_intensity =
        TransformIntensityCloud(*visualization_source, result.transform);
    // ===== END CHANGE: export local intensity target map =====

    pcl::io::savePCDFileASCII((output_dir / "source_initial.pcd").string(),
                              source_initial_intensity);
    pcl::io::savePCDFileASCII((output_dir / "source_registered.pcd").string(),
                              source_registered_intensity);
    pcl::io::savePCDFileASCII((output_dir / "target_map.pcd").string(),
                              *visualization_target);
    std::filesystem::remove(output_dir / "target_local.pcd");

    std::ofstream json(output_dir / "diagnostics.json");
    json << std::fixed << std::setprecision(12);
    json << "{\n";
    json << "  \"notes\": \"Parking-lot real-scene DCReg visualization export. "
            "The source cloud is a single LiDAR frame after local preprocessing; "
            "the visualization target cloud is a 100 m local prior-map crop "
            "downsampled with a 0.5 m ground-plane grid while preserving "
            "intensity.\",\n";
    json << "  \"source_pcd\": \"" << JoinPath(test_case, test_case.source_pcd)
         << "\",\n";
    json << "  \"target_pcd\": \"" << JoinPath(test_case, test_case.target_pcd)
         << "\",\n";
    json << "  \"source_range_crop_m\": " << kSourceRangeCropM << ",\n";
    json << "  \"source_voxel_leaf_size_m\": " << kSourceLeafSizeM << ",\n";
    json << "  \"target_visualization_radius_m\": "
         << kTargetVisualizationRadiusM << ",\n";
    json << "  \"target_visualization_voxel_leaf_size_m\": "
         << kTargetVisualizationLeafSizeM << ",\n";
    json << "  \"point_counts\": {\n";
    json << "    \"source_raw\": " << clouds.raw_source_point_count << ",\n";
    json << "    \"source_cropped\": " << clouds.cropped_source_point_count
         << ",\n";
    json << "    \"source_used\": " << clouds.source->size() << ",\n";
    json << "    \"target_raw\": " << clouds.raw_target_point_count << ",\n";
    json << "    \"target_visualized\": " << visualization_target->size()
         << "\n";
    json << "  },\n";
    json << "  \"initial_correspondences\": " << analysis.correspondence_count
         << ",\n";
    json << "  \"initial_rmse\": " << analysis.initial_rmse << ",\n";
    json << "  \"registration\": {\n";
    json << "    \"success\": " << (result.success ? "true" : "false") << ",\n";
    json << "    \"iterations\": " << result.iterations << ",\n";
    json << "    \"rmse\": " << result.rmse << ",\n";
    json << "    \"fitness\": " << result.fitness << "\n";
    json << "  },\n";
    json << "  \"initial_transform\": ";
    WriteJsonMatrix(json, initial_transform);
    json << ",\n";
    json << "  \"final_transform\": ";
    WriteJsonMatrix(json, result.transform);
    json << ",\n";
    json << "  \"degeneracy\": {\n";
    json << "    \"degenerate_mask\": "
         << JsonMask(analysis.characterization.degenerate_mask) << ",\n";
    json << "    \"cond_full\": " << analysis.detection.cond_full << ",\n";
    json << "    \"cond_schur_rot\": "
         << analysis.characterization.cond_schur_rot << ",\n";
    json << "    \"cond_schur_trans\": "
         << analysis.characterization.cond_schur_trans << ",\n";
    json << "    \"raw_lambda_rot\": ";
    WriteJsonVector(json, analysis.detection.lambda_schur_rot);
    json << ",\n";
    json << "    \"raw_lambda_trans\": ";
    WriteJsonVector(json, analysis.detection.lambda_schur_trans);
    json << ",\n";
    json << "    \"aligned_lambda_rpy\": ";
    WriteJsonVector(json, analysis.characterization.aligned_lambda_schur_rot);
    json << ",\n";
    json << "    \"aligned_lambda_xyz\": ";
    WriteJsonVector(json, analysis.characterization.aligned_lambda_schur_trans);
    json << ",\n";
    json << "    \"rot_axis_contribution_ratio\": ";
    WriteJsonMatrix(json, analysis.characterization.rot_axis_contribution_ratio);
    json << ",\n";
    json << "    \"trans_axis_contribution_ratio\": ";
    WriteJsonMatrix(json,
                    analysis.characterization.trans_axis_contribution_ratio);
    json << ",\n";
    json << "    \"preconditioned_delta\": ";
    WriteJsonVector(json, analysis.solve_report.delta);
    json << ",\n";
    json << "    \"pcg_iterations\": " << analysis.solve_report.iterations
         << ",\n";
    json << "    \"qr_fallback\": "
         << (analysis.solve_report.used_qr_fallback ? "true" : "false")
         << "\n";
    json << "  }\n";
    json << "}\n";

    std::cout << "Visualization export: " << output_dir << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  ParsedInput input;
  try {
    input = ParseInput(argc, argv);
  } catch (const std::invalid_argument&) {
    PrintUsage(argv[0], dcreg::kParkingLotPk01Case);
    return 1;
  }
  ApplyExampleOverrides(&input.test_case);

  const RunOptions options = [] {
    RunOptions run_options;
    run_options.algorithm = dcreg::Algorithm::kDCReg;
    run_options.parameterization = dcreg::Parameterization::kSO3;
    run_options.parallel_mode = dcreg::ParallelMode::kAuto;
    return run_options;
  }();

  PreparedClouds clouds;
  std::string error_message;
  if (!PrepareClouds(input.test_case, &clouds, &error_message)) {
    std::cerr << error_message << '\n';
    if (input.using_default_paths) {
      PrintDefaultPathHint();
      PrintUsage(argv[0], dcreg::kParkingLotPk01Case);
    }
    return 1;
  }

  try {
    RunExample(input.test_case, options, clouds,
               input.export_visualization ? input.export_directory : "");
  } catch (const std::exception& error) {
    std::cerr << "parking-lot example failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
