import logging
import os
import os.path as osp
import time

# ===== BEGIN CHANGE: suppress numexpr startup info logs =====
logging.getLogger('numexpr').setLevel(logging.WARNING)
logging.getLogger('numexpr.utils').setLevel(logging.WARNING)
os.environ.setdefault('NUMEXPR_MAX_THREADS', str(min(os.cpu_count() or 8, 8)))
# ===== END CHANGE: suppress numexpr startup info logs =====

import numpy as np
import torch
import torch.utils.data
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from config import make_cfg
from model import create_model
from geotransformer.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
)
from geotransformer.utils.torch import to_cuda
from geotransformer.modules.registration.metrics import isotropic_transform_error
from geotransformer.utils.pointcloud import apply_transform, get_transform_from_rotation_translation, inverse_transform



# ----------------------------------------------------------------------------- #
# Hard-coded configuration                                                      #
# ----------------------------------------------------------------------------- #
WORKING_DIR = osp.dirname(osp.realpath(__file__))
ROOT_DIR = osp.dirname(osp.dirname(WORKING_DIR))
ALGORITHM_NAME = 'GeoTransformer'

TARGET_POINT_CLOUD = '/home/xchu/Downloads/ltloc-results/simulated_pc/shifted_cylinder/measured_cloud_shifted_cylinder.pcd'  # 更新为真实 target PCD
SOURCE_POINT_CLOUD = None  # 若已有 source PCD，可在此填写路径；留空则由目标生成
INITIAL_POSE = (0.2, 0.8, 0.5, 0.1, 0.1, 2.0)  # (tx, ty, tz, roll, pitch, yaw)
POSE_IN_RADIANS = False  # 若 INITIAL_POSE 中角度已是弧度，则改为 True
FORCE_GT_IDENTITY = True  # True: 评价时将 GT 视为单位阵

USE_GPU = True  # True：若可用则使用 GPU；False：始终使用 CPU
SNAPSHOT_PATH = osp.join(ROOT_DIR, 'weights', 'geotransformer-kitti.pth.tar')
SAMPLE_THRESHOLD = 1
KEEP_RATIO = 0.8
RRE_THRESHOLD_DEG = 5.0  # degrees
RTE_THRESHOLD_M = 0.2  # meters
RESULTS_TXT = osp.join(WORKING_DIR, 'custom_infer_results.txt')

# ===== BEGIN CHANGE: align post-registration metrics with cpp point-to-plane =====
# 这些参数与 C++ ICP 评估逻辑保持同口径；其中搜索半径需与对应 YAML 保持一致。
POINT_TO_POINT_FITNESS_THRESHOLD = 0.2  # meters
ICP_SEARCH_RADIUS = 1.0  # meters
ICP_NORMAL_K = 5
ICP_MAX_PLANE_THICKNESS = 0.2  # meters
ICP_WEIGHT_SLOPE = 0.9
ICP_MIN_EFFECTIVE_WEIGHT = 0.1
ICP_MIN_NORMAL_NORM = 1e-6
ICP_MIN_EFFECTIVE_CORRESPONDENCES = 10
# ===== END CHANGE: align post-registration metrics with cpp point-to-plane =====
# ----------------------------------------------------------------------------- #


class SinglePairDataset(torch.utils.data.Dataset):
    def __init__(self, ref_points, src_points, transform):
        self.sample = {
            'seq_id': np.int64(0),
            'ref_frame': np.int64(0),
            'src_frame': np.int64(1),
            'ref_points': ref_points.astype(np.float32),
            'src_points': src_points.astype(np.float32),
            'ref_feats': np.ones((ref_points.shape[0], 1), dtype=np.float32),
            'src_feats': np.ones((src_points.shape[0], 1), dtype=np.float32),
            'transform': transform.astype(np.float32),
        }

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.sample


def load_point_cloud(path):
    ext = osp.splitext(path)[1].lower()
    if ext == '.npy':
        points = np.load(path)
    elif ext in ['.pcd', '.ply', '.xyz']:
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(path)
        if len(pcd.points) == 0:
            raise ValueError(f'Point cloud {path} is empty.')
        points = np.asarray(pcd.points)
    elif ext == '.bin':
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
    else:
        raise ValueError(f'Unsupported point cloud format: {ext}')
    return points.astype(np.float32)


def pose_to_transform(pose, radians=False):
    if pose is None:
        raise ValueError('INITIAL_POSE must be specified when SOURCE_POINT_CLOUD is None.')
    translation = np.asarray(pose[:3], dtype=np.float64)
    angles = np.asarray(pose[3:], dtype=np.float64)
    rotation = Rotation.from_euler('xyz', angles, degrees=not radians).as_matrix()
    transform = get_transform_from_rotation_translation(rotation, translation)
    return transform.astype(np.float32)


# ===== BEGIN CHANGE: align post-registration metrics with cpp point-to-plane =====
def compute_chamfer_distance(aligned_points, target_points):
    if aligned_points.shape[0] == 0 or target_points.shape[0] == 0:
        return np.nan

    target_tree = cKDTree(target_points)
    dists_src_to_tgt, _ = target_tree.query(aligned_points, k=1, workers=-1)

    source_tree = cKDTree(aligned_points)
    dists_tgt_to_src, _ = source_tree.query(target_points, k=1, workers=-1)

    return float(dists_src_to_tgt.mean() + dists_tgt_to_src.mean())


# ===== BEGIN CHANGE: add point-to-point rmse logging =====
def compute_point_to_point_metrics(aligned_points, target_points, inlier_threshold):
    if aligned_points.shape[0] == 0 or target_points.shape[0] == 0:
        return float('inf'), 0.0, 0

    target_tree = cKDTree(target_points)
    dists_src_to_tgt, _ = target_tree.query(aligned_points, k=1, workers=-1)
    inlier_mask = dists_src_to_tgt <= inlier_threshold
    inlier_count = int(inlier_mask.sum())
    if inlier_count > 0:
        rmse = float(np.sqrt(np.mean(np.square(dists_src_to_tgt[inlier_mask]))))
    else:
        rmse = float('inf')
    fitness = inlier_count / aligned_points.shape[0]
    return rmse, fitness, inlier_count
# ===== END CHANGE: add point-to-point rmse logging =====


def compute_cpp_style_point_to_plane_metrics(
    aligned_points,
    target_points,
    search_radius,
    normal_k,
    max_plane_thickness,
    weight_slope,
    min_effective_weight,
    min_normal_norm,
    min_effective_correspondences,
):
    if aligned_points.shape[0] == 0 or target_points.shape[0] < normal_k:
        return float('inf'), 0.0, 0, 0

    target_tree = cKDTree(target_points)
    neighbor_dists, neighbor_indices = target_tree.query(
        aligned_points,
        k=normal_k,
        distance_upper_bound=search_radius,
        workers=-1,
    )

    if normal_k == 1:
        neighbor_dists = neighbor_dists[:, None]
        neighbor_indices = neighbor_indices[:, None]

    plane_thickness_sq = max_plane_thickness * max_plane_thickness
    total_distance_sq = 0.0
    candidate_corr_count = 0
    effective_corr_count = 0

    for point, dists, indices in zip(aligned_points, neighbor_dists, neighbor_indices):
        if not np.all(np.isfinite(dists)) or dists[-1] >= search_radius:
            continue

        candidate_corr_count += 1
        neighbors = target_points[indices]
        plane_coeffs, _, _, _ = np.linalg.lstsq(
            neighbors.astype(np.float64),
            -np.ones(normal_k, dtype=np.float64),
            rcond=None,
        )
        normal_norm = np.linalg.norm(plane_coeffs)
        if normal_norm < min_normal_norm:
            continue

        normal = plane_coeffs / normal_norm
        plane_offset = 1.0 / normal_norm

        plane_residuals = neighbors @ normal + plane_offset
        if np.max(np.square(plane_residuals)) >= plane_thickness_sq:
            continue

        point_to_plane_dist = float(point @ normal + plane_offset)
        weight = max(0.0, 1.0 - weight_slope * abs(point_to_plane_dist))
        if weight <= min_effective_weight:
            continue

        effective_corr_count += 1
        total_distance_sq += point_to_plane_dist * point_to_plane_dist

    fitness = effective_corr_count / aligned_points.shape[0]
    if effective_corr_count < min_effective_correspondences:
        rmse = float('inf')
    else:
        rmse = float(np.sqrt(total_distance_sq / effective_corr_count))

    return rmse, fitness, candidate_corr_count, effective_corr_count
# ===== END CHANGE: align post-registration metrics with cpp point-to-plane =====


def write_results_txt(path, est_transform, metrics):
    transform_str = np.array2string(
        est_transform,
        formatter={'float_kind': lambda x: f'{x: .6f}'},
        max_line_width=200,
    )
    with open(path, 'w') as f:
        f.write('Custom inference results\n')
        f.write(f'Algorithm: {ALGORITHM_NAME}\n')
        f.write(f'Target point cloud: {TARGET_POINT_CLOUD}\n')
        f.write(f"Source point cloud: {SOURCE_POINT_CLOUD or 'generated from target'}\n")
        f.write(f'GT identity enforced: {FORCE_GT_IDENTITY}\n\n')
        f.write('CPP-style point-to-plane parameters:\n')
        f.write(f'  search_radius: {ICP_SEARCH_RADIUS:.6f}\n')
        f.write(f'  normal_k: {ICP_NORMAL_K}\n')
        f.write(f'  max_plane_thickness: {ICP_MAX_PLANE_THICKNESS:.6f}\n')
        f.write(f'  weight_slope: {ICP_WEIGHT_SLOPE:.6f}\n')
        f.write(f'  min_effective_weight: {ICP_MIN_EFFECTIVE_WEIGHT:.6f}\n')
        f.write(f'  min_effective_correspondences: {ICP_MIN_EFFECTIVE_CORRESPONDENCES}\n\n')
        f.write('Estimated transform (4x4, row-major):\n')
        f.write(transform_str + '\n\n')
        f.write('Metrics:\n')
        f.write(f'  RRE (deg): {metrics["rre_deg"]:.6f}\n')
        f.write(f'  RTE (m): {metrics["rte_m"]:.6f}\n')
        f.write(f'  Registration Recall (<= {RRE_THRESHOLD_DEG} deg & <= {RTE_THRESHOLD_M} m): {metrics["rr"]:.6f}\n')
        f.write(f'  Chamfer Distance (m): {metrics["chamfer"]:.6f}\n')
        f.write(f'  ICP RMSE (point-to-plane, m): {metrics["rmse"]:.6f}\n')
        f.write(f'  ICP Fitness (effective/source): {metrics["fitness"]:.6f}\n')
        f.write(f'  Point-to-Point RMSE (<= {POINT_TO_POINT_FITNESS_THRESHOLD} m, m): {metrics["p2p_rmse"]:.6f}\n')
        f.write(f'  Point-to-Point Fitness (<= {POINT_TO_POINT_FITNESS_THRESHOLD} m): {metrics["p2p_fitness"]:.6f}\n')
        f.write(f'  ICP Candidate Correspondence Count: {metrics["candidate_corr_count"]}\n')
        f.write(f'  ICP Effective Correspondence Count: {metrics["effective_corr_count"]}\n')
        f.write(f'  Point-to-Point Inlier Count: {metrics["p2p_inlier_count"]}\n')
        f.write(f'  Runtime (ms): {metrics["runtime_ms"]:.3f}\n')
        f.write('\n')


def prepare_batch(ref_points, src_points, transform, cfg, neighbor_limits):
    dataset = SinglePairDataset(ref_points, src_points, transform)
    collated = registration_collate_fn_stack_mode(
        [dataset[0]],
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        precompute_data=True,
    )
    return collated


def main():
    # ===== BEGIN CHANGE: announce geotransformer algorithm =====
    print(f'[{ALGORITHM_NAME}] Starting custom single-pair registration inference.')
    # ===== END CHANGE: announce geotransformer algorithm =====

    if not osp.isfile(TARGET_POINT_CLOUD):
        raise FileNotFoundError(f'Target point cloud not found: {TARGET_POINT_CLOUD}. 请在 custom_infer.py 顶部更新路径。')
    if SOURCE_POINT_CLOUD is None and INITIAL_POSE is None:
        raise ValueError('未提供 SOURCE_POINT_CLOUD 时，必须设置 INITIAL_POSE。')
    if not osp.isfile(SNAPSHOT_PATH):
        raise FileNotFoundError(f'Snapshot not found: {SNAPSHOT_PATH}. 请检查预训练权重路径。')

    cfg = make_cfg()

    # Load model and weights.
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        if USE_GPU and not torch.cuda.is_available():
            print('Warning: CUDA requested but not available. Falling back to CPU.')
        device = torch.device('cpu')

    model = create_model(cfg).to(device)
    state_dict = torch.load(SNAPSHOT_PATH, map_location='cpu')
    if 'model' not in state_dict:
        raise RuntimeError('Snapshot does not contain a "model" key.')
    model.load_state_dict(state_dict['model'], strict=True)
    model.eval()

    # Load target point cloud.
    ref_points = load_point_cloud(TARGET_POINT_CLOUD)

    applied_pose = None
    if SOURCE_POINT_CLOUD is not None:
        if not osp.isfile(SOURCE_POINT_CLOUD):
            raise FileNotFoundError(f'Source point cloud not found: {SOURCE_POINT_CLOUD}. 请在 custom_infer.py 顶部更新路径。')
        src_points = load_point_cloud(SOURCE_POINT_CLOUD)
    else:
        applied_pose = pose_to_transform(INITIAL_POSE, radians=POSE_IN_RADIANS)
        src_points = apply_transform(ref_points.copy(), applied_pose)

    if FORCE_GT_IDENTITY:
        transform_gt = np.eye(4, dtype=np.float32)
    elif SOURCE_POINT_CLOUD is None:
        transform_gt = inverse_transform(applied_pose)
    else:
        transform_gt = pose_to_transform(INITIAL_POSE, radians=POSE_IN_RADIANS) if INITIAL_POSE is not None else np.eye(4, dtype=np.float32)

    dataset = SinglePairDataset(ref_points, src_points, transform_gt)
    neighbor_limits = calibrate_neighbors_stack_mode(
        dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        keep_ratio=KEEP_RATIO,
        sample_threshold=SAMPLE_THRESHOLD,
    )

    batch = prepare_batch(ref_points, src_points, transform_gt, cfg, neighbor_limits)
    if device.type == 'cuda':
        batch = to_cuda(batch)

    start_time = time.perf_counter()
    with torch.no_grad():
        output = model(batch)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    runtime_ms = (time.perf_counter() - start_time) * 1000.0

    est_transform = output['estimated_transform']
    if est_transform.dim() == 2:
        est_transform = est_transform.unsqueeze(0)

    gt_transform = batch['transform']
    if gt_transform.dim() == 2:
        gt_transform = gt_transform.unsqueeze(0)

    rre, rte = isotropic_transform_error(gt_transform, est_transform)
    rre_deg = float(rre.item())
    rte_m = float(rte.item())
    rr = 1.0 if (rre_deg <= RRE_THRESHOLD_DEG and rte_m <= RTE_THRESHOLD_M) else 0.0
    est_transform_np = est_transform.squeeze(0).detach().cpu().numpy()
    aligned_src_points = apply_transform(src_points.copy(), est_transform_np)

    # ===== BEGIN CHANGE: align post-registration metrics with cpp point-to-plane =====
    chamfer = compute_chamfer_distance(aligned_src_points, ref_points)
    p2p_rmse, p2p_fitness, p2p_inlier_count = compute_point_to_point_metrics(
        aligned_src_points,
        ref_points,
        POINT_TO_POINT_FITNESS_THRESHOLD,
    )
    rmse, fitness, candidate_corr_count, effective_corr_count = compute_cpp_style_point_to_plane_metrics(
        aligned_src_points,
        ref_points,
        ICP_SEARCH_RADIUS,
        ICP_NORMAL_K,
        ICP_MAX_PLANE_THICKNESS,
        ICP_WEIGHT_SLOPE,
        ICP_MIN_EFFECTIVE_WEIGHT,
        ICP_MIN_NORMAL_NORM,
        ICP_MIN_EFFECTIVE_CORRESPONDENCES,
    )
    # ===== END CHANGE: align post-registration metrics with cpp point-to-plane =====

    print('Estimated transform:\n', est_transform_np)
    print('--- Metrics ---')
    print(f'Rotation Error (deg): {rre_deg:.6f}')
    print(f'Translation Error (m): {rte_m:.6f}')
    print(f'Registration Recall (<= {RRE_THRESHOLD_DEG} deg & <= {RTE_THRESHOLD_M} m): {rr:.1f}')
    print(f'Chamfer Distance (m): {chamfer:.6f}')
    print(f'ICP RMSE (point-to-plane, m): {rmse:.6f}')
    print(f'ICP Fitness (effective/source): {fitness:.6f}')
    print(f'Point-to-Point RMSE (<= {POINT_TO_POINT_FITNESS_THRESHOLD} m, m): {p2p_rmse:.6f}')
    print(f'Point-to-Point Fitness (<= {POINT_TO_POINT_FITNESS_THRESHOLD} m): {p2p_fitness:.6f}')
    print(f'ICP Candidate Correspondence Count: {candidate_corr_count}')
    print(f'ICP Effective Correspondence Count: {effective_corr_count}')
    print(f'Point-to-Point Inlier Count: {p2p_inlier_count}')
    print(f'Runtime (ms): {runtime_ms:.3f}')

    metrics = {
        'rre_deg': rre_deg,
        'rte_m': rte_m,
        'rr': rr,
        'chamfer': chamfer,
        'rmse': rmse,
        'fitness': fitness,
        'p2p_rmse': p2p_rmse,
        'p2p_fitness': p2p_fitness,
        'candidate_corr_count': candidate_corr_count,
        'effective_corr_count': effective_corr_count,
        'p2p_inlier_count': p2p_inlier_count,
        'runtime_ms': runtime_ms,
    }
    write_results_txt(RESULTS_TXT, est_transform_np, metrics)
    # ===== BEGIN CHANGE: english results log message =====
    print(f'Evaluation results written to {RESULTS_TXT}')
    # ===== END CHANGE: english results log message =====


if __name__ == '__main__':
    main()
