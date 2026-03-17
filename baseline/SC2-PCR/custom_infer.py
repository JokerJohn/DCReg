import json
import os.path as osp
import time
from datetime import datetime

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from benchmark_utils_predator import rotation_error, translation_error
from SC2_PCR import Matcher


# ----------------------------------------------------------------------------- #
# Hard-coded configuration                                                      #
# ----------------------------------------------------------------------------- #
WORKING_DIR = osp.dirname(osp.realpath(__file__))
ROOT_DIR = WORKING_DIR
ALGORITHM_NAME = 'SC2-PCR'
KITTI_CONFIG_PATH = osp.join(ROOT_DIR, 'config_json', 'config_KITTI.json')
RUN_LOG_PATH = osp.join(ROOT_DIR, 'custom_infer_log.txt')

TARGET_POINT_CLOUD = '/home/xchu/Downloads/ltloc-results/simulated_pc/shifted_cylinder/measured_cloud_shifted_cylinder.pcd'  # 更新为真实 target PCD
SOURCE_POINT_CLOUD = None  # 若已有 source PCD，可在此填写路径；留空则由目标生成
INITIAL_POSE = (0.2, 0.8, 0.5, 0.1, 0.1, 2.0)  # (tx, ty, tz, roll, pitch, yaw)
POSE_IN_RADIANS = False  # 若 INITIAL_POSE 中角度已是弧度，则改为 True
FORCE_GT_IDENTITY = True  # True: 评价时将 GT 视为单位阵

USE_GPU = True  # True：若可用则使用 GPU；False：始终使用 CPU
USE_FPFH_PIPELINE = True  # True: 使用 FPFH 流程（不降采样）；False: 直接使用完整点云
INLIER_THRESHOLD = 0.2  # meters，用于点云评估
MAX_MATCHER_ITERATIONS = 30
NORMAL_KNN = 5
SEARCH_RADIUS_METERS = 1.0

# ===== BEGIN CHANGE: point-to-plane rmse evaluation =====
POINT_TO_PLANE_MAX_PLANE_THICKNESS = 0.2  # meters
POINT_TO_PLANE_WEIGHT_SLOPE = 0.9
POINT_TO_PLANE_MIN_EFFECTIVE_WEIGHT = 0.1
POINT_TO_PLANE_MIN_NORMAL_NORM = 1e-6
# ===== END CHANGE: point-to-plane rmse evaluation =====

def load_kitti_config(path):
    if not osp.isfile(path):
        raise FileNotFoundError(f'KITTI 配置文件不存在: {path}')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_matcher(config):
    return Matcher(
        inlier_threshold=config['inlier_threshold'],
        num_node=config['num_node'],
        use_mutual=config['use_mutual'],
        d_thre=config['d_thre'],
        num_iterations=config['num_iterations'],
        ratio=config['ratio'],
        nms_radius=config['nms_radius'],
        max_points=config['max_points'],
        k1=config['k1'],
        k2=config['k2'],
    )


def get_descriptor_settings(config):
    """
    提供 FPFH 所需的固定搜索参数（不依赖 KITTI JSON 中的 downsample）。
    """
    return {
        'type': config.get('descriptor', 'fpfh').lower(),
        'normal_knn': NORMAL_KNN,
        'feature_radius': SEARCH_RADIUS_METERS,
        'feature_max_nn': 100,
    }


def compute_fpfh_features(points, descriptor_cfg):
    import open3d as o3d

    if points.shape[0] == 0:
        raise ValueError('点云为空，无法计算 FPFH 特征。')

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if hasattr(o3d, 'pipelines') and hasattr(o3d.pipelines, 'registration'):
        registration_mod = o3d.pipelines.registration
    elif hasattr(o3d, 'registration'):
        registration_mod = o3d.registration
    else:
        raise AttributeError('当前 Open3D 版本缺少 registration 模块，无法计算 FPFH 特征。')

    normal_knn = descriptor_cfg.get('normal_knn', NORMAL_KNN)
    feature_radius = descriptor_cfg.get('feature_radius', SEARCH_RADIUS_METERS)
    feature_max_nn = descriptor_cfg.get('feature_max_nn', 100)
    normal_param = o3d.geometry.KDTreeSearchParamKNN(normal_knn)
    feature_param = o3d.geometry.KDTreeSearchParamHybrid(radius=feature_radius, max_nn=feature_max_nn)
    pcd.estimate_normals(normal_param)
    fpfh = registration_mod.compute_fpfh_feature(
        pcd,
        feature_param,
    )
    points_ds = np.asarray(pcd.points, dtype=np.float32)
    features = np.asarray(fpfh.data, dtype=np.float32).T
    features /= (np.linalg.norm(features, axis=1, keepdims=True) + 1e-6)
    return points_ds, features


def build_raw_point_features(points):
    keypoints = points.astype(np.float32)
    centered = keypoints - keypoints.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms = np.where(norms < 1e-6, 1.0, norms)
    features = centered / norms
    return keypoints, features


def prepare_matcher_inputs(points, device, descriptor_cfg=None, use_fpfh=False):
    if points.shape[0] == 0:
        raise ValueError('点云为空，无法构建匹配输入。')
    if use_fpfh:
        if descriptor_cfg is None:
            raise ValueError('使用 FPFH 流程时必须提供 descriptor 配置。')
        keypoints, features = compute_fpfh_features(points, descriptor_cfg)
    else:
        keypoints, features = build_raw_point_features(points)
    keypoints_tensor = torch.from_numpy(keypoints).float().to(device)[None, ...]
    features_tensor = torch.from_numpy(features).float().to(device)[None, ...]
    return keypoints_tensor, features_tensor


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


def get_transform_from_rotation_translation(rotation, translation):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform.astype(np.float32)


def inverse_transform(transform):
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    inv_rot = rot.T
    inv_trans = -inv_rot @ trans
    inv = np.eye(4, dtype=np.float32)
    inv[:3, :3] = inv_rot
    inv[:3, 3] = inv_trans
    return inv


def apply_transform(points, transform):
    if points.shape[0] == 0:
        return points
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    homogeneous = np.concatenate([points, ones], axis=1)
    transformed = homogeneous @ transform.T
    return transformed[:, :3]


# ===== BEGIN CHANGE: point-to-plane rmse evaluation =====
def _knn_query(tree, points, k=1, distance_upper_bound=np.inf):
    for kwargs in ({'workers': -1}, {'n_jobs': -1}, {}):
        try:
            return tree.query(points, k=k, distance_upper_bound=distance_upper_bound, **kwargs)
        except TypeError:
            continue
    return tree.query(points, k=k, distance_upper_bound=distance_upper_bound)


def compute_point_to_plane_rmse(
    aligned_points,
    target_points,
    normal_knn,
    search_radius,
    max_plane_thickness,
    weight_slope,
    min_effective_weight,
    min_normal_norm,
):
    if aligned_points.shape[0] == 0 or target_points.shape[0] < normal_knn:
        return float('inf'), 0.0, 0

    target_tree = cKDTree(target_points)
    neighbor_dists, neighbor_indices = _knn_query(
        target_tree,
        aligned_points,
        k=normal_knn,
        distance_upper_bound=search_radius,
    )

    if normal_knn == 1:
        neighbor_dists = neighbor_dists[:, None]
        neighbor_indices = neighbor_indices[:, None]

    plane_thickness_sq = max_plane_thickness * max_plane_thickness
    total_distance_sq = 0.0
    effective_corr_count = 0

    for point, dists, indices in zip(aligned_points, neighbor_dists, neighbor_indices):
        if not np.all(np.isfinite(dists)) or dists[-1] >= search_radius:
            continue

        neighbors = target_points[indices]
        plane_coeffs, _, _, _ = np.linalg.lstsq(
            neighbors.astype(np.float64),
            -np.ones(normal_knn, dtype=np.float64),
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

        total_distance_sq += point_to_plane_dist * point_to_plane_dist
        effective_corr_count += 1

    if effective_corr_count == 0:
        return float('inf'), 0.0, 0

    fitness = effective_corr_count / aligned_points.shape[0]
    rmse = float(np.sqrt(total_distance_sq / effective_corr_count))
    return rmse, fitness, effective_corr_count
# ===== END CHANGE: point-to-plane rmse evaluation =====


def compute_pointwise_metrics(aligned_points, target_points, inlier_threshold):
    if aligned_points.shape[0] == 0 or target_points.shape[0] == 0:
        return np.nan, np.nan, 0.0, 0

    target_tree = cKDTree(target_points)
    dists_src_to_tgt, _ = _knn_query(target_tree, aligned_points)

    source_tree = cKDTree(aligned_points)
    dists_tgt_to_src, _ = _knn_query(source_tree, target_points)

    chamfer = float(dists_src_to_tgt.mean() + dists_tgt_to_src.mean())

    inlier_mask = dists_src_to_tgt <= inlier_threshold
    inlier_count = int(inlier_mask.sum())
    if inlier_count > 0:
        rmse = float(np.sqrt(np.mean(np.square(dists_src_to_tgt[inlier_mask]))))
    else:
        rmse = float('inf')
    fitness = inlier_count / aligned_points.shape[0]

    return chamfer, rmse, fitness, inlier_count


def append_run_summary(summary):
    lines = []
    lines.append('=' * 60)
    lines.append(f"Timestamp: {summary['timestamp']}")
    lines.append(f"Algorithm: {summary['config']['algorithm']}")
    lines.append(f"Target PCD: {summary['config']['target_pcd']}")
    lines.append(f"Source PCD: {summary['config']['source_pcd']}")
    lines.append(f"Use FPFH: {summary['config']['use_fpfh']}")
    lines.append(f"Initial Pose: {summary['config']['initial_pose']}")
    lines.append(f"Force GT Identity: {summary['config']['force_gt_identity']}")
    lines.append(f"Matcher Iterations: {summary['config']['matcher_iterations']}")
    lines.append(f"Normal KNN: {summary['config']['normal_knn']}, Search Radius: {summary['config']['search_radius']} m")
    lines.append('--- Transform ---')
    for row in summary['transform']:
        lines.append('  ' + ' '.join(f'{val: .6f}' for val in row))
    lines.append('--- Metrics ---')
    for key, val in summary['metrics'].items():
        lines.append(f'{key}: {val}')
    lines.append('')
    with open(RUN_LOG_PATH, 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    # ===== BEGIN CHANGE: announce sc2-pcr algorithm =====
    print(f'[{ALGORITHM_NAME}] Starting custom single-pair registration inference.')
    # ===== END CHANGE: announce sc2-pcr algorithm =====

    if not osp.isfile(TARGET_POINT_CLOUD):
        raise FileNotFoundError(f'Target point cloud not found: {TARGET_POINT_CLOUD}. 请在 custom_infer.py 顶部更新路径。')
    if SOURCE_POINT_CLOUD is None and INITIAL_POSE is None:
        raise ValueError('未提供 SOURCE_POINT_CLOUD 时，必须设置 INITIAL_POSE。')

    config = load_kitti_config(KITTI_CONFIG_PATH)
    config['num_iterations'] = MAX_MATCHER_ITERATIONS
    descriptor_cfg = get_descriptor_settings(config) if USE_FPFH_PIPELINE else None

    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        if USE_GPU and not torch.cuda.is_available():
            print('Warning: CUDA not available, falling back to CPU. 设置 USE_GPU=False 可消除此警告。')
        device = torch.device('cpu')

    matcher = build_matcher(config)

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

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    src_keypts, src_feats = prepare_matcher_inputs(src_points, device, descriptor_cfg, USE_FPFH_PIPELINE)
    ref_keypts, ref_feats = prepare_matcher_inputs(ref_points, device, descriptor_cfg, USE_FPFH_PIPELINE)
    with torch.no_grad():
        est_transform, _, _, _ = matcher.estimator(src_keypts, ref_keypts, src_feats, ref_feats)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    runtime_ms = (time.perf_counter() - start_time) * 1000.0

    gt_transform_tensor = torch.from_numpy(transform_gt).float().to(device)[None, ...]
    rre = rotation_error(est_transform[:, :3, :3], gt_transform_tensor[:, :3, :3])
    rte = translation_error(est_transform[:, :3, 3:], gt_transform_tensor[:, :3, 3:])

    est_transform_np = est_transform.squeeze(0).detach().cpu().numpy()
    aligned_src_points = apply_transform(src_points.copy(), est_transform_np)

    # ===== BEGIN CHANGE: point-to-plane rmse evaluation =====
    chamfer, p2p_rmse, p2p_fitness, point_inlier_count = compute_pointwise_metrics(
        aligned_src_points, ref_points, INLIER_THRESHOLD
    )
    rmse, fitness, effective_corr_count = compute_point_to_plane_rmse(
        aligned_src_points,
        ref_points,
        NORMAL_KNN,
        SEARCH_RADIUS_METERS,
        POINT_TO_PLANE_MAX_PLANE_THICKNESS,
        POINT_TO_PLANE_WEIGHT_SLOPE,
        POINT_TO_PLANE_MIN_EFFECTIVE_WEIGHT,
        POINT_TO_PLANE_MIN_NORMAL_NORM,
    )
    # ===== END CHANGE: point-to-plane rmse evaluation =====

    print('Estimated transform:\n', est_transform_np)
    print('--- Metrics ---')
    print(f'Rotation Error (deg): {rre.item():.6f}')
    print(f'Translation Error (m): {rte.item():.6f}')
    print(f'Chamfer Distance (m): {chamfer:.6f}')
    print(f'RMSE (point-to-plane, m): {rmse:.6f}')
    print(f'ICP Fitness (effective/source): {fitness:.6f}')
    print(f'Point-to-Point RMSE (<= {INLIER_THRESHOLD} m, m): {p2p_rmse:.6f}')
    print(f'Point-to-Point Fitness (<= {INLIER_THRESHOLD} m): {p2p_fitness:.6f}')
    print(f'Effective Correspondence Count: {effective_corr_count}')
    print(f'Point-to-Point Inlier Count: {point_inlier_count}')
    print(f'Runtime (ms): {runtime_ms:.3f}')

    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'algorithm': ALGORITHM_NAME,
            'target_pcd': TARGET_POINT_CLOUD,
            'source_pcd': SOURCE_POINT_CLOUD or '<generated from target>',
            'initial_pose': INITIAL_POSE,
            'force_gt_identity': FORCE_GT_IDENTITY,
            'use_fpfh': USE_FPFH_PIPELINE,
            'matcher_iterations': MAX_MATCHER_ITERATIONS,
            'normal_knn': NORMAL_KNN,
            'search_radius': SEARCH_RADIUS_METERS,
        },
        'transform': est_transform_np.tolist(),
        'metrics': {
            'Rotation Error (deg)': f'{rre.item():.6f}',
            'Translation Error (m)': f'{rte.item():.6f}',
            'Chamfer Distance (m)': f'{chamfer:.6f}',
            'RMSE (point-to-plane, m)': f'{rmse:.6f}',
            'ICP Fitness (effective/source)': f'{fitness:.6f}',
            'Point-to-Point RMSE (<= 0.2 m, m)': f'{p2p_rmse:.6f}',
            'Point-to-Point Fitness (<= 0.2 m)': f'{p2p_fitness:.6f}',
            'Effective Correspondence Count': str(effective_corr_count),
            'Point-to-Point Inlier Count': str(point_inlier_count),
            'Runtime (ms)': f'{runtime_ms:.3f}',
        },
    }
    append_run_summary(summary)


if __name__ == '__main__':
    main()
