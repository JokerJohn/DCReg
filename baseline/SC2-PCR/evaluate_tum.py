import argparse
import math
import os.path as osp
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

# ----------------------------------------------------------------------------- #
# 固定配置：修改这里即可切换数据目录与评估设置
# ----------------------------------------------------------------------------- #
# BASE_DIR = '/home/xchu/data/ltloc_result/geo_transformer/20220216_corridor_day_ref_geo/sc2-pcr_results'
# GROUND_TRUTH_FILENAME = '20220216_corridor_day_fp.txt'

# BASE_DIR = '/home/xchu/data/ltloc_result/geo_transformer/20220225_building_day_ref_geo/sc2-pcr_results'
# GROUND_TRUTH_FILENAME = '20220225_building_day.txt'

# the gt of cave was collected by other sensors, we should use the align metarix from evo
# BASE_DIR = '/home/xchu/data/ltloc_result/geo_transformer/cave02_geo/sc2-pcr_results'
# GROUND_TRUTH_FILENAME = 'cave02_gt.txt'

# BASE_DIR = '/home/xchu/data/ltloc_result/geo_transformer/stairs_bob_geo/sc2-pcr_results'
# GROUND_TRUTH_FILENAME = 'stairs_bob.txt'

BASE_DIR = '/home/xchu/data/ltloc_result/geo_transformer/parkinglot_raw_geo/sc2-pcr_results'
GROUND_TRUTH_FILENAME = 'Parkinglot-2023-10-28-18-59-01_0.005_ins_tum.txt'


METHOD_FILES = {
    'sc2pcr_raw': 'sc2pcr_raw.txt',
    'sc2pcr_fpfh': 'sc2pcr_fpfh.txt',
}

# 针对默认数据的坐标系对齐矩阵（若输出已在地图坐标系，可设为 None）
METHOD_TRANSFORMS = {
    
    # 'sc2pcr_raw': None,
    # 'sc2pcr_raw':  [0.47169305, -0.88167747,  0.012268, -76.33166652, #cave02
    #                 0.87642797,  0.46726569 ,-0.11634772 ,-7.61253097,
    #                 0.09684875 , 0.06563243 , 0.99313277 ,-21.59179161,
    #                     0,          0,          0,          1        ],
    'sc2pcr_raw':  [ 0.95262182,  0.30170595, -0.0385382,  671.14904087, #pk01
                    -0.30204938,  0.95328662, -0.00328467, 382.228027,
                    0.03574694,  0.01476949 , 0.99925173, 10.62040281,
                    0  ,        0     ,     0,          1        ],


    # 'sc2pcr_fpfh': None,
    #  'sc2pcr_fpfh': [ 0.58367397, -0.81046122, -0.04977256, -72.31513802, #cave02
    #                 0.81120337,  0.57931911,  0.07961452, -7.91938851,
    #                 -0.03569028, -0.08684459,  0.99558235, -22.29082439,
    #                 0,          0,          0,          1        ],
    'sc2pcr_fpfh':  [ 9.69134928e-01,  2.44078588e-01, -3.46862142e-02, 645.29738372,
                    -2.44226422e-01,  9.69718234e-01, -2.59079018e-05, 400.15770055,
                    3.36295308e-02,  8.49639823e-03,  9.99398252e-01 ,8.33160979,
                    0 ,0, 0, 1],
}

OUTPUT_FILENAME = 'evaluation_summary.txt'
MAX_TIME_DIFF = 0.01  # seconds
RRE_THRESHOLD = 5.0  # degrees
RTE_THRESHOLD = 0.2  # meters
# ----------------------------------------------------------------------------- #


def load_tum_file(path: str) -> List[Dict]:
    poses = []
    with open(path, 'r') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            timestamp = float(parts[0])
            translation = np.asarray(parts[1:4], dtype=np.float64)
            quat = np.asarray(parts[4:8], dtype=np.float64)
            norm = np.linalg.norm(quat)
            if norm == 0:
                continue
            quat = quat / norm
            rotation = Rotation.from_quat(quat).as_matrix()
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = rotation
            transform[:3, 3] = translation
            poses.append({'timestamp': timestamp, 'transform': transform})
    poses.sort(key=lambda x: x['timestamp'])
    return poses


def get_transform_matrix(values: Optional[List[float]]) -> Optional[np.ndarray]:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.size != 16:
        raise ValueError('Transform list must contain exactly 16 values.')
    return array.reshape(4, 4)


def apply_global_transform(poses: List[Dict], matrix: Optional[np.ndarray]) -> List[Dict]:
    if matrix is None:
        return poses
    transformed = []
    for pose in poses:
        new_transform = matrix @ pose['transform']
        transformed.append({'timestamp': pose['timestamp'], 'transform': new_transform})
    return transformed


def compute_rre_rte(est: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    rot_est = est[:3, :3]
    rot_gt = gt[:3, :3]
    rot_delta = rot_est @ rot_gt.T
    trace = np.trace(rot_delta)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    rre = math.degrees(math.acos(cos_theta))
    rte = float(np.linalg.norm(est[:3, 3] - gt[:3, 3]))
    return rre, rte


def find_nearest_pose(timestamp: float, gt_timestamps: np.ndarray) -> Tuple[int, float]:
    idx = int(np.argmin(np.abs(gt_timestamps - timestamp)))
    time_diff = float(abs(gt_timestamps[idx] - timestamp))
    return idx, time_diff


@dataclass
class FrameEval:
    index: int
    timestamp: float
    gt_timestamp: Optional[float]
    time_diff: Optional[float]
    rre_deg: float
    rte_m: float
    success: bool


def evaluate_method(
    name: str,
    poses: List[Dict],
    gt_poses: List[Dict],
    max_time_diff: Optional[float],
    rre_threshold: float,
    rte_threshold: float,
) -> Dict:
    if not poses:
        return {
            'name': name,
            'frames': [],
            'total': 0,
            'matched': 0,
            'avg_rre_deg': float('nan'),
            'avg_rte_m': float('nan'),
            'rr': float('nan'),
        }
    gt_timestamps = np.asarray([p['timestamp'] for p in gt_poses], dtype=np.float64)
    frames: List[FrameEval] = []
    rre_values = []
    rte_values = []
    success_count = 0
    matched = 0
    time_diffs = []

    for idx, pose in enumerate(poses):
        if gt_timestamps.size == 0:
            frames.append(FrameEval(idx, pose['timestamp'], None, None, float('nan'), float('nan'), False))
            continue
        nearest_idx, time_diff = find_nearest_pose(pose['timestamp'], gt_timestamps)
        if max_time_diff is not None and time_diff > max_time_diff:
            frames.append(FrameEval(idx, pose['timestamp'], gt_timestamps[nearest_idx], time_diff, float('nan'), float('nan'), False))
            continue
        gt_pose = gt_poses[nearest_idx]
        rre, rte = compute_rre_rte(pose['transform'], gt_pose['transform'])
        matched += 1
        rre_values.append(rre)
        rte_values.append(rte)
        time_diffs.append(time_diff)
        success = (rre <= rre_threshold) and (rte <= rte_threshold)
        if success:
            success_count += 1
        frames.append(FrameEval(idx, pose['timestamp'], gt_pose['timestamp'], time_diff, rre, rte, success))

    avg_rre = float(np.mean(rre_values)) if rre_values else float('nan')
    avg_rte = float(np.mean(rte_values)) if rte_values else float('nan')
    rr = (success_count / matched) if matched > 0 else float('nan')
    avg_time_diff = float(np.mean(time_diffs)) if time_diffs else float('nan')
    max_time_diff = float(np.max(time_diffs)) if time_diffs else float('nan')

    return {
        'name': name,
        'frames': frames,
        'total': len(poses),
        'matched': matched,
        'avg_rre_deg': avg_rre,
        'avg_rte_m': avg_rte,
        'rr': rr,
        'avg_time_diff': avg_time_diff,
        'max_time_diff': max_time_diff,
    }


def write_results(path: str, summaries: List[Dict]):
    with open(path, 'w') as f:
        for summary in summaries:
            f.write(f'Method: {summary["name"]}\n')
            f.write(f'  frames_total: {summary["total"]}\n')
            f.write(f'  frames_matched: {summary["matched"]}\n')
            f.write(f'  avg_rre_deg: {summary["avg_rre_deg"]:.6f}\n')
            f.write(f'  avg_rte_m: {summary["avg_rte_m"]:.6f}\n')
            f.write(f'  registration_recall: {summary["rr"]:.6f}\n')
            f.write(f'  avg_time_diff: {summary["avg_time_diff"]:.6f}\n')
            f.write(f'  max_time_diff: {summary["max_time_diff"]:.6f}\n')
            f.write('  per_frame:\n')
            f.write('    index timestamp gt_timestamp time_diff rre_deg rte_m success\n')
            for frame in summary['frames']:
                f.write(
                    f'    {frame.index} '
                    f'{frame.timestamp:.9f} '
                    f'{frame.gt_timestamp if frame.gt_timestamp is not None else float("nan"):.9f} '
                    f'{frame.time_diff if frame.time_diff is not None else float("nan"):.6f} '
                    f'{frame.rre_deg:.6f} '
                    f'{frame.rte_m:.6f} '
                    f'{int(frame.success)}\n'
                )
            f.write('\n')


def collect_methods_via_constants(base_dir: str) -> List[Tuple[str, str]]:
    methods = []
    for name, relative_path in METHOD_FILES.items():
        absolute_path = osp.normpath(osp.join(base_dir, relative_path))
        methods.append((name, absolute_path))
    return methods


def parse_args():
    parser = argparse.ArgumentParser(description='评估 TUM 轨迹 (支持 SC2-PCR FPFH/原始版本)。')
    parser.add_argument('--base_dir', type=str, default=BASE_DIR, help='包含待评估轨迹的目录')
    parser.add_argument('--gt', type=str, default=GROUND_TRUTH_FILENAME, help='GT 轨迹文件名或路径（相对 base_dir）')
    parser.add_argument('--max_time_diff', type=float, default=MAX_TIME_DIFF, help='允许的时间戳差 (秒)')
    parser.add_argument('--rre_threshold', type=float, default=RRE_THRESHOLD, help='RRE 成功阈值 (度)')
    parser.add_argument('--rte_threshold', type=float, default=RTE_THRESHOLD, help='RTE 成功阈值 (米)')
    parser.add_argument('--skip_missing', dest='skip_missing', action='store_true', default=True, help='若某方法轨迹缺失则跳过')
    parser.add_argument('--no_skip_missing', dest='skip_missing', action='store_false', help='缺失轨迹时直接报错')
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = args.base_dir

    ground_truth_path = args.gt
    if not osp.isabs(ground_truth_path):
        ground_truth_path = osp.join(base_dir, ground_truth_path)
    if not osp.isfile(ground_truth_path):
        raise FileNotFoundError(f'Ground truth trajectory not found: {ground_truth_path}')
    methods = collect_methods_via_constants(base_dir)
    if not methods:
        raise RuntimeError('No methods defined in METHOD_FILES.')

    gt_poses = load_tum_file(ground_truth_path)
    if not gt_poses:
        raise RuntimeError(f'Ground truth trajectory {ground_truth_path} is empty or invalid.')

    summaries = []
    for name, path in methods:
        if not osp.isfile(path):
            msg = f'Trajectory for method "{name}" not found: {path}'
            if args.skip_missing:
                print(f'[WARN] {msg} -> skip')
                continue
            raise FileNotFoundError(msg)
        method_poses = load_tum_file(path)
        init_transform = get_transform_matrix(METHOD_TRANSFORMS.get(name))
        method_poses = apply_global_transform(method_poses, init_transform)
        summary = evaluate_method(
            name,
            method_poses,
            gt_poses,
            args.max_time_diff,
            args.rre_threshold,
            args.rte_threshold,
        )
        summaries.append(summary)

    if not summaries:
        raise RuntimeError('没有可评估的方法轨迹，请检查路径配置。')

    output_path = osp.join(base_dir, OUTPUT_FILENAME)
    write_results(output_path, summaries)
    print(f'Evaluation finished. Results written to {output_path}')
    print('Overall metrics:')
    for summary in summaries:
        print(
            f"  {summary['name']}: "
            f"frames={summary['total']} matched={summary['matched']} "
            f"avg_rre_deg={summary['avg_rre_deg']:.4f} "
            f"avg_rte_m={summary['avg_rte_m']:.4f} "
            f"RR={summary['rr']:.4f} "
            f"avg_delta_t={summary['avg_time_diff']:.4f}s "
            f"max_delta_t={summary['max_time_diff']:.4f}s"
        )
    print('建议：若 avg/max 时间差偏大，请检查 GT 与估计轨迹时间戳是否可靠或需要同步处理。')


if __name__ == '__main__':
    main()
