import math
import os.path as osp
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

# ----------------------------------------------------------------------------- #
# 固定配置：修改这里即可切换数据目录与评估设置
# ----------------------------------------------------------------------------- #
# BASE_DIR = '/home/xchu/data/ltloc_result/geo_transformer/stairs_bob_geo/geo_results'
# GROUND_TRUTH_FILENAME = 'stairs_bob.txt'

# BASE_DIR = '/home/xchu/data/ltloc_result/geo_transformer/20220216_corridor_day_ref_geo/geo_results'
# GROUND_TRUTH_FILENAME = '20220216_corridor_day_fp.txt'

# BASE_DIR = '/home/xchu/data/ltloc_result/geo_transformer/20220225_building_day_ref_geo/geo_results_60'
# GROUND_TRUTH_FILENAME = '20220225_building_day.txt'

BASE_DIR = '/home/xchu/data/ltloc_result/geo_transformer/cave02_geo/geo_results'
GROUND_TRUTH_FILENAME = 'cave02_gt.txt'

# BASE_DIR = '/home/xchu/data/ltloc_result/geo_transformer/parkinglot_raw_geo/geo_results'
# GROUND_TRUTH_FILENAME = 'Parkinglot-2023-10-28-18-59-01_0.005_ins_tum.txt'


METHOD_FILES = {
    'odom': 'odom.txt',
    # 'o3d': 'o3d.txt',
    'geo': 'geo_tum.txt',
    'me-sr': 'mesr.txt',
    'tsvd': 'tsvd.txt',
    'treg': 'treg.txt',
    'fcn-sr': 'fcnsr.txt',
    'ours': 'ours.txt',
}


# stairs
METHOD_TRANSFORMS = {
    # 'odom': [ -0.519301, 0.850557, 0.082936 ,-11.347226, 
    #               -0.852164, -0.522691, 0.024698, 3.002144,
    #               0.064357, -0.057849, 0.996249, -0.715776,
    #               0.000000, 0.000000, 0.000000, 1.000000 ],
    # 'odom': [0.962393,  -0.269109 , 0.0371485 ,   6.26396, #CORRIDOR
    #             0.267793,   0.962772 , 0.0368319  ,0.0850816,
    #             -0.0456773, -0.0254987  , 0.998631  , 0.792745,
    #             0,        0,       0,        1 ],
    # 'odom': [ 0.448165, -0.893951 ,-0.000477, -41.163830, #building
    #             0.891230, 0.446760 ,0.078197, -46.008873,
    #             -0.069691, -0.035470, 0.996938, 0.261990,
    #             0.000000, 0.000000, 0.000000, 1.000000 ],
    'odom': [ 0.274473488331, 0.961305737495, 0.023577133194,33.809207916260, #cave02
                  -0.961209475994, 0.274974942207, -0.021568179131,-72.459877014160,
                  -0.027216726914, -0.016742672771, 0.999489367008,21.207557678223,
                  0,      0,     0,      1 ],
    # 'odom':  [  0.76876314, -0.6395325 ,  0.00119299, 282.75579377, #parkinglot
    #             0.63904927,  0.76825218,  0.03747841, -55.82057474,
    #             -0.02488518, -0.02804964,  0.99929673, -0.71615374,
    #             0.,          0.,          0.,          1.        ],


    # 'o3d': None,
    'o3d':  [ 0.27480901, -0.9611057, -0.02749245, -78.34210115,#cave02 
        0.961186,    0.27533661, -0.01764175, -12.17765038,
        0.02452527, -0.02157725,  0.99946632, -23.58530797,
        0.,          0.,          0.,          1.        ],
    # 'o3d': None,

    # 'geo': None,
    'geo': [0.27452767, -0.96119587, -0.02714879, -78.35699512, #cave02
                0.96129376,  0.27502465, -0.01660542, -12.21854335,
                0.02342765, -0.02153932,  0.99949347, -23.55019242,
                    0.,          0.,          0.,          1.        ],

    # 'geo':  [ 9.58634309e-01,  2.84640539e-01, -1.60036438e-04 ,649.42757157 , #parkinglot
    #         -2.84639983e-01,  9.58633289e-01,  1.51573581e-03, 391.58017046,
    #         5.84856114e-04, -1.40748358e-03,  9.99998838e-01, 0.73059172,
    #         0, 0, 0, 1],


    # 'me-sr': None,
    'me-sr': [ 0.25606466, -0.96368973,  0.07571653, -76.6122192,  #PK01
                0.96526279,  0.2591184,   0.03354686, -11.35897773,
                -0.05194831,  0.06449618,  0.99656491, -24.26807413,
                0, 0, 0, 1 ],
    # 'me-sr': [ 9.56856562e-01,  2.90560693e-01, -6.06851982e-05, 652.02678867, #parkinglot
    #         -2.90560305e-01,  9.56855587e-01,  1.44747997e-03, 390.13966664,
    #          4.78647754e-04, -1.36739800e-03,  9.99998951e-01, 0.74844327,
    #          0. ,         0.   ,       0.   ,       1.        ],



    # 'tsvd': None,
    'tsvd': [ 0.20561696, -0.96916571,  0.13579209, -78.47409686,
               0.96989028,  0.18330341, -0.16035183, -11.92592464,
               0.13051634,  0.16467449,  0.97767469, -22.29047258,
               0.,          0.,          0.,          1.        ],
    # 'tsvd': [9.58609907e-01,  2.84722711e-01, -1.58341678e-04, 649.436674,  #parkinglot
    #         -2.84722162e-01,  9.58608898e-01,  1.50689670e-03,  391.53451211,
    #         5.80835456e-04, -1.39944272e-03,  9.99998852e-01, 0.73547153,
    #         0.,         0.,         0.,         1.        ],



    # 'treg': None,
    'treg': [0.2613755,  -0.96489518,  0.02569315, -77.89600107,
        0.96389586,  0.25951699, -0.05962968, -11.79536861,
        0.05086858,  0.04035126,  0.99788986, -23.62278773,
        0, 0, 0, 1],
    # 'treg': [ 9.57857533e-01,  2.87243662e-01, -1.61349743e-04, 650.73378359, #parkinglot
    #         -2.87243137e-01,  9.57856599e-01,  1.45503815e-03, 390.81997075,
    #         5.72500402e-04, -1.34737265e-03,  9.99998928e-01,  0.76192944,
    #         0,         0,       0,       1],

    # 'fcn-sr': None,
     'fcn-sr': [ 0.12928489, -0.98491115,  0.11504539, -79.13894675,
                 0.8274384,   0.17108987,  0.53485882, -22.45992177,
                -0.54647152,  0.02604381,  0.83707264, -3.10159705,
                 0, 0, 0, 1],
    # 'fcn-sr': [ 0.95620204 , 0.29269839, -0.00230663, 6.53209289e+02, #parkinglot
    #         -0.2926869,   0.95619857,  0.00432194, 3.89294975e+02,
    #             0.00347063, -0.00345752,  0.999988,  4.48267741e-02,
    #             0,          0,          0,          1        ],



    # 'ours': None,
    'ours': [0.27452767, -0.96119587, -0.02714879, -78.35699512, #cave02
                0.96129376,  0.27502465, -0.01660542, -12.21854335,
                0.02342765, -0.02153932,  0.99949347, -23.55019242,
                    0.,          0.,          0.,          1.        ],
    # 'ours': [ 9.58634309e-01,  2.84640539e-01, -1.60036438e-04 ,649.42757157 ,#parkinglot
    #         -2.84639983e-01,  9.58633289e-01,  1.51573581e-03, 391.58017046,
    #         5.84856114e-04, -1.40748358e-03,  9.99998838e-01, 0.73059172,
    #         0, 0, 0, 1],
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


def collect_methods_via_constants() -> List[Tuple[str, str]]:
    methods = []
    for name, relative_path in METHOD_FILES.items():
        absolute_path = osp.normpath(osp.join(BASE_DIR, relative_path))
        methods.append((name, absolute_path))
    return methods


def main():
    ground_truth_path = osp.join(BASE_DIR, GROUND_TRUTH_FILENAME)
    if not osp.isfile(ground_truth_path):
        raise FileNotFoundError(f'Ground truth trajectory not found: {ground_truth_path}')
    methods = collect_methods_via_constants()
    if not methods:
        raise RuntimeError('No methods defined in METHOD_FILES.')

    gt_poses = load_tum_file(ground_truth_path)
    if not gt_poses:
        raise RuntimeError(f'Ground truth trajectory {ground_truth_path} is empty or invalid.')

    summaries = []
    for name, path in methods:
        if not osp.isfile(path):
            raise FileNotFoundError(f'Trajectory for method "{name}" not found: {path}')
        method_poses = load_tum_file(path)
        init_transform = get_transform_matrix(METHOD_TRANSFORMS.get(name))
        method_poses = apply_global_transform(method_poses, init_transform)
        summary = evaluate_method(name, method_poses, gt_poses, MAX_TIME_DIFF, RRE_THRESHOLD, RTE_THRESHOLD)
        summaries.append(summary)

    output_path = osp.join(BASE_DIR, OUTPUT_FILENAME)
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
