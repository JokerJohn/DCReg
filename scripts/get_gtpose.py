import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd

def read_tum_trajectory(filename):
    """
    Read trajectory in TUM format
    Format: timestamp tx ty tz qx qy qz qw
    """
    data = pd.read_csv(filename, sep=' ', header=None, 
                       names=['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
    return data

def find_nearest_pose(gt_trajectory, query_timestamp, time_offset=0.0, max_time_diff=0.1):
    """
    Find the nearest pose in ground truth trajectory based on timestamp
    
    Args:
        gt_trajectory: pandas DataFrame with TUM format trajectory
        query_timestamp: timestamp to query
        time_offset: time offset to add to query_timestamp
        max_time_diff: maximum allowed time difference (seconds)
    
    Returns:
        nearest_pose: dict with timestamp and 4x4 transformation matrix
    """
    # Apply time offset
    adjusted_timestamp = query_timestamp + time_offset
    
    # Find nearest timestamp
    time_diffs = np.abs(gt_trajectory['timestamp'] - adjusted_timestamp)
    min_idx = time_diffs.idxmin()
    min_diff = time_diffs[min_idx]
    
    # Check if the time difference is within acceptable range
    if min_diff > max_time_diff:
        print(f"Warning: Nearest timestamp has {min_diff:.3f}s difference, exceeding max allowed {max_time_diff}s")
        return None
    
    # Extract pose at nearest timestamp
    pose_data = gt_trajectory.iloc[min_idx]
    
    # Convert quaternion to rotation matrix
    quat = [pose_data['qx'], pose_data['qy'], pose_data['qz'], pose_data['qw']]
    rotation = R.from_quat(quat)
    rotation_matrix = rotation.as_matrix()
    
    # Build 4x4 transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = [pose_data['tx'], pose_data['ty'], pose_data['tz']]
    
    return {
        'timestamp': pose_data['timestamp'],
        'original_timestamp': query_timestamp,
        'time_diff': min_diff,
        'transform': transform_matrix
    }

def transform_gt_to_icp(gt_pose_matrix, icp_to_gt_transform):
    """
    Transform GT pose back to ICP map frame
    
    Args:
        gt_pose_matrix: 4x4 pose matrix in GT coordinate system
        icp_to_gt_transform: 4x4 transformation matrix from ICP to GT
    
    Returns:
        icp_pose: 4x4 pose matrix in ICP coordinate system
    """
    # If T_gt = T_icp_to_gt @ T_icp, then T_icp = T_icp_to_gt^(-1) @ T_gt
    # Use the inverse of the ICP-to-GT transform
    gt_to_icp_transform = np.linalg.inv(icp_to_gt_transform)
    icp_pose = gt_to_icp_transform @ gt_pose_matrix
    return icp_pose

def pose_to_tum_format(timestamp, pose_matrix):
    """
    Convert 4x4 pose matrix to TUM format string
    """
    # Extract translation
    tx, ty, tz = pose_matrix[:3, 3]
    
    # Extract rotation and convert to quaternion
    rotation_matrix = pose_matrix[:3, :3]
    rotation = R.from_matrix(rotation_matrix)
    qx, qy, qz, qw = rotation.as_quat()
    
    return f"{timestamp:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}"

def matrix_to_euler_xyz(matrix):
    """
    Convert a 4x4 pose matrix to Euler angles in XYZ order (roll, pitch, yaw)
    
    Args:
        matrix: 4x4 pose matrix
    
    Returns:
        euler_angles: array of [roll, pitch, yaw] in degrees
    """
    rotation_matrix = matrix[:3, :3]
    rotation = R.from_matrix(rotation_matrix)
    # 'xyz' means roll around x, then pitch around y, then yaw around z
    euler_rad = rotation.as_euler('xyz', degrees=False)
    euler_deg = np.degrees(euler_rad)
    return euler_deg

def main():
    # ===== Configuration =====
    # Input timestamp (can be directly set here)
    query_timestamp = 1698490984.107097864  # 1976
    
    # Time offset between sensors (if needed)
    time_offset = 0.0  # seconds, adjust if there's systematic time difference
    
    # Coordinate transformation from ICP to GT (4x4 matrix)
    # This is the transformation from ICP coordinate to GT coordinate
    # icp_to_gt_transform = np.array([
    #     [0.95815046, 0.28626486, -0.00035885, 650.247619],
    #     [-0.28626212, 0.95814548, 0.00335208, 391.113304],
    #     [0.00130341, -0.00310907, 0.99999432, -0.04985837],
    #     [0.0, 0.0, 0.0, 1.0]
    # ])
    icp_to_gt_transform = np.array([
        [0.95866517, 0.28453655, -0.00018691, 649.38716756],
        [-0.28453596, 0.95866416, 0.00152137, 391.60799651],
        [0.00061207, -0.00140531, 0.99999882, 0.75341653],
        [0.0, 0.0, 0.0, 1.0]
    ])




    # Ground truth trajectory file path
    gt_trajectory_file = "/home/xchu/data/ltloc_result/parkinglot_raw/Parkinglot-2023-10-28-18-59-01_0.005_ins_tum.txt"
    
    # Maximum allowed time difference (seconds)
    max_time_diff = 0.01  # 10ms
    
    # ===== Process =====
    # Read ground truth trajectory
    print(f"Reading ground truth trajectory from: {gt_trajectory_file}")
    gt_trajectory = read_tum_trajectory(gt_trajectory_file)
    print(f"Loaded {len(gt_trajectory)} poses")
    
    # Find nearest pose
    print(f"\nSearching for pose at timestamp: {query_timestamp}")
    nearest_pose = find_nearest_pose(gt_trajectory, query_timestamp, time_offset, max_time_diff)
    
    if nearest_pose is None:
        print("Error: No pose found within acceptable time range")
        return
    
    print(f"Found pose at timestamp: {nearest_pose['timestamp']}")
    print(f"Time difference: {nearest_pose['time_diff']:.6f} seconds")
    
    # Transform GT pose back to ICP map frame
    icp_pose = transform_gt_to_icp(nearest_pose['transform'], icp_to_gt_transform)
    
    # Output results
    print("\n===== Results =====")
    print(f"Original GT pose (4x4 matrix):")
    print(nearest_pose['transform'])
    
    print(f"\nICP-to-GT transformation matrix:")
    print(icp_to_gt_transform)
    
    print(f"\nGT-to-ICP transformation matrix (inverse):")
    print(np.linalg.inv(icp_to_gt_transform))
    
    print(f"\nTransformed pose in ICP/map coordinate (4x4 matrix):")
    print(icp_pose)
    
    # Get Euler angles
    euler_angles = matrix_to_euler_xyz(icp_pose)
    print(f"\nEuler angles (roll, pitch, yaw) in degrees:")
    print(f"Roll (X):  {euler_angles[0]:.6f}°")
    print(f"Pitch (Y): {euler_angles[1]:.6f}°")
    print(f"Yaw (Z):   {euler_angles[2]:.6f}°")
    
    # Convert to TUM format for easy comparison
    tum_string = pose_to_tum_format(nearest_pose['timestamp'], icp_pose)
    print(f"\nTransformed pose in TUM format:")
    print(tum_string)
    
    # Save result to file
    output_file = f"gt_pose_in_icp_frame_{int(query_timestamp)}.txt"
    with open(output_file, 'w') as f:
        f.write("# Ground truth pose transformed to ICP/map frame\n")
        f.write(f"# Query timestamp: {query_timestamp}\n")
        f.write(f"# GT timestamp: {nearest_pose['timestamp']}\n")
        f.write(f"# Time difference: {nearest_pose['time_diff']:.6f}s\n")
        f.write("#\n")
        
        # TUM format
        f.write("# TUM format: timestamp tx ty tz qx qy qz qw\n")
        f.write(tum_string + "\n\n")
        
        # 4x4 Matrix
        f.write("# 4x4 Transformation Matrix:\n")
        for i in range(4):
            for j in range(4):
                f.write(f"{icp_pose[i,j]:.6f}")
                if j < 3:
                    f.write(" ")
            f.write("\n")
        f.write("\n")
        
        # Translation
        f.write("# Translation (x, y, z):\n")
        f.write(f"{icp_pose[0,3]:.6f} {icp_pose[1,3]:.6f} {icp_pose[2,3]:.6f}\n\n")
        
        # Euler angles
        f.write("# Euler angles in degrees (roll, pitch, yaw) - XYZ order:\n")
        f.write(f"{euler_angles[0]:.6f} {euler_angles[1]:.6f} {euler_angles[2]:.6f}\n")
        f.write(f"# Roll (X):  {euler_angles[0]:.6f}°\n")
        f.write(f"# Pitch (Y): {euler_angles[1]:.6f}°\n")
        f.write(f"# Yaw (Z):   {euler_angles[2]:.6f}°\n")
        
    print(f"\nSaved result to: {output_file}")

    #  add some codes to transform my pose to euler angles
    # my pose is
    # my_pose = np.array([
    #     [-0.932253, -0.299166, -0.203480, -85.258712],
    #     [0.287973, -0.954011, 0.083272, -464.541501],
    #     [-0.219034, 0.019034, 0.975532, -1.092935],
    #     [0.000000, 0.000000, 0.000000, 1.000000]
    # ])
    # my_pose = my_pose.reshape(4, 4)
    # my_euler_angles = matrix_to_euler_xyz(my_pose)
    # print(f"\nMy pose in Euler angles (roll, pitch, yaw) in degrees:")
    # print(f"Roll (X):  {my_euler_angles[0]:.6f}°")
    # print(f"Pitch (Y): {my_euler_angles[1]:.6f}°")
    # print(f"Yaw (Z):   {my_euler_angles[2]:.6f}°")
    # # print translation
    # print(f"\nMy pose translation (x, y, z):")
    # print(f"{my_pose[0,3]:.6f} {my_pose[1,3]:.6f} {my_pose[2,3]:.6f}")

if __name__ == "__main__":
    main()