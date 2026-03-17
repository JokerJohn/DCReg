import torch
import numpy as np
import random
import math
import open3d as o3d
from utils.pointcloud import make_point_cloud


def exact_auc(errors, thresholds):
    """
    Calculate the exact area under curve, borrow from https://github.com/magicleap/SuperGluePretrainedNetwork
    """
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e) / t)
    return aucs


def set_seed(seed=51):
    """
    Set the random seed for reproduce the results.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def icp_refine(src_keypts, tgt_keypts, pred_trans):
    """
    ICP algorithm to refine the initial transformation
    Input:
        - src_keypts [1, num_corr, 3] FloatTensor
        - tgt_keypts [1, num_corr, 3] FloatTensor
        - pred_trans [1, 4, 4] FloatTensor, initial transformation
    """
    src_pcd = make_point_cloud(src_keypts.detach().cpu().numpy()[0])
    tgt_pcd = make_point_cloud(tgt_keypts.detach().cpu().numpy()[0])
    initial_trans = pred_trans[0].detach().cpu().numpy()
    # change the convension of transforamtion because open3d use left multi.
    refined_T = o3d.registration.registration_icp(
        src_pcd, tgt_pcd, 0.10, initial_trans,
        o3d.registration.TransformationEstimationPointToPoint()).transformation
    refined_T = torch.from_numpy(refined_T[None, :, :]).to(pred_trans.device).float()
    return refined_T


def is_rotation_matrix(R):
    """    
    Checks if a matrix is a valid rotation matrix.
    Input:
        - R: [3, 3] rotation matrix
    Output:
        - True/False
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-3


def rot_to_euler(R):
    """    
    Convert the rotation matrix to euler angles(degree)
    Input:
        - R: [3, 3] rotation matrix
    Output:
        - alpha. [3], the rotation angle (in degrees) along x,y,z axis.
    """
    assert (is_rotation_matrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x * 180 / np.pi, y * 180 / np.pi, z * 180 / np.pi])