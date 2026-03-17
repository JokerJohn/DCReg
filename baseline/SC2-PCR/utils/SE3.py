import torch
import random
import numpy as np

def rotation_matrix(num_axis, augment_rotation):
    """
    Sample rotation matrix along [num_axis] axis and [0 - augment_rotation] angle
    Input
        - num_axis:          rotate along how many axis
        - augment_rotation:  rotate by how many angle
    Output
        - R: [3, 3] rotation matrix
    """
    assert num_axis == 1 or num_axis == 3 or num_axis == 0
    if  num_axis == 0:
        return np.eye(3)
    angles = np.random.rand(3) * 2 * np.pi * augment_rotation
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    # R = Rx @ Ry @ Rz
    if num_axis == 1:
        return random.choice([Rx, Ry, Rz]) 
    return Rx @ Ry @ Rz

def translation_matrix(augment_translation):
    """
    Sample translation matrix along 3 axis and [augment_translation] meter
    Input
        - augment_translation:  translate by how many meters
    Output
        - t: [3, 1] translation matrix
    """
    T = np.random.rand(3) * augment_translation
    return T.reshape(3, 1)
    
def transform(pts, trans):
    """
    Applies the SE3 transformations, support torch.Tensor and np.ndarry.  Equation: trans_pts = R @ pts + t
    Input
        - pts: [num_pts, 3] or [bs, num_pts, 3], pts to be transformed
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    Output
        - pts: [num_pts, 3] or [bs, num_pts, 3] transformed pts
    """
    if len(pts.shape) == 3:
        trans_pts = trans[:, :3, :3] @ pts.permute(0,2,1) + trans[:, :3, 3:4]
        return trans_pts.permute(0,2,1)
    else:
        trans_pts = trans[:3, :3] @ pts.T + trans[:3, 3:4]
        return trans_pts.T

def decompose_trans(trans):
    """
    Decompose SE3 transformations into R and t, support torch.Tensor and np.ndarry.
    Input
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    Output
        - R: [3, 3] or [bs, 3, 3], rotation matrix
        - t: [3, 1] or [bs, 3, 1], translation matrix
    """
    if len(trans.shape) == 3:
        return trans[:, :3, :3], trans[:, :3, 3:4]
    else:
        return trans[:3, :3], trans[:3, 3:4]
    
def integrate_trans(R, t):
    """
    Integrate SE3 transformations from R and t, support torch.Tensor and np.ndarry.
    Input
        - R: [3, 3] or [bs, 3, 3], rotation matrix
        - t: [3, 1] or [bs, 3, 1], translation matrix
    Output
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    """
    if len(R.shape) == 3:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4)[None].repeat(R.shape[0], 1, 1).to(R.device)
        else:
            trans = np.eye(4)[None]
        trans[:, :3, :3] = R
        trans[:, :3, 3:4] = t.view([-1, 3, 1])
    else:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4).to(R.device)
        else:
            trans = np.eye(4)
        trans[:3, :3] = R
        trans[:3, 3:4] = t
    return trans

def concatenate(trans1, trans2):
    """
    Concatenate two SE3 transformations, support torch.Tensor and np.ndarry.
    Input
        - trans1: [4, 4] or [bs, 4, 4], SE3 transformation matrix
        - trans2: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    Output:
        - trans1 @ trans2
    """    
    R1, t1 = decompose_trans(trans1)
    R2, t2 = decompose_trans(trans2)
    R_cat = R1 @ R2
    t_cat = R1 @ t2 + t1
    trans_cat = integrate_trans(R_cat, t_cat)
    return trans_cat
