import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score
from utils.SE3 import *
import warnings

warnings.filterwarnings('ignore')


class TransformationLoss(nn.Module):
    def __init__(self, re_thre=15, te_thre=30):
        super(TransformationLoss, self).__init__()
        self.re_thre = re_thre  # rotation error threshold (deg)
        self.te_thre = te_thre  # translation error threshold (cm)

    # def forward(self, trans, gt_trans, src_keypts, tgt_keypts, probs):
    def forward(self, trans, gt_trans, src_keypts, tgt_keypts, probs):
        """
        Transformation Loss
        Inputs:
            - trans:      [bs, 4, 4] SE3 transformation matrices
            - gt_trans:   [bs, 4, 4] ground truth SE3 transformation matrices
            - src_keypts: [bs, num_corr, 3]
            - tgt_keypts: [bs, num_corr, 3]
            - probs:     [bs, num_corr] predicted inlier probability
        Outputs:
            - loss     transformation loss 
            - recall   registration recall (re < re_thre & te < te_thre)
            - RE       rotation error 
            - TE       translation error
            - RMSE     RMSE under the predicted transformation
        """
        bs = trans.shape[0]
        R, t = decompose_trans(trans)
        gt_R, gt_t = decompose_trans(gt_trans)

        recall = 0
        RE = torch.tensor(0.0).to(trans.device)
        TE = torch.tensor(0.0).to(trans.device)
        RMSE = torch.tensor(0.0).to(trans.device)
        loss = torch.tensor(0.0).to(trans.device)
        for i in range(bs):
            re = torch.acos(torch.clamp((torch.trace(R[i].T @ gt_R[i]) - 1) / 2.0, min=-1, max=1))
            te = torch.sqrt(torch.sum((t[i] - gt_t[i]) ** 2))
            warp_src_keypts = transform(src_keypts[i], trans[i])
            rmse = torch.norm(warp_src_keypts - tgt_keypts, dim=-1).mean()
            re = re * 180 / np.pi
            te = te * 100
            if te < self.te_thre and re < self.re_thre:
                recall += 1
            RE += re
            TE += te
            RMSE += rmse

            pred_inliers = torch.where(probs[i] > 0)[0]
            if len(pred_inliers) < 1:
                loss += torch.tensor(0.0).to(trans.device)
            else:
                warp_src_keypts = transform(src_keypts[i], trans[i])
                loss +=  ((warp_src_keypts - tgt_keypts)**2).sum(-1).mean()

        return loss / bs, recall * 100.0 / bs, RE / bs, TE / bs, RMSE / bs

class ClassificationLoss(nn.Module):
    def __init__(self, balanced=True):
        super(ClassificationLoss, self).__init__()
        self.balanced = balanced

    def forward(self, pred, gt, weight=None):
        """ 
        Classification Loss for the inlier confidence
        Inputs:
            - pred: [bs, num_corr] predicted logits/labels for the putative correspondences
            - gt:   [bs, num_corr] ground truth labels
        Outputs:(dict)
            - loss          (weighted) BCE loss for inlier confidence 
            - precision:    inlier precision (# kept inliers / # kepts matches)
            - recall:       inlier recall (# kept inliers / # all inliers)
            - f1:           (precision * recall * 2) / (precision + recall)
            - logits_true:  average logits for inliers
            - logits_false: average logits for outliers
        """
        num_pos = torch.relu(torch.sum(gt) - 1) + 1
        num_neg = torch.relu(torch.sum(1 - gt) - 1) + 1
        if weight is not None:
            loss = nn.BCEWithLogitsLoss(reduction='none')(pred, gt.float())
            loss = torch.mean(loss * weight)
        elif self.balanced is False:
            loss = nn.BCEWithLogitsLoss(reduction='mean')(pred, gt.float())
        else:
            loss = nn.BCEWithLogitsLoss(pos_weight=num_neg * 1.0 / num_pos, reduction='mean')(pred, gt.float())

        # compute precision, recall, f1
        pred_labels = pred > 0
        gt, pred_labels, pred = gt.detach().cpu().numpy(), pred_labels.detach().cpu().numpy(), pred.detach().cpu().numpy()
        precision = precision_score(gt[0], pred_labels[0])
        recall = recall_score(gt[0], pred_labels[0])
        f1 = f1_score(gt[0], pred_labels[0])
        mean_logit_true = np.sum(pred * gt) / max(1, np.sum(gt))
        mean_logit_false = np.sum(pred * (1 - gt)) / max(1, np.sum(1 - gt))

        eval_stats = {
            "loss": loss,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "logit_true": float(mean_logit_true),
            "logit_false": float(mean_logit_false)
        }
        return eval_stats




