import json
import sys
sys.path.append('.')
import argparse
import logging
from tqdm import tqdm
from easydict import EasyDict as edict
from evaluate_metric import TransformationLoss, ClassificationLoss
from dataset import KITTILoader
from benchmark_utils import set_seed, icp_refine
from benchmark_utils_predator import *
from utils.timer import Timer
from SC2_PCR import Matcher
set_seed()
from utils.SE3 import *


def eval_KITTI_per_pair(loader, matcher, trans_evaluator, cls_evaluator, config):
    """
    Evaluate our model on KITTI testset.
    """
    num_pair = loader.__len__()
    # 0.success, 1.RE, 2.TE, 3.input inlier number, 4.input inlier ratio,  5. output inlier number 
    # 6. output inlier precision, 7. output inlier recall, 8. output inlier F1 score 9. model_time, 10. data_time 11. scene_ind
    stats = np.zeros([num_pair, 12])

    data_timer, model_timer = Timer(), Timer()
    with torch.no_grad():
        for i in tqdm(range(num_pair)):
            #################################
            # 1. load data
            #################################
            data_timer.tic()
            src_keypts, tgt_keypts, src_features, tgt_features, gt_trans = loader.get_data(i)
            data_time = data_timer.toc()

            #################################
            # 2. match descriptor and compute rigid transformation
            #################################
            model_timer.tic()
            pred_trans, pred_labels, src_keypts_corr, tgt_keypts_corr = matcher.estimator(src_keypts, tgt_keypts,
                                                                                          src_features,
                                                                                          tgt_features)
            model_time = model_timer.toc()

            #################################
            # 3. generate the ground-truth classification result
            #################################
            frag1_warp = transform(src_keypts_corr, gt_trans)
            distance = torch.sum((frag1_warp - tgt_keypts_corr) ** 2, dim=-1) ** 0.5
            gt_labels = (distance < config.inlier_threshold).float()

            #################################
            # 4. evaluate result
            #################################
            loss, recall, Re, Te, rmse = trans_evaluator(pred_trans, gt_trans, src_keypts_corr, tgt_keypts_corr,
                                                         pred_labels)
            class_stats = cls_evaluator(pred_labels, gt_labels)

            # save statistics
            stats[i, 0] = float(recall / 100.0)                      # success
            stats[i, 1] = float(Re)                                  # Re (deg)
            stats[i, 2] = float(Te)                                  # Te (cm)
            stats[i, 3] = int(torch.sum(gt_labels))                  # input inlier number 
            stats[i, 4] = float(torch.mean(gt_labels.float()))       # input inlier ratio
            stats[i, 5] = int(torch.sum(gt_labels[pred_labels > 0])) # output inlier number 
            stats[i, 6] = float(class_stats['precision'])            # output inlier precision 
            stats[i, 7] = float(class_stats['recall'])               # output inlier recall
            stats[i, 8] = float(class_stats['f1'])                   # output inlier f1 score
            stats[i, 9] = model_time
            stats[i, 10] = data_time
            stats[i, 11] = -1

            if recall == 0:
                from benchmark_utils import rot_to_euler
                R_gt, t_gt = gt_trans[0][:3, :3], gt_trans[0][:3, -1]
                euler = rot_to_euler(R_gt.detach().cpu().numpy())

                input_ir = float(torch.mean(gt_labels.float()))
                input_i = int(torch.sum(gt_labels))
                output_i = int(torch.sum(gt_labels[pred_labels > 0]))
                logging.info(f"Pair {i}, GT Rot: {euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}, Trans: {t_gt[0]:.2f}, {t_gt[1]:.2f}, {t_gt[2]:.2f}, RE: {float(Re):.2f}, TE: {float(Te):.2f}")
                logging.info((f"\tInput Inlier Ratio :{input_ir*100:.2f}%(#={input_i}), Output: IP={float(class_stats['precision'])*100:.2f}%(#={output_i}) IR={float(class_stats['recall'])*100:.2f}%"))

    return stats

def eval_KITTI(config):
    loader = KITTILoader(root=config.data_path,
                                 descriptor=config.descriptor,
                                 inlier_threshold=config.inlier_threshold,
                                 num_node=config.num_node,
                                 use_mutual=config.use_mutual,
                                 )
    matcher = Matcher(inlier_threshold=config.inlier_threshold,
                      num_node=config.num_node,
                      use_mutual=config.use_mutual,
                      d_thre=config.d_thre,
                      num_iterations=config.num_iterations,
                      ratio=config.ratio,
                      nms_radius=config.nms_radius,
                      max_points=config.max_points,
                      k1=config.k1,
                      k2=config.k2, )
    trans_evaluator = TransformationLoss(re_thre=config.re_thre, te_thre=config.te_thre)
    cls_evaluator = ClassificationLoss()

    stats = eval_KITTI_per_pair(loader, matcher, trans_evaluator, cls_evaluator, config)
    logging.info(f"Max memory allicated: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}GB")

    # pair level average 
    allpair_stats = stats
    allpair_average = allpair_stats.mean(0)
    correct_pair_average = allpair_stats[allpair_stats[:, 0] == 1].mean(0)
    logging.info(f"*"*40)
    logging.info(f"All {allpair_stats.shape[0]} pairs, Mean Success Rate={allpair_average[0]*100:.2f}%, Mean Re={correct_pair_average[1]:.2f}, Mean Te={correct_pair_average[2]:.2f}")
    logging.info(f"\tInput:  Mean Inlier Num={allpair_average[3]:.2f}(ratio={allpair_average[4]*100:.2f}%)")
    logging.info(f"\tOutput: Mean Inlier Num={allpair_average[5]:.2f}(precision={allpair_average[6]*100:.2f}%, recall={allpair_average[7]*100:.2f}%, f1={allpair_average[8]*100:.2f}%)")
    logging.info(f"\tMean model time: {allpair_average[9]:.2f}s, Mean data time: {allpair_average[10]:.2f}s")

    return allpair_stats

if __name__ == '__main__':
    from config import str2bool
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='', type=str, help='snapshot dir')
    parser.add_argument('--solver', default='SVD', type=str, choices=['SVD', 'RANSAC'])
    parser.add_argument('--use_icp', default=False, type=str2bool)
    parser.add_argument('--save_npz', default=False, type=str2bool)
    args = parser.parse_args()

    config_path = args.config_path
    config = json.load(open(config_path, 'r'))
    config = edict(config)

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_Devices
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    log_filename = f'logs/KITTI-{config.descriptor}.log'
    logging.basicConfig(level=logging.INFO,
                        filename=log_filename,
                        filemode='a',
                        format="")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    stats = eval_KITTI(config)

    if args.save_npz:
        save_path = log_filename.replace('.log', '.npy')
        np.save(save_path, stats)
        print(f"Save the stats in {save_path}")
