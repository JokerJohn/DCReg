import json
import sys
sys.path.append('.')
import argparse
import logging
from tqdm import tqdm
from easydict import EasyDict as edict
from evaluate_metric import TransformationLoss, ClassificationLoss
from dataset import ThreeDLoMatchLoader
from benchmark_utils import set_seed, icp_refine
from benchmark_utils_predator import *
from utils.timer import Timer
from SC2_PCR import Matcher
set_seed()
from utils.SE3 import *
from collections import defaultdict

def eval_3DLoMatch_scene(loader, matcher, trans_evaluator, cls_evaluator, scene_ind, config):
    num_pair = loader.__len__()
    final_poses = np.zeros([num_pair, 4, 4])

    # 0.success, 1.RE, 2.TE, 3.input inlier number, 4.input inlier ratio,  5. output inlier number
    # 6. output inlier precision, 7. output inlier recall, 8. output inlier F1 score 9. model_time, 10. data_time 11. scene_ind
    stats = np.zeros([num_pair, 12])

    data_timer, model_timer = Timer(), Timer()
    with torch.no_grad():
        error_pair = []
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
                                                                                          src_features, tgt_features)
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

            #################################
            # record the evaluation results.
            #################################
            # save statistics
            stats[i, 0] = float(recall / 100.0)  # success
            stats[i, 1] = float(Re)  # Re (deg)
            stats[i, 2] = float(Te)  # Te (cm)
            stats[i, 3] = int(torch.sum(gt_labels))                  # input inlier number
            stats[i, 4] = float(torch.mean(gt_labels.float()))       # input inlier ratio
            stats[i, 5] = int(torch.sum(gt_labels[pred_labels > 0])) # output inlier number
            stats[i, 6] = float(class_stats['precision'])            # output inlier precision
            stats[i, 7] = float(class_stats['recall'])               # output inlier recall
            stats[i, 8] = float(class_stats['f1'])                   # output inlier f1 score
            stats[i, 9] = model_time
            stats[i, 10] = data_time
            stats[i, 11] = scene_ind
            final_poses[i] = pred_trans[0].detach().cpu().numpy()
        print(error_pair)

    return stats, final_poses


def eval_3DLoMatch(config):
    loader = ThreeDLoMatchLoader(root=config.data_path,
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

    allpair_stats, allpair_poses = eval_3DLoMatch_scene(loader, matcher, trans_evaluator, cls_evaluator, 0, config)

    allpair_average = allpair_stats.mean(0)
    allpair_status_ndarray = np.array(allpair_stats, dtype=float)

    benchmark_predator(allpair_poses, gt_folder='benchmarks/3DLoMatch')
    
    # benchmarking using the registration recall defined in DGR 
    allpair_average = allpair_stats.mean(0)
    correct_pair_average = allpair_stats[allpair_stats[:, 0] == 1].mean(0)
    logging.info(f"*" * 40)
    logging.info(f"All {allpair_stats.shape[0]} pairs, Mean Reg Recall={allpair_average[0] * 100:.2f}%, Mean Re={correct_pair_average[1]:.2f}, Mean Te={correct_pair_average[2]:.2f}")
    logging.info(f"\tInput:  Mean Inlier Num={allpair_average[3]:.2f}(ratio={allpair_average[4] * 100:.2f}%)")
    logging.info(f"\tOutput: Mean Inlier Num={allpair_average[5]:.2f}(precision={allpair_average[6] * 100:.2f}%, recall={allpair_average[7] * 100:.2f}%, f1={allpair_average[8] * 100:.2f}%)")
    logging.info(f"\tMean model time: {allpair_average[9]:.2f}s, Mean data time: {allpair_average[10]:.2f}s")

    # all_stats_npy = np.concatenate([v for k, v in all_stats.items()], axis=0)

    return allpair_stats


def benchmark_predator(pred_poses, gt_folder):
    scenes = sorted(os.listdir(gt_folder))
    scene_names = [os.path.join(gt_folder,ele) for ele in scenes]

    re_per_scene = defaultdict(list)
    te_per_scene = defaultdict(list)
    re_all, te_all, precision, recall = [], [], [], []
    n_valids= []

    short_names=['Kitchen','Home 1','Home 2','Hotel 1','Hotel 2','Hotel 3','Study','MIT Lab']
    logging.info(("Scene\t¦ prec.\t¦ rec.\t¦ re\t¦ te\t¦ samples\t¦"))
    
    start_ind = 0
    for idx,scene in enumerate(scene_names):
        # ground truth info
        gt_pairs, gt_traj = read_trajectory(os.path.join(scene, "gt.log"))
        n_valid=0
        for ele in gt_pairs:
            diff=abs(int(ele[0])-int(ele[1]))
            n_valid+=diff>1
        n_valids.append(n_valid)

        n_fragments, gt_traj_cov = read_trajectory_info(os.path.join(scene,"gt.info"))

        # estimated info
        # est_pairs, est_traj = read_trajectory(os.path.join(est_folder,scenes[idx],'est.log'))
        est_traj = pred_poses[start_ind:start_ind + len(gt_pairs)]
        start_ind = start_ind + len(gt_pairs)

        temp_precision, temp_recall,c_flag = evaluate_registration(n_fragments, est_traj, gt_pairs, gt_pairs, gt_traj, gt_traj_cov)
        
        # Filter out the estimated rotation matrices
        ext_gt_traj = extract_corresponding_trajectors(gt_pairs,gt_pairs, gt_traj)

        re = rotation_error(torch.from_numpy(ext_gt_traj[:,0:3,0:3]), torch.from_numpy(est_traj[:,0:3,0:3])).cpu().numpy()[np.array(c_flag)==0]
        te = translation_error(torch.from_numpy(ext_gt_traj[:,0:3,3:4]), torch.from_numpy(est_traj[:,0:3,3:4])).cpu().numpy()[np.array(c_flag)==0]

        re_per_scene['mean'].append(np.mean(re))
        re_per_scene['median'].append(np.median(re))
        re_per_scene['min'].append(np.min(re))
        re_per_scene['max'].append(np.max(re))
        
        te_per_scene['mean'].append(np.mean(te))
        te_per_scene['median'].append(np.median(te))
        te_per_scene['min'].append(np.min(te))
        te_per_scene['max'].append(np.max(te))


        re_all.extend(re.reshape(-1).tolist())
        te_all.extend(te.reshape(-1).tolist())

        precision.append(temp_precision)
        recall.append(temp_recall)

        logging.info("{}\t¦ {:.3f}\t¦ {:.3f}\t¦ {:.3f}\t¦ {:.3f}\t¦ {:3d}¦".format(short_names[idx], temp_precision, temp_recall, np.median(re), np.median(te), n_valid))
        # np.save(f'{est_folder}/{scenes[idx]}/flag.npy',c_flag)
    
    weighted_precision = (np.array(n_valids) * np.array(precision)).sum() / np.sum(n_valids)

    logging.info("Mean precision: {:.3f}: +- {:.3f}".format(np.mean(precision),np.std(precision)))
    logging.info("Weighted precision: {:.3f}".format(weighted_precision))

    logging.info("Mean median RRE: {:.3f}: +- {:.3f}".format(np.mean(re_per_scene['median']), np.std(re_per_scene['median'])))
    logging.info("Mean median RTE: {:.3F}: +- {:.3f}".format(np.mean(te_per_scene['median']),np.std(te_per_scene['median'])))
    

if __name__ == '__main__':
    from config import str2bool

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='', type=str, help='snapshot dir')
    parser.add_argument('--solver', default='SVD', type=str, choices=['SVD', 'RANSAC'])
    parser.add_argument('--use_icp', default=False, type=str2bool)
    parser.add_argument('--save_npy', default=False, type=str2bool)
    args = parser.parse_args()

    config_path = args.config_path
    config = json.load(open(config_path, 'r'))
    config = edict(config)

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_Devices
    if not os.path.exists("./logs"):
        os.makedirs("./logs")

    log_filename = f'logs/3DLoMatch-{config.descriptor}.log'
    logging.basicConfig(level=logging.INFO,
                        filename=log_filename,
                        filemode='a',
                        format="")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # evaluate on the test set
    stats = eval_3DLoMatch(config)
    if args.save_npy:
        save_path = log_filename.replace('.log', '.npy')
        np.save(save_path, stats)
        print(f"Save the stats in {save_path}")
