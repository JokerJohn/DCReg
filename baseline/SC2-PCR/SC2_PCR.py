import torch
from common import knn, rigid_transform_3d
from utils.SE3 import transform
import numpy as np



class Matcher():
    def __init__(self,
                 inlier_threshold=0.10,
                 num_node='all',
                 use_mutual=True,
                 d_thre=0.1,
                 num_iterations=10,
                 ratio=0.2,
                 nms_radius=0.1,
                 max_points=8000,
                 k1=30,
                 k2=20,
                 select_scene=None,
                 ):
        self.inlier_threshold = inlier_threshold
        self.num_node = num_node
        self.use_mutual = use_mutual
        self.d_thre = d_thre
        self.num_iterations = num_iterations  # maximum iteration of power iteration algorithm
        self.ratio = ratio # the maximum ratio of seeds.
        self.max_points = max_points
        self.nms_radius = nms_radius
        self.k1 = k1
        self.k2 = k2

    def pick_seeds(self, dists, scores, R, max_num):
        """
        Select seeding points using Non Maximum Suppression. (here we only support bs=1)
        Input:
            - dists:       [bs, num_corr, num_corr] src keypoints distance matrix
            - scores:      [bs, num_corr]     initial confidence of each correspondence
            - R:           float              radius of nms
            - max_num:     int                maximum number of returned seeds
        Output:
            - picked_seeds: [bs, num_seeds]   the index to the seeding correspondences
        """
        assert scores.shape[0] == 1

        # parallel Non Maximum Suppression (more efficient)
        score_relation = scores.T >= scores  # [num_corr, num_corr], save the relation of leading_eig
        # score_relation[dists[0] >= R] = 1  # mask out the non-neighborhood node
        score_relation = score_relation.bool() | (dists[0] >= R).bool()
        is_local_max = score_relation.min(-1)[0].float()

        score_local_max = scores * is_local_max
        sorted_score = torch.argsort(score_local_max, dim=1, descending=True)

        # max_num = scores.shape[1]

        return_idx = sorted_score[:, 0: max_num].detach()

        return return_idx

    def cal_seed_trans(self, seeds, SC2_measure, src_keypts, tgt_keypts):
        """
        Calculate the transformation for each seeding correspondences.
        Input:
            - seeds:         [bs, num_seeds]              the index to the seeding correspondence
            - SC2_measure: [bs, num_corr, num_channels]
            - src_keypts:    [bs, num_corr, 3]
            - tgt_keypts:    [bs, num_corr, 3]
        Output: leading eigenvector
            - final_trans:       [bs, 4, 4]             best transformation matrix (after post refinement) for each batch.
        """
        bs, num_corr, num_channels = SC2_measure.shape[0], SC2_measure.shape[1], SC2_measure.shape[2]
        k1 = self.k1
        k2 = self.k2
        
        if k1 > num_channels:
            k1 = 4
            k2 = 4

        #################################
        # The first stage consensus set sampling
        # Finding the k1 nearest neighbors around each seed
        #################################
        sorted_score = torch.argsort(SC2_measure, dim=2, descending=True)
        knn_idx = sorted_score[:, :, 0: k1]
        sorted_value, _ = torch.sort(SC2_measure, dim=2, descending=True)
        idx_tmp = knn_idx.contiguous().view([bs, -1])
        idx_tmp = idx_tmp[:, :, None]
        idx_tmp = idx_tmp.expand(-1, -1, 3)

        #################################
        # construct the local SC2 measure of each consensus subset obtained in the first stage.
        #################################
        src_knn = src_keypts.gather(dim=1, index=idx_tmp).view([bs, -1, k1, 3])  # [bs, num_seeds, k, 3]
        tgt_knn = tgt_keypts.gather(dim=1, index=idx_tmp).view([bs, -1, k1, 3])
        src_dist = ((src_knn[:, :, :, None, :] - src_knn[:, :, None, :, :]) ** 2).sum(-1) ** 0.5
        tgt_dist = ((tgt_knn[:, :, :, None, :] - tgt_knn[:, :, None, :, :]) ** 2).sum(-1) ** 0.5
        cross_dist = torch.abs(src_dist - tgt_dist)
        local_hard_SC_measure = (cross_dist < self.d_thre).float()
        local_SC2_measure = torch.matmul(local_hard_SC_measure[:, :, :1, :], local_hard_SC_measure)

        #################################
        # perform second stage consensus set sampling
        #################################
        sorted_score = torch.argsort(local_SC2_measure, dim=3, descending=True)
        knn_idx_fine = sorted_score[:, :, :, 0: k2]

        #################################
        # construct the soft SC2 matrix of the consensus set
        #################################
        num = knn_idx_fine.shape[1]
        knn_idx_fine = knn_idx_fine.contiguous().view([bs, num, -1])[:, :, :, None]
        knn_idx_fine = knn_idx_fine.expand(-1, -1, -1, 3)
        src_knn_fine = src_knn.gather(dim=2, index=knn_idx_fine).view([bs, -1, k2, 3])  # [bs, num_seeds, k, 3]
        tgt_knn_fine = tgt_knn.gather(dim=2, index=knn_idx_fine).view([bs, -1, k2, 3])

        src_dist = ((src_knn_fine[:, :, :, None, :] - src_knn_fine[:, :, None, :, :]) ** 2).sum(-1) ** 0.5
        tgt_dist = ((tgt_knn_fine[:, :, :, None, :] - tgt_knn_fine[:, :, None, :, :]) ** 2).sum(-1) ** 0.5
        cross_dist = torch.abs(src_dist - tgt_dist)
        local_hard_measure = (cross_dist < self.d_thre * 2).float()
        local_SC2_measure = torch.matmul(local_hard_measure, local_hard_measure) / k2
        local_SC_measure = torch.clamp(1 - cross_dist ** 2 / self.d_thre ** 2, min=0)
        # local_SC2_measure = local_SC_measure * local_SC2_measure
        local_SC2_measure = local_SC_measure
        local_SC2_measure = local_SC2_measure.view([-1, k2, k2])


        #################################
        # Power iteratation to get the inlier probability
        #################################
        local_SC2_measure[:, torch.arange(local_SC2_measure.shape[1]), torch.arange(local_SC2_measure.shape[1])] = 0
        total_weight = self.cal_leading_eigenvector(local_SC2_measure, method='power')
        total_weight = total_weight.view([bs, -1, k2])
        total_weight = total_weight / (torch.sum(total_weight, dim=-1, keepdim=True) + 1e-6)

        #################################
        # calculate the transformation by weighted least-squares for each subsets in parallel
        #################################
        total_weight = total_weight.view([-1, k2])
        src_knn = src_knn_fine
        tgt_knn = tgt_knn_fine
        src_knn, tgt_knn = src_knn.view([-1, k2, 3]), tgt_knn.view([-1, k2, 3])

        #################################
        # compute the rigid transformation for each seed by the weighted SVD
        #################################
        seedwise_trans = rigid_transform_3d(src_knn, tgt_knn, total_weight)
        seedwise_trans = seedwise_trans.view([bs, -1, 4, 4])

        #################################
        # calculate the inlier number for each hypothesis, and find the best transformation for each point cloud pair
        #################################
        pred_position = torch.einsum('bsnm,bmk->bsnk', seedwise_trans[:, :, :3, :3],
                                     src_keypts.permute(0, 2, 1)) + seedwise_trans[:, :, :3,
                                                                    3:4]  # [bs, num_seeds, num_corr, 3]
        #################################
        # calculate the inlier number for each hypothesis, and find the best transformation for each point cloud pair
        #################################
        pred_position = pred_position.permute(0, 1, 3, 2)
        L2_dis = torch.norm(pred_position - tgt_keypts[:, None, :, :], dim=-1)  # [bs, num_seeds, num_corr]
        seedwise_fitness = torch.sum((L2_dis < self.inlier_threshold).float(), dim=-1)  # [bs, num_seeds]
        batch_best_guess = seedwise_fitness.argmax(dim=1)
        best_guess_ratio = seedwise_fitness[0, batch_best_guess]
        final_trans = seedwise_trans.gather(dim=1,index=batch_best_guess[:, None, None, None].expand(-1, -1, 4, 4)).squeeze(1)

        return final_trans

    def cal_leading_eigenvector(self, M, method='power'):
        """
        Calculate the leading eigenvector using power iteration algorithm or torch.symeig
        Input:
            - M:      [bs, num_corr, num_corr] the compatibility matrix
            - method: select different method for calculating the learding eigenvector.
        Output:
            - solution: [bs, num_corr] leading eigenvector
        """
        if method == 'power':
            # power iteration algorithm
            leading_eig = torch.ones_like(M[:, :, 0:1])
            leading_eig_last = leading_eig
            for i in range(self.num_iterations):
                leading_eig = torch.bmm(M, leading_eig)
                leading_eig = leading_eig / (torch.norm(leading_eig, dim=1, keepdim=True) + 1e-6)
                if torch.allclose(leading_eig, leading_eig_last):
                    break
                leading_eig_last = leading_eig
            leading_eig = leading_eig.squeeze(-1)
            return leading_eig
        elif method == 'eig':  # cause NaN during back-prop
            e, v = torch.symeig(M, eigenvectors=True)
            leading_eig = v[:, :, -1]
            return leading_eig
        else:
            exit(-1)

    def cal_confidence(self, M, leading_eig, method='eig_value'):
        """
        Calculate the confidence of the spectral matching solution based on spectral analysis.
        Input:
            - M:          [bs, num_corr, num_corr] the compatibility matrix
            - leading_eig [bs, num_corr]           the leading eigenvector of matrix M
        Output:
            - confidence
        """
        if method == 'eig_value':
            # max eigenvalue as the confidence (Rayleigh quotient)
            max_eig_value = (leading_eig[:, None, :] @ M @ leading_eig[:, :, None]) / (
                        leading_eig[:, None, :] @ leading_eig[:, :, None])
            confidence = max_eig_value.squeeze(-1)
            return confidence
        elif method == 'eig_value_ratio':
            # max eigenvalue / second max eigenvalue as the confidence
            max_eig_value = (leading_eig[:, None, :] @ M @ leading_eig[:, :, None]) / (
                        leading_eig[:, None, :] @ leading_eig[:, :, None])
            # compute the second largest eigen-value
            B = M - max_eig_value * leading_eig[:, :, None] @ leading_eig[:, None, :]
            solution = torch.ones_like(B[:, :, 0:1])
            for i in range(self.num_iterations):
                solution = torch.bmm(B, solution)
                solution = solution / (torch.norm(solution, dim=1, keepdim=True) + 1e-6)
            solution = solution.squeeze(-1)
            second_eig = solution
            second_eig_value = (second_eig[:, None, :] @ B @ second_eig[:, :, None]) / (
                        second_eig[:, None, :] @ second_eig[:, :, None])
            confidence = max_eig_value / second_eig_value
            return confidence
        elif method == 'xMx':
            # max xMx as the confidence (x is the binary solution)
            # rank = torch.argsort(leading_eig, dim=1, descending=True)[:, 0:int(M.shape[1]*self.ratio)]
            # binary_sol = torch.zeros_like(leading_eig)
            # binary_sol[0, rank[0]] = 1
            confidence = leading_eig[:, None, :] @ M @ leading_eig[:, :, None]
            confidence = confidence.squeeze(-1) / M.shape[1]
            return confidence

    def post_refinement(self, initial_trans, src_keypts, tgt_keypts, it_num, weights=None):
        """
        Perform post refinement using the initial transformation matrix, only adopted during testing.
        Input
            - initial_trans: [bs, 4, 4]
            - src_keypts:    [bs, num_corr, 3]
            - tgt_keypts:    [bs, num_corr, 3]
            - weights:       [bs, num_corr]
        Output:
            - final_trans:   [bs, 4, 4]
        """
        assert initial_trans.shape[0] == 1
        inlier_threshold = 1.2

        # inlier_threshold_list = [self.inlier_threshold] * it_num

        if self.inlier_threshold == 0.10:  # for 3DMatch
            inlier_threshold_list = [0.10] * it_num
        else:  # for KITTI
            inlier_threshold_list = [1.2] * it_num

        previous_inlier_num = 0
        for inlier_threshold in inlier_threshold_list:
            warped_src_keypts = transform(src_keypts, initial_trans)

            L2_dis = torch.norm(warped_src_keypts - tgt_keypts, dim=-1)
            pred_inlier = (L2_dis < inlier_threshold)[0]  # assume bs = 1
            inlier_num = torch.sum(pred_inlier)
            if abs(int(inlier_num - previous_inlier_num)) < 1:
                break
            else:
                previous_inlier_num = inlier_num
            initial_trans = rigid_transform_3d(
                A=src_keypts[:, pred_inlier, :],
                B=tgt_keypts[:, pred_inlier, :],
                ## https://link.springer.com/article/10.1007/s10589-014-9643-2
                # weights=None,
                weights=1 / (1 + (L2_dis / inlier_threshold) ** 2)[:, pred_inlier],
                # weights=((1-L2_dis/inlier_threshold)**2)[:, pred_inlier]
            )
        return initial_trans

    def match_pair(self, src_keypts, tgt_keypts, src_features, tgt_features):
        N_src = src_features.shape[1]
        N_tgt = tgt_features.shape[1]
        # use all point or sample points.
        if self.num_node == 'all':
            src_sel_ind = np.arange(N_src)
            tgt_sel_ind = np.arange(N_tgt)
        else:
            src_sel_ind = np.random.choice(N_src, self.num_node)
            tgt_sel_ind = np.random.choice(N_tgt, self.num_node)
        src_desc = src_features[:, src_sel_ind, :]
        tgt_desc = tgt_features[:, tgt_sel_ind, :]
        src_keypts = src_keypts[:, src_sel_ind, :]
        tgt_keypts = tgt_keypts[:, tgt_sel_ind, :]

        # match points in feature space.
        distance = torch.sqrt(2 - 2 * (src_desc[0] @ tgt_desc[0].T) + 1e-6)
        distance = distance.unsqueeze(0)
        source_idx = torch.argmin(distance[0], dim=1)
        idx_range = torch.arange(source_idx.shape[0], device=source_idx.device)[:, None]
        corr = torch.cat([idx_range, source_idx[:, None]], dim=-1)

        # generate correspondences
        src_keypts_corr = src_keypts[:, corr[:, 0]]
        tgt_keypts_corr = tgt_keypts[:, corr[:, 1]]

        return src_keypts_corr, tgt_keypts_corr

    def SC2_PCR(self, src_keypts, tgt_keypts):
        """
        Input:
            - src_keypts: [bs, num_corr, 3]
            - tgt_keypts: [bs, num_corr, 3]
        Output:
            - pred_trans:   [bs, 4, 4], the predicted transformation matrix.
            - pred_labels:  [bs, num_corr], the predicted inlier/outlier label (0,1), for classification loss calculation.
        """
        bs, num_corr = src_keypts.shape[0], tgt_keypts.shape[1]

        #################################
        # downsample points
        #################################
        if num_corr > self.max_points:
            src_keypts = src_keypts[:, :self.max_points, :]
            tgt_keypts = tgt_keypts[:, :self.max_points, :]
            num_corr = self.max_points

        #################################
        # compute cross dist
        #################################
        src_dist = torch.norm((src_keypts[:, :, None, :] - src_keypts[:, None, :, :]), dim=-1)
        target_dist = torch.norm((tgt_keypts[:, :, None, :] - tgt_keypts[:, None, :, :]), dim=-1)
        cross_dist = torch.abs(src_dist - target_dist)

        #################################
        # compute first order measure
        #################################
        SC_dist_thre = self.d_thre
        SC_measure = torch.clamp(1.0 - cross_dist ** 2 / SC_dist_thre ** 2, min=0)
        hard_SC_measure = (cross_dist < SC_dist_thre).float()

        #################################
        # select reliable seed correspondences
        #################################
        confidence = self.cal_leading_eigenvector(SC_measure, method='power')
        seeds = self.pick_seeds(src_dist, confidence, R=self.nms_radius, max_num=int(num_corr * self.ratio))

        #################################
        # compute second order measure
        #################################
        SC2_dist_thre = self.d_thre / 2
        hard_SC_measure_tight = (cross_dist < SC2_dist_thre).float()
        seed_hard_SC_measure = hard_SC_measure.gather(dim=1,
                                index=seeds[:, :, None].expand(-1, -1, num_corr))
        seed_hard_SC_measure_tight = hard_SC_measure_tight.gather(dim=1,
                                index=seeds[:, :, None].expand(-1, -1, num_corr))
        SC2_measure = torch.matmul(seed_hard_SC_measure_tight, hard_SC_measure_tight) * seed_hard_SC_measure

        #################################
        # compute the seed-wise transformations and select the best one
        #################################
        final_trans = self.cal_seed_trans(seeds, SC2_measure, src_keypts, tgt_keypts)

        #################################
        # refine the result by recomputing the transformation over the whole set
        #################################
        final_trans = self.post_refinement(final_trans, src_keypts, tgt_keypts, 20)

        return final_trans

    def estimator(self, src_keypts, tgt_keypts, src_features, tgt_features):
        """
        Input:
            - src_keypts: [bs, num_corr, 3]
            - tgt_keypts: [bs, num_corr, 3]
            - src_features: [bs, num_corr, C]
            - tgt_features: [bs, num_corr, C]
        Output:
            - pred_trans:   [bs, 4, 4], the predicted transformation matrix
            - pred_trans:   [bs, num_corr], the predicted inlier/outlier label (0,1)
            - src_keypts_corr:  [bs, num_corr, 3], the source points in the matched correspondences
            - tgt_keypts_corr:  [bs, num_corr, 3], the target points in the matched correspondences
        """
        #################################
        # generate coarse correspondences
        #################################
        src_keypts_corr, tgt_keypts_corr = self.match_pair(src_keypts, tgt_keypts, src_features, tgt_features)

        #################################
        # use the proposed SC2-PCR to estimate the rigid transformation
        #################################
        pred_trans = self.SC2_PCR(src_keypts_corr, tgt_keypts_corr)

        frag1_warp = transform(src_keypts_corr, pred_trans)
        distance = torch.sum((frag1_warp - tgt_keypts_corr) ** 2, dim=-1) ** 0.5
        pred_labels = (distance < self.inlier_threshold).float()

        return pred_trans, pred_labels, src_keypts_corr, tgt_keypts_corr
