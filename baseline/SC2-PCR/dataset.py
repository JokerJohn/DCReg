import os
import pickle
import torch.utils.data as data
from utils.SE3 import *
import  torch


class ThreeDLoader(data.Dataset):
    def __init__(self,
                 root,
                 descriptor='fcgf',
                 inlier_threshold=0.10,
                 num_node=5000,
                 downsample=0.03,
                 use_mutual=False,
                 select_scene=None,
                 ):
        self.root = root
        self.descriptor = descriptor
        assert descriptor in ['fcgf', 'fpfh']
        self.inlier_threshold = inlier_threshold
        self.num_node = num_node
        self.use_mutual = use_mutual
        self.sigma_spat = 0.1
        self.num_iterations = 10  # maximum iteration of power iteration algorithm
        self.ratio = 0.1  # the maximum ratio of seeds.
        self.nms_radius = 0.1

        # containers
        self.gt_trans = {}

        self.scene_list = [
            '7-scenes-redkitchen',
            'sun3d-home_at-home_at_scan1_2013_jan_1',
            'sun3d-home_md-home_md_scan9_2012_sep_30',
            'sun3d-hotel_uc-scan3',
            'sun3d-hotel_umd-maryland_hotel1',
            'sun3d-hotel_umd-maryland_hotel3',
            'sun3d-mit_76_studyroom-76-1studyroom2',
            'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
        ]
        if select_scene in self.scene_list:
            self.scene_list = [select_scene]

        # load ground truth transformation
        for scene in self.scene_list:
            scene_path = f'{self.root}/fragments/{scene}'
            gt_path = f'{self.root}/gt_result/{scene}-evaluation'
            for k, v in self.__loadlog__(gt_path).items():
                self.gt_trans[f'{scene}@{k}'] = v

    def get_data(self, index):
        key = list(self.gt_trans.keys())[index]
        scene = key.split('@')[0]
        src_id = key.split('@')[1].split('_')[0]
        tgt_id = key.split('@')[1].split('_')[1]

        # load point coordinates and pre-computed per-point local descriptors
        if self.descriptor == 'fcgf':
            src_data = np.load(f"{self.root}/fragments/{scene}/cloud_bin_{src_id}_fcgf.npz")
            tgt_data = np.load(f"{self.root}/fragments/{scene}/cloud_bin_{tgt_id}_fcgf.npz")
            src_keypts = src_data['xyz']
            tgt_keypts = tgt_data['xyz']
            src_features = src_data['feature']
            tgt_features = tgt_data['feature']
        elif self.descriptor == 'fpfh':
            src_data = np.load(f"{self.root}/fragments/{scene}/cloud_bin_{src_id}_fpfh.npz")
            tgt_data = np.load(f"{self.root}/fragments/{scene}/cloud_bin_{tgt_id}_fpfh.npz")
            src_keypts = src_data['xyz']
            tgt_keypts = tgt_data['xyz']
            src_features = src_data['feature']
            tgt_features = tgt_data['feature']
            src_features = src_features / (np.linalg.norm(src_features, axis=1, keepdims=True) + 1e-6)
            tgt_features = tgt_features / (np.linalg.norm(tgt_features, axis=1, keepdims=True) + 1e-6)

        # compute ground truth transformation
        gt_trans = np.linalg.inv(self.gt_trans[key])  # the given ground truth trans is target-> source

        return torch.from_numpy(src_keypts.astype(np.float32)).cuda()[None], \
            torch.from_numpy(tgt_keypts.astype(np.float32)).cuda()[None], \
            torch.from_numpy(src_features.astype(np.float32)).cuda()[None], \
            torch.from_numpy(tgt_features.astype(np.float32)).cuda()[None], \
            torch.from_numpy(gt_trans.astype(np.float32)).cuda()[None], \


    def __len__(self):
        return self.gt_trans.keys().__len__()

    def __loadlog__(self, gtpath):
        with open(os.path.join(gtpath, 'gt.log')) as f:
            content = f.readlines()
        result = {}
        i = 0
        while i < len(content):
            line = content[i].replace("\n", "").split("\t")[0:3]
            trans = np.zeros([4, 4])
            trans[0] = np.fromstring(content[i+1], dtype=float, sep=' \t')
            trans[1] = np.fromstring(content[i+2], dtype=float, sep=' \t')
            trans[2] = np.fromstring(content[i+3], dtype=float, sep=' \t')
            trans[3] = np.fromstring(content[i+4], dtype=float, sep=' \t')
            i = i + 5
            result[f'{int(line[0])}_{int(line[1])}'] = trans
        return result

class ThreeDLoMatchLoader(data.Dataset):
    def __init__(self,
            root,
            descriptor='fcgf',
            inlier_threshold=0.10,
            num_node=5000,
            use_mutual=True,
            downsample=0.03,
            ):
        self.root = root
        self.descriptor = descriptor
        assert descriptor in ['fcgf', 'fpfh', 'predator']
        self.inlier_threshold = inlier_threshold
        self.num_node = num_node
        self.use_mutual = use_mutual
        self.downsample = downsample

        with open('3DLoMatch.pkl', 'rb') as f:
            self.infos = pickle.load(f)

    def get_data(self, index):

        gt_trans = integrate_trans(self.infos['rot'][index], self.infos['trans'][index])
        scene = self.infos['src'][index].split('/')[1]
        src_id = self.infos['src'][index].split('/')[-1].split('_')[-1].replace('.pth', '')
        tgt_id = self.infos['tgt'][index].split('/')[-1].split('_')[-1].replace('.pth', '')

        # load point coordinates and pre-computed per-point local descriptors
        if self.descriptor == 'fcgf':
            src_data = np.load(f"{self.root}/fragments/{scene}/cloud_bin_{src_id}_fcgf.npz")
            tgt_data = np.load(f"{self.root}/fragments/{scene}/cloud_bin_{tgt_id}_fcgf.npz")
            src_keypts = src_data['xyz']
            tgt_keypts = tgt_data['xyz']
            src_features = src_data['feature']
            tgt_features = tgt_data['feature']

            src_keypts = torch.from_numpy(src_keypts.astype(np.float32)).cuda()
            tgt_keypts = torch.from_numpy(tgt_keypts.astype(np.float32)).cuda()
            src_features = torch.from_numpy(src_features.astype(np.float32)).cuda()
            tgt_features = torch.from_numpy(tgt_features.astype(np.float32)).cuda()
            gt_trans = torch.from_numpy(gt_trans.astype(np.float32)).cuda()

        elif self.descriptor == 'fpfh':
            src_data = np.load(f"{self.root}/fragments/{scene}/cloud_bin_{src_id}_fpfh.npz")
            tgt_data = np.load(f"{self.root}/fragments/{scene}/cloud_bin_{tgt_id}_fpfh.npz")
            src_keypts = src_data['xyz']
            tgt_keypts = tgt_data['xyz']
            src_features = src_data['feature']
            tgt_features = tgt_data['feature']
            src_features = src_features / (np.linalg.norm(src_features, axis=1, keepdims=True) + 1e-6)
            tgt_features = tgt_features / (np.linalg.norm(tgt_features, axis=1, keepdims=True) + 1e-6)

            src_keypts = torch.from_numpy(src_keypts.astype(np.float32)).cuda()
            tgt_keypts = torch.from_numpy(tgt_keypts.astype(np.float32)).cuda()
            src_features = torch.from_numpy(src_features.astype(np.float32)).cuda()
            tgt_features = torch.from_numpy(tgt_features.astype(np.float32)).cuda()
            gt_trans = torch.from_numpy(gt_trans.astype(np.float32)).cuda()
        elif self.descriptor == "predator":
            data_dict = torch.load(
                f'{self.root}/{index}.pth')
            len_src = data_dict['len_src']
            src_keypts = data_dict['pcd'][:len_src, :].cuda()
            tgt_keypts = data_dict['pcd'][len_src:, :].cuda()
            src_features = data_dict['feats'][:len_src].cuda()
            tgt_features = data_dict['feats'][len_src:].cuda()
            saliency, overlap = data_dict['saliency'], data_dict['overlaps']
            src_overlap, src_saliency = overlap[:len_src], saliency[:len_src]
            tgt_overlap, tgt_saliency = overlap[len_src:], saliency[len_src:]
            src_scores = src_overlap * src_saliency
            tgt_scores = tgt_overlap * tgt_saliency
            if (src_keypts.size(0) > self.num_node):
                idx = np.arange(src_keypts.size(0))
                probs = (src_scores / src_scores.sum()).numpy().flatten()
                idx = np.random.choice(idx, size=self.num_node, replace=False, p=probs)
                src_keypts, src_features = src_keypts[idx], src_features[idx]
            if (tgt_keypts.size(0) > self.num_node):
                idx = np.arange(tgt_keypts.size(0))
                probs = (tgt_scores / tgt_scores.sum()).numpy().flatten()
                idx = np.random.choice(idx, size=self.num_node, replace=False, p=probs)
                tgt_keypts, tgt_features = tgt_keypts[idx], tgt_features[idx]
            gt_trans = integrate_trans(data_dict['rot'], data_dict['trans']).cuda()

        return src_keypts[None], tgt_keypts[None], src_features[None], tgt_features[None], gt_trans[None]


    def __len__(self):
        return len(self.infos['rot'])

class KITTILoader(data.Dataset):
    def __init__(self,
                root,
                descriptor='fcgf',
                inlier_threshold=0.60,
                num_node=5000,
                use_mutual=True,
                downsample=0.30
                ):
        self.root = root
        self.descriptor = descriptor
        assert descriptor in ['fcgf', 'fpfh']
        self.inlier_threshold = inlier_threshold
        self.num_node = num_node
        self.use_mutual = use_mutual
        self.downsample = downsample

        # containers
        self.ids_list = []

        for filename in os.listdir(f"{self.root}/"):
            self.ids_list.append(os.path.join(f"{self.root}/", filename))

        # self.ids_list = sorted(self.ids_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    def get_data(self, index):
        # load meta data
        filename = self.ids_list[index]
        data = np.load(filename)
        src_keypts = data['xyz0']
        tgt_keypts = data['xyz1']
        src_features = data['features0']
        tgt_features = data['features1']
        if self.descriptor == 'fpfh':
            src_features = src_features / (np.linalg.norm(src_features, axis=1, keepdims=True) + 1e-6)
            tgt_features = tgt_features / (np.linalg.norm(tgt_features, axis=1, keepdims=True) + 1e-6)

        # compute ground truth transformation
        gt_trans = data['gt_trans']

        return torch.from_numpy(src_keypts.astype(np.float32)).cuda()[None], \
               torch.from_numpy(tgt_keypts.astype(np.float32)).cuda()[None], \
               torch.from_numpy(src_features.astype(np.float32)).cuda()[None], \
               torch.from_numpy(tgt_features.astype(np.float32)).cuda()[None], \
               torch.from_numpy(gt_trans.astype(np.float32)).cuda()[None]

    def __len__(self):
        return len(self.ids_list)


