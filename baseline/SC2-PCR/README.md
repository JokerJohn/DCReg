# SC^2-PCR: A Second Order Spatial Compatibility for Efficient and Robust Point Cloud Registration (CVPR 2022)

PyTorch implementation of the paper:

[SC^2-PCR: A Second Order Spatial Compatibility for Efficient and Robust Point Cloud Registration](https://arxiv.org/abs/2203.14453).

Zhi Chen, [Kun Sun](https://scholar.google.com/citations?user=Ay6kCm4AAAAJ&hl=en), Fan Yang, [Wenbing Tao](https://scholar.google.co.uk/citations?user=jRDPE2AAAAAJ&hl=zh-CN&oi=ao).

## Introduction

In this paper, we present a second order spatial compatibility (SC^2) measure based method for efficient and robust point cloud registration (PCR), called SC^2-PCR. Firstly, we propose a second order spatial compatibility (SC^2) measure to compute the similarity between correspondences. It considers the global compatibility instead of local consistency, allowing for more distinctive clustering between inliers and outliers at early stage. Based on this measure, our registration pipeline employs a global spectral technique to find some reliable seeds from the initial correspondences. Then we design a two-stage strategy to expand each seed to a consensus set based on the SC^2 measure matrix. Finally, we feed each consensus set to a weighted SVD algorithm to generate a candidate rigid transformation and select the best model as the final result. Our method can guarantee to find a certain number of outlier-free consensus sets using fewer samplings, making the model estimation more efficient and robust. In addition, the proposed SC^2 measure is general and can be easily plugged into deep learning based frameworks. Extensive experiments are carried out to investigate the performance of our method.

![](figures/pipeline.png)

## Requirements

If you are using conda, you may configure SC2-PCR as:

    conda env create -f environment.yml
    conda activate SC2_PCR
    
## 3DMatch

### Data preparation

Downsample and extract FPFH and FCGF descriptors for each frame of the 3DMatch test dataset. [Here](https://drive.google.com/file/d/1kRwuTHlNPr9siENcEMddCO23Oaq0cz-X/view?usp=sharing) we provide the processed test set with pre-computed FPFH/FCGF descriptors. The data should be organized as follows:

```
--data--3DMatch                
        ├── fragments                 
        │   ├── 7-scene-redkitechen/
        |   |   ├── cloud_bin_0.ply
        |   |   ├── cloud_bin_0_fcgf.npz
        |   |   ├── cloud_bin_0_fpfh.npz
        │   |   └── ...      
        │   ├── sun3d-home_at-home_at_scan1_2013_jan_1/      
        │   └── ...                
        ├── gt_result                   
        │   ├── 7-scene-redkitechen-evaluation/   
        |   |   ├── 3dmatch.log
        |   |   ├── gt.info
        |   |   ├── gt.log
        │   |   └── ...
        │   ├── sun3d-home_at-home_at_scan1_2013_jan_1-evaluation/
        │   └── ...                               
```

### Testing

Use the following command for testing.

```bash
python ./test_3DMatch.py --config_path config_json/config_3DMatch.json
```

The CUDA_DEVICE and basic parameters can be changed in the json file.

## 3DLoMatch

### Data preparation

FPFH and FCGF descriptors can be prepared in the same way as testing 3DMatch. If you want to test the [predator](https://github.com/prs-eth/OverlapPredator) descriptor, you should first follow the offical instruction of predator to extract the descriptors for 3DMatch dataset and organize the data as follows: 

```
--data--3DLoMatch                
        ├── 0.pth        
        ├── 1.pth                 
        ├── ...  
        └── 1780.pth
```

### Testing

Use the following command for testing.

```bash
python ./test_3DLoMatch.py --config_path config_json/config_3DLoMatch.json
```

## KITTI odometry

### Data preparation

Downsample and extract FPFH and FCGF descriptors for each frame of the KITTI test dataset. The raw point clouds can be download from [KITTI Odometry website.](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). For your convenience, [here](https://drive.google.com/drive/folders/1sxkHYjWHhSUE3IcvmZ2p1ziw1LqJqqfc?usp=sharing) we provide the pre-computed FPFH and FCGF descriptors for the KITTI test set.

```
--data--KITTI                
        ├── fpfh_test                 
        │   ├── pair_0.npz        
        |   ├── pair_1.npz                
        |   ├── ...  
        |   └── pair_554.npz
        ├── fcgf_test                
        │   ├── pair_0.npz        
        |   ├── pair_1.npz                
        |   ├── ...  
        |   └── pair_554.npz
```

### Testing

Use the following command for testing.

```bash
python ./test_KITTI.py --config_path config_json/config_KITTI.json
```

## Results

### 3DMatch

We evaluate SC^2-PCR on the standard 3DMatch benchmarks:

|    Benchmark   | RR(%) | RE(°) |TE(cm) | IP(%) | IR(%) | F1(%) |
|:---------------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|  3DMatch+FPFH  | 83.98 | 2.18  | 6.56  | 72.48 | 78.33 | 75.10 |
|  3DMatch+FCGF  | 93.28 | 2.08  | 6.55  | 78.94 | 86.39 | 82.20 |

### 3DMatch

We evaluate SC^2-PCR on the standard 3DLoMatch benchmarks:

|       Benchmark     | RR(%) | RE(°) |TE(cm) | IP(%) | IR(%) | F1(%) |
|:--------------------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| 3DLoMatch+FCGF      | 57.83 | 3.77  | 10.46 | 44.87 | 53.69 | 48.38 |
| 3DLoMatch+Predator  | 69.46 | 3.46  | 9.58  | 56.98 | 67.47 | 61.08 |

### KITTI odometry

We evaluate SC^2-PCR on the standard KITTI benchmarks:

|    Benchmark   | RR(%) | RE(°) |TE(cm) | IP(%) | IR(%) | F1(%) |
|:---------------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|    KITTI+FPFH  | 99.64 | 0.32  | 7.23  | 93.63 | 95.89 | 94.63 |
|    KITTI+FCGF  | 98.20 | 0.33  | 20.95 | 82.01 | 91.03 | 85.90 |

## Citation

```bibtex
@InProceedings{Chen_2022_CVPR,
    author    = {Chen, Zhi and Sun, Kun and Yang, Fan and Tao, Wenbing},
    title     = {SC2-PCR: A Second Order Spatial Compatibility for Efficient and Robust Point Cloud Registration},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {13221-13231}
}
```

## Acknowledgements

- [PointDSC](https://github.com/XuyangBai/PointDSC)
- [FCGF](https://github.com/chrischoy/FCGF)
- [DGR](https://github.com/chrischoy/DeepGlobalRegistration)
- [PREDATOR](https://github.com/prs-eth/OverlapPredator)

