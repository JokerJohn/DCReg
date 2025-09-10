<div align="center">

<h1>DCReg: Decoupled Characterization for Efficient Degenerate LiDAR Registration</h1>

[**Xiangcheng Hu**](https://github.com/JokerJohn)<sup>1</sup> · [**Xieyuanli Chen**](https://chen-xieyuanli.github.io/)<sup>2&dagger;</sup> · [**Mingkai Jia**](https://scholar.google.com/citations?user=fcpTdvcAAAAJ&hl=en)<sup>1</sup> ·
[**Jin Wu**](https://zarathustr.github.io/) <sup>3*</sup>
<br>
 [**Ping Tan**](https://facultyprofiles.hkust.edu.hk/profiles.php?profile=ping-tan-pingtan#publications)<sup>1</sup>· [**Steven L. Waslander**](https://www.trailab.utias.utoronto.ca/steven-waslander)<sup>4</sup>

<sup>1</sup>HKUST&emsp;&emsp;&emsp;<sup>2</sup>NUDT&emsp;&emsp;&emsp;<sup>3</sup>USTB &emsp;&emsp;&emsp;<sup>4</sup>U of T
<br>
&dagger;Project lead&emsp;*Corresponding author

<a href="https://arxiv.org/abs/2509.06285"><img src='https://img.shields.io/badge/ArXiv-DCReg-red' alt='Paper PDF'></a>[![video](https://img.shields.io/badge/Video-Bilibili-74b9ff?logo=bilibili&logoColor=red)]( https://www.bilibili.com/video/BV1jsHQzCEra/?share_source=copy_web)[![GitHub Stars](https://img.shields.io/github/stars/JokerJohn/DCReg.svg)](https://github.com/JokerJohn/DCReg/stargazers) [![GitHub Issues](https://img.shields.io/github/issues/JokerJohn/DCReg.svg)](https://github.com/JokerJohn/DCReg/issues)[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)<a href="https://github.com/JokerJohn/DCReg/blob/main/">

</div>

**[DCReg](https://arxiv.org/abs/2509.06285)** (**D**ecoupled **C**haracterization for ill-conditioned **Reg**istration) is a principled framework that addresses ill-conditioned point cloud registration problems, achieving **20% - 50% accuracy improvement and 5-100 times** speedup over state-of-the-art methods.

- **Reliable ill-conditioning detection**: Decouples rotation and translation via Schur complement decomposition for ill-conditioning detection， eliminating coupling effects that mask degeneracy patterns.
- **Quantitative characterization**: Maps mathematical eigenspaces to physical motion space, revealing which and to what extent specific motions lack constraints
- **Targeted mitigation**: Employs targeted preconditioning that stabilizes only degenerate directions while preserving observable information.

DCReg seamlessly integrates with existing registration pipelines through an efficient PCG solver with a single interpretable parameter.



## Timeline

**2025/09/09:** the preprint paper is [online](https://arxiv.org/abs/2509.06285), baseline codes will be published first!



## Methods

![image-20250908194217193](./README/image-20250908194217193.png)

| ![image-20250908194259196](./README/image-20250908194259196.png) | ![image-20250908194344328](./README/image-20250908194344328.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |



## Baseline and dataset

| ![image-20250908194440555](./README/image-20250908194440555.png) |
| ------------------------------------------------------------ |
| ![image-20250909214128111](./README/image-20250909214128111.png) |

| ![image-20250908194514540](./README/image-20250908194514540.png) | ![image-20250908194526477](./README/image-20250908194526477.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |



## Video demo

![image-20250910212340395](./README/image-20250910212340395.png)

| Scenarios                                                    | Characterization                                             | Features                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![pk01_dcreg_seg](./README/8391c3ce-45dc-4b86-aed7-b496dc33ba87.gif) | ![image-20250910213549613](./README/image-20250910213549613.png) | <span style="font-size:12px;">Planer degeneracy, **t0-t1-r2** degenerate, the main components of motion sources are  **X-Y-Yaw**. e.g. t0 = 90.0% X + xx %Y + xx% Z. the related angles of X with t0 is 4.5 deg, that means X should be the main reason. **see figure 16.** </span>|
| ![](./README/45fc2afe-c7f9-41a1-ab93-e8cd96ee0d16.gif)       | ![image-20250910213208822](./README/image-20250910213208822.png) |  <span style="font-size:12px;">narrow stairs, spares features cause this degeneracy. sometimes t2, sometimes r0-r1. **see figure 17.**</span> |
| ![corridor_dcreg_x5](./README/corridor_dcreg_x5.gif)         | ![image-20250910213259165](./README/image-20250910213259165.png) |  <span style="font-size:12px;">narrow passage, r0-t0 or r0, depends on your measurements.</span>   |
| ![dcreg_x50](./README/dcreg_x50.gif)                         | ![image-20250910213415142](./README/image-20250910213415142.png) |  <span style="font-size:12px;">rich features but within narrow environments. r0-t0 or r0.</span>   |



### Controlled Simulation Analysis

| ![image-20250908194819193](./README/image-20250908194819193.png) |
| ------------------------------------------------------------ |
| ![image-20250908194834002](./README/image-20250908194834002.png) |

| ![image-20250908194848247](./README/image-20250908194848247.png) | ![image-20250908194901218](./README/image-20250908194901218.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

### Real-world Performance Evaluation

### localization and mapping

![image-20250908195036175](./README/image-20250908195036175.png)

| ![image-20250908195103021](./README/image-20250908195103021.png) | ![image-20250908195117064](./README/image-20250908195117064.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

### Degeneracy Characterization

| ![image-20250908195356150](./README/image-20250908195356150.png) |
| ------------------------------------------------------------ |
| ![image-20250908195410597](./README/image-20250908195410597.png) |


### Degeneracy Detection

![image-20250908195304202](./README/image-20250908195304202.png)

<div align="center">
![image-20250908195247186](./README/image-20250908195247186.png) 
</div>


| ![image-20250908195226346](./README/image-20250908195226346.png) | ![image-20250908195236593](./README/image-20250908195236593.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

## Ablation and Hybrid Analysis

| ![image-20250908195458538](./README/image-20250908195458538.png) | ![image-20250908195511133](./README/image-20250908195511133.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |



## Run-time analysis

| ![image-20250908195549384](./README/image-20250908195549384.png) | ![image-20250908195600116](./README/image-20250908195600116.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

## Parameter 

<div align="center">

![image-20250908195629999](./README/image-20250908195629999.png)
</div>

## Acknowledgment

The authors gratefully acknowledge the valuable contributions that made this work possible. 

- We extend special thanks to [Dr. Binqian Jiang](https://github.com/lewisjiang) and [Dr. Jianhao Jiao](https://gogojjh.github.io/) for their insightful discussions that significantly contributed to refining the theoretical framework presented in this paper. 
- We also appreciate [Mr. Turcan Tuna](https://www.turcantuna.com/) for his technical assistance with the baseline algorithm implementation.

## Contributors

<a href="https://github.com/JokerJohn/DCReg/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=JokerJohn/DCReg" />
</a>
