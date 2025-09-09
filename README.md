<div align="center">

<h1>DCReg: Decoupled Characterization for Efficient Degenerate LiDAR Registration System</h1>

[**Xiangcheng Hu**](https://github.com/JokerJohn)<sup>1</sup> · [**Xieyuanli Chen**](https://chen-xieyuanli.github.io/)<sup>2&dagger;</sup> · [**Mingkai Jia**](https://scholar.google.com/citations?user=fcpTdvcAAAAJ&hl=en)<sup>1</sup> ·
[**Jin Wu**](https://zarathustr.github.io/) <sup>3*</sup>
<br>
 [**Ping Tan**](https://facultyprofiles.hkust.edu.hk/profiles.php?profile=ping-tan-pingtan#publications)<sup>1</sup> and  [**Steven L. Waslander**](https://www.trailab.utias.utoronto.ca/steven-waslander)<sup>4</sup>

<sup>1</sup>HKUST&emsp;&emsp;&emsp;<sup>2</sup>NUDT&emsp;&emsp;&emsp;<sup>3</sup>USTB &emsp;&emsp;&emsp;<sup>4</sup>U of T
<br>
&dagger;Project lead&emsp;*Corresponding author

<a href="https://arxiv.org/pdf/2408.03723"><img src='https://img.shields.io/badge/ArXiv-DCReg-red' alt='Paper PDF'></a>[![GitHub Stars](https://img.shields.io/github/stars/JokerJohn/DCReg.svg)](https://github.com/JokerJohn/DCReg/stargazers) [![GitHub Issues](https://img.shields.io/github/issues/JokerJohn/DCReg.svg)](https://github.com/JokerJohn/DCReg/issues)[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)<a href="https://github.com/JokerJohn/DCReg/blob/main/">

</div>



In this study, we introduce **[DCReg](https://arxiv.org/abs/2509.06285)** (**D**ecoupled **C**haracterization for ill-conditioned **Reg**istration), a principled framework
that systematically addresses the ill-conditioned registration problems through three integrated innovations. 

- First, DCReg achieves **reliable ill-conditioning detection** by employing a Schur complement decomposition to the hessian matrix. This technique decouples the registration problem into clean rotational and translational subspaces, eliminating coupling effects that mask degeneracy patterns in conventional analyses. 
- Second, within these cleanly subspaces, we develop **quantitative characterization** techniques that establish explicit mappings between mathematical eigenspaces and physical motion directions, providing actionable insights about which specific motions lack constraints. 
- Finally, leveraging this clean subspace, we design a **targeted mitigation** strategy: a novel preconditioner that selectively stabilizes only the identified ill-conditioned directions while preserving all well-constrained information in observable space. This enables efficient and robust optimization via the Preconditioned Conjugate Gradient method with a single physical interpretable parameter. 

Extensive experiments demonstrate **DCReg** achieves **at least 20% - 50% improvement in localization accuracy and 5-100 times speedup** over state-of-the-art methods across diverse environments. 



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

### Degeneracy Detection

![image-20250908195304202](./README/image-20250908195304202.png)

<div align="center">

![image-20250908195247186](./README/image-20250908195247186.png) 
</div>


| ![image-20250908195226346](./README/image-20250908195226346.png) | ![image-20250908195236593](./README/image-20250908195236593.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

### Degeneracy Characterization

![image-20250908195356150](./README/image-20250908195356150.png)

![image-20250908195410597](./README/image-20250908195410597.png)

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

The authors gratefully acknowledge the valuable contributions that made this work possible. We extend special thanks to [Mr. Binqian Jiang](https://github.com/lewisjiang) and [Dr. Jianhao Jiao](https://gogojjh.github.io/) for their insightful discussions that significantly contributed to refining the theoretical framework presented in this paper. We also appreciate [Mr. Turcan Tuna](https://www.turcantuna.com/) for his technical assistance with the baseline algorithm implementation.

## Contributors

<a href="https://github.com/JokerJohn/DCReg/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=JokerJohn/DCReg" />
</a>