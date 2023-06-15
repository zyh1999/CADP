# Is Centralized Training with Decentralized Execution Framework Centralized Enough for MARL?

[![License: Apache](https://img.shields.io/badge/License-Apache-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2305.17352-b31b1b.svg)](https://arxiv.org/abs/2305.17352)

 Official codebase for paper [Is Centralized Training with Decentralized Execution Framework Centralized Enough for Multi-Agent Learning?](https://arxiv.org/abs/2305.17352). This codebase is based on the open-source [PyMARL](https://github.com/oxwhirl/pymarl) and [On-Policy](https://github.com/marlbenchmark/on-policy) framework and please refer to that repo for more documentation.

<div align="center">
<img src="https://github.com/zyh1999/CADP/blob/main/introduction.png" width="50%">
</div>

## Overview

**TLDR:** Our main contribution is the first dedicated attempt towards fully centralized training for CTDE, a highly practical yet largely overlooked problem, achieved through adopting a prunable agent communication mechanism. We propose a novel Centralized Advising and Decentralized Pruning (CADP) framework to promote explicit agent cooperation during training while still ensure the independent policies for execution. CADP is designed to provide a new general training framework for different MARL methods based on CTDE.

**Abstract:** Centralized Training with Decentralized Execution (CTDE) has recently emerged as a popular framework for cooperative Multi-Agent Reinforcement Learning (MARL), where agents can use additional global state information to guide training in a centralized way and make their own decisions only based on decentralized local policies. Despite the encouraging results achieved, CTDE makes an independence assumption on agent policies, which limits agents to adopt global cooperative information from each other during centralized training. Therefore, we argue that existing CTDE methods cannot fully utilize global information for training, leading to an inefficient joint-policy exploration and even suboptimal results. In this paper, we introduce a novel Centralized Advising and Decentralized Pruning (CADP) framework for multi-agent reinforcement learning, that not only enables an efficacious message exchange among agents during training but also guarantees the independent policies for execution. Firstly, CADP endows agents the explicit communication channel to seek and take advices from different agents for more centralized training. To further ensure the decentralized execution, we propose a smooth model pruning mechanism to progressively constraint the agent communication into a closed one without degradation in agent cooperation capability. Empirical evaluations on StarCraft II micromanagement and Google Research Football benchmarks demonstrate that the proposed framework achieves superior performance compared with the state-of-the-art counterparts.


![image](https://github.com/zyh1999/CADP/blob/main/framework.png)


## Prerequisites

#### Install dependencies

See `requirment.txt` file for more information about how to install the dependencies.



#### Install StarCraft II

Please use the Blizzard's [repository](https://github.com/Blizzard/s2client-proto#downloads) to download the Linux version 4.10 of StarCraft II. By default, the game is expected to be in `~/StarCraftII/` directory. This can be changed by setting the environment variable `SC2PATH`.

```diff
- Please pay attention to the version of SC2 you are using for your experiments. 
- We use the latest version SC2.4.10 for all SMAC experiments instead of SC2.4.6.2.69232.
- Performance is not comparable across versions.
```

The SMAC maps used for all experiments is in `CADP/src/envs/starcraft2/maps/SMAC_Maps` directory. You should place the `SMAC_Maps` directory in `StarCraftII/Maps`.



#### Install GFootball

Please follow the Google Research Football's [repository](https://github.com/google-research/football) to download GFootball environment.




## Usage

Please follow the instructions below to replicate the results in the paper.




#### SMAC
```bash
# VD Methods
cd CADP-VD
python src/main.py --config=<ALG_NAME>_CADP --env-config=sc2 with env_args.map_name=<MAP_NAME>
# ALG_NAME: VDN QMIX QPLEX
# MAP_NAME: 5m_vs_6m corridor 3s5z_vs_3s6z

# PG Method (MAPPO)
cd CADP-PG/onpolicy
sh scripts/train_smac_scripts/train_smac_<MAP_NAME>_cadp.sh
# MAP_NAME: 5m_vs_6m corridor 3s5z_vs_3s6z
```

<div align="center">
<img src="https://github.com/zyh1999/CADP/blob/main/exp-smac.png" width="100%">
</div>


#### GFootball

```bash
python src/main.py --config=<ALG_NAME>_CADP --env-config=gfootball with env_args.map_name=<MAP_NAME> optimizer='rmsprop'
# ALG_NAME: QMIX
# MAP_NAME: academy_3_vs_1_with_keeper academy_counterattack_easy
```



<div align="center">
<img src="https://github.com/zyh1999/CADP/blob/main/exp-gfootball.png" width="66%">
</div>


## Citation

If you find this work useful for your research, please cite our paper:

```
@article{zhou2023CADP,
  title={Is Centralized Training with Decentralized Execution Framework Centralized Enough for MARL?},
  author={Zhou, Yihe and Liu, Shunyu and Qing, Yunpeng and Chen, Kaixuan and Zheng, Tongya and Huang, Yanhao and Song, Jie and Song, Mingli},
  journal={arXiv preprint arXiv:2305.17352},
  year={2023}
}
```

## Contact

Please feel free to contact me via email (<zhouyihe@zju.edu.cn>, <liushunyu@zju.edu.cn>) if you are interested in my research :)
