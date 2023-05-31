# Is Centralized Training with Decentralized Execution Framework Centralized Enough for MARL?

 Official codebase for paper [Is Centralized Training with Decentralized Execution Framework Centralized Enough for Multi-Agent Learning?](). This codebase is based on the open-source [PyMARL](https://github.com/oxwhirl/pymarl) and [On-Policy](https://github.com/marlbenchmark/on-policy) framework and please refer to that repo for more documentation.



## 1. Prerequisites

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




## 2. Usage

Please follow the instructions below to replicate the results in the paper.




#### CADP for VD
```bash
cd CADP-VD

# SMAC
python src/main.py --config=[alg_name]_CADP --env-config=sc2 with env_args.map_name=[map_name]

# GRF
python src/main.py --config=[alg_name]_CADP --env-config=gfootball with env_args.map_name=[map_name] optimizer='rmsprop'
```



#### CADP for PG

```bash
cd CADP-PG/onpolicy

# SMAC
sh scripts/train_smac_scripts/train_smac_[map_name]_cadp.sh
```


## 3. Citation

If you find this work useful for your research, please cite our paper:

```
@article{Zhou2023CADP,
  title={Is Centralized Training with Decentralized Execution Framework Centralized Enough for MARL?},
  author={Yihe Zhou, Shunyu Liu, Yunpeng Qing, Kaixuan Chen, Tongya Zheng, Yanhao Huang, Jie Song, Mingli Song},
  journal={arXiv preprint arXiv:2305.17352},
  year={2023}
}
```
