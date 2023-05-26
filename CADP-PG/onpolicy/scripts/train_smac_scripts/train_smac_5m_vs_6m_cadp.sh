#!/bin/sh
env="StarCraft2"
map="5m_vs_6m"
algo="rmappo"
exp="cadp"
seed_max=5

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python scripts/train/train_smac.py --use_CADP --use_wandb --cadp_breakpoint 6000000 \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 \
    --num_env_steps 10000000 --ppo_epoch 10 --clip_param 0.05 --use_value_active_masks --use_eval --eval_episodes 32
done
