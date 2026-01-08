#!/bin/bash

export MUJOCO_GL="egl"
# export CUDA_VISIBLE_DEVICES=3

python main.py \
    --agent "agents/$2.py" \
    --project "MFQ" \
    --run_group "$2:$5" \
    --task_name $1 \
    --env_name "$1-mh-low_dim" \
    --horizon_length 1 \
    --agent.lr 3e-4 \
    --eval_interval 100000 \
    --eval_episodes 50 \
    --video_episodes 10 \
    --offline_steps 1000000 \
    --save_dir "exp/" \
    --agent.num_critic 2 \
    --agent.latent_dist "$3" \
    --seed $6 \
    --agent.alpha $4 \
    --agent.extract_method "ddpg"
    # --agent.use_DiT
    # --agent.time_dist "$6" \
    # --agent.alpha $4 \
    # --agent.extract_method "ddpg" \