#!/bin/bash

export MUJOCO_GL="egl"

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
    --agent.mf_method "jit_mf" \
    --agent.rl_method "ddpg" \
    --agent.extract_method "ddpg" \
    --agent.num_critic 2 \
    --agent.latent_dist "$3" \
    --agent.time_dist "discrete" \
    --agent.alpha $4 \
    # --agent.weight_decay 0.1

# #-task$2-v0 \