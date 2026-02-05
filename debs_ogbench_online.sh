#!/bin/bash

export MUJOCO_GL="egl"

python main.py \
    --agent "agents/$2.py" \
    --project "LPS_ONLINE" \
    --run_group "$2:$5" \
    --task_name $1 \
    --task_num $6 \
    --env_name "$1-singletask-task$6-v0" \
    --horizon_length 5 \
    --agent.lr 3e-4 \
    --eval_interval 100000 \
    --eval_episodes 50 \
    --video_episodes 10 \
    --agent.extract_method $8 \
    --offline_steps 1000000 \
    --online_steps 1000000 \
    --save_dir "exp/" \
    --agent.num_critic 2 \
    --agent.latent_dist "$3" \
    --agent.alpha $4 \
    --seed $7 \
    --agent.critic_agg "min" \
    --agent.mf_method "jit_mf" \

# #-task$2-v0 \
