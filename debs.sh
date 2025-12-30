#!/bin/bash

MUJOCO_GL=egl
python main.py \
    --agent "agents/$3.py" \
    --project "debs" \
    --run_group $3 \
    --task_name $1 \
    --task_num $2 \
    --env_name "$1-singletask-v0" \
    --horizon_length 5 \
    --agent.lr 3e-4 \
    --eval_interval 10000 \
    --video_episodes 10 \
    --offline_steps 100000 \
    --save_dir "exp/" \
    --agent.mf_method='jit_mf' \
    --agent.rl_method='ddpg' \
    --agent.extract_method='ddpg' \
    --agent.num_critic=2 \
    --agent.latent_dist='sphere' \
    --agent.late_update=False \
    # --agent.alpha=1.0 \

# #-task$2-v0 \