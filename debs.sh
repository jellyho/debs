#!/bin/bash

MUJOCO_GL=egl
python main.py \
    --agent "agents/$3.py" \
    --project "debs" \
    --run_group=$3 \
    --env_name=$1-singletask-task$2-v0 \
    --sparse=False \
    --horizon_length 5 \
    --agent.lr 3e-4 \
    --eval_interval 100000 \
    --video_episodes 10 \
    --offline_steps 1000000 \
    --save_dir "exp/" \
    --agent.mf_method='jit_mf' \
    --agent.rl_method='ddpg' \
    --agent.extract_method='onestep_ddpg' \
    --agent.num_critic=2 \
    --agent.latent_dist='normal' \
    --agent.late_update=False \
    --agent.alpha=3.0 \
