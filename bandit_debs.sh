#!/bin/bash
# CUDA_VISIBLE_DEVICES=6

MUJOCO_GL=egl
python main.py \
    --agent "agents/$2.py" \
    --project "debs" \
    --run_group=$2 \
    --env_name="bandit-$1" \
    --horizon_length 1 \
    --agent.lr 3e-4 \
    --eval_interval 10000 \
    --video_episodes 1 \
    --offline_steps 100000 \
    --save_dir "exp/" \
    --agent.mf_method='jit_mf' \
    --agent.rl_method='ddpg' \
    --agent.extract_method='onestep_ddpg' \
    --agent.num_critic=2 \
    --agent.latent_dist='normal' \
    --agent.late_update=False \
    --agent.alpha 0.1 \