#!/bin/bash
# CUDA_VISIBLE_DEVICES=6

MUJOCO_GL=egl
python main.py \
    --agent "agents/$2.py" \
    --run_group=$2 \
    --env_name="bandit-$1" \
    --sparse=False \
    --horizon_length 1 \
    --agent.lr 3e-4 \
    --eval_interval 10000 \
    --video_episodes 10 \
    --offline_steps 100000 \
    --save_dir "exp/" \
    --agent.v_min=0.0 \
    --agent.num_bins=101 \
    --agent.v_max=1.0 \
    --agent.mf_method='jit_mf' \
    --agent.num_critic=2 \
    # --agent.noisy_actor=True
    # --agent.target_mode='mean'