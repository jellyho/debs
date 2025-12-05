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
    --eval_interval 2000 \
    --video_episodes 10 \
    --offline_steps 10000 \
    --save_dir "exp/" \
    # --agent.cfg 2.0
