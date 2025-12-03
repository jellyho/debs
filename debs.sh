#!/bin/bash
# CUDA_VISIBLE_DEVICES=6

MUJOCO_GL=egl
python main.py \
    --agent "agents/$3.py
    --run_group=$3 \
    --env_name=$1-play-singletask-task$2-v0 \
    --sparse=False \
    --horizon_length 10 \
    --agent.lr 3e-4 \
    --eval_interval 100000 \
    --video_episodes 10 \
    --offline_steps 500000 \
    --save_dir "exp/" \
    --agent.cfg 2.0
