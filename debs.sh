#!/bin/bash

MUJOCO_GL=egl
python main.py \
    --run_group=reproduce \
    --env_name=$1-play-singletask-task$2-v0 \
    --sparse=False \
    --horizon_length=10 \
    --agent.lr 3e-4 \
    --eval_interval 100000 \
    --video_episodes 10