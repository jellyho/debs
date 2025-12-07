#!/bin/bash
# CUDA_VISIBLE_DEVICES=6

MUJOCO_GL=egl
python main_eval.py \
    --checkpoint_path "/home/v-hokyunim/Offline/debs/exp/debs/debs/cube-single-play-singletask-task1-v0/sd00020251206_163727" \
    --checkpoint_step 200000
