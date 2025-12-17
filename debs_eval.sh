#!/bin/bash
# CUDA_VISIBLE_DEVICES=6

MUJOCO_GL=egl
python main_eval.py \
    --checkpoint_path "/home/v-hokyunim/Offline/debs/exp/debs/hldebs/cube-single-play-singletask-task5-v0/sd00020251211_111714" \
    --checkpoint_step 200000
