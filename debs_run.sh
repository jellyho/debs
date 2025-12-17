#!/bin/bash

source /opt/conda/etc/profile.d/conda.sh
conda activate debs

MUJOCO_GL=egl
python main.py \
    --seed $5 \
    --agent "agents/$3.py" \
    --run_group=$3 \
    --env_name=$1-singletask-task$2-v0 \
    --sparse=False \
    --horizon_length $4 \
    --agent.lr 3e-4 \
    --eval_interval 100000 \
    --video_episodes 10 \
    --offline_steps 1000000 \
    --save_dir "exp/" \
    --agent.num_bins=101 \
    --agent.mf_method='jit_mf' \
    # --agent.target_mode='mean'
