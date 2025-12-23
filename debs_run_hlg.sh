#!/bin/bash

source /opt/conda/etc/profile.d/conda.sh
conda activate debs

MUJOCO_GL=egl
python main.py \
    --agent "agents/$3.py" \
    --run_group=${10}:$3 \
    --env_name=$1-singletask-task$2-v0 \
    --horizon_length $4 \
    --agent.lr 3e-4 \
    --seed $5 \
    --eval_interval 100000 \
    --video_episodes 10 \
    --offline_steps 1000000 \
    --save_dir "exp/" \
    --agent.num_bins=101 \
    --agent.mf_method='jit_mf' \
    --agent.num_critic=$6 \
    --agent.latent_dist=$7 \
    --agent.late_update=$8 \
    --agent.extract_method=$9 \
    --agent.target_mode=${11} \
