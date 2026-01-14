#!/bin/bash

export MUJOCO_GL="egl"

python main.py \
    --agent "agents/$2.py" \
    --project "MFQ" \
    --run_group "$2:$5" \
    --task_name $1 \
    --task_num $6 \
    --env_name "visual-$1-singletask-task$6-v0" \
    --horizon_length 5 \
    --agent.lr 3e-4 \
    --eval_interval 50000 \
    --eval_episodes 50 \
    --video_episodes 0 \
    --offline_steps 500000 \
    --save_dir "exp/" \
    --agent.num_critic 2 \
    --agent.latent_dist "$3" \
    --agent.alpha $4 \
    --seed $7 \
    --agent.encoder "impala_small" \
    --p_aug=0.5 \
    --agent.extract_method "ddpg" \
    --agent.mf_method "jit_mf"
    
    # --agent.weight_decay 0.1 \

# #-task$2-v0 \
