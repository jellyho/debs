#!/bin/bash

export MUJOCO_GL="egl"

python main_compute.py \
    --agent "agents/$2.py" \
    --project "MFQ_OGBENCH_COMPUTE" \
    --run_group "$2:$5" \
    --task_name $1 \
    --task_num 1 \
    --env_name "$1-singletask-task1-v0" \
    --horizon_length 5 \
    --agent.lr 3e-4 \
    --eval_interval 1000 \
    --save_interval 100000 \
    --eval_episodes 50 \
    --video_episodes 10 \
    --offline_steps 1000 \
    --save_dir "$8" \
    --agent.num_critic 2 \
    --agent.alpha $4 \
    --agent.extract_method "ddpg" \
    --agent.mf_method "jit_mf" \
    --seed $6 \
    --log_interval 2 \
    --agent.critic_hidden_dims "(256, 256, 256, 256)" \
    --agent.latent_actor_hidden_dims "(256, 256)" \
    # --agent.weight_decay 0.1

# #-task$2-v0 \
