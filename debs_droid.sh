#!/bin/bash

export MUJOCO_GL="egl"

python main.py \
    --agent "agents/$2.py" \
    --project "MFQ_DROID" \
    --run_group "$2:$5" \
    --droid_dataset_dir "$7" \
    --droid_use_failure true \
    --task_name $1 \
    --task_num 0 \
    --env_name "$1" \
    --horizon_length 5 \
    --agent.lr 3e-4 \
    --eval_interval 10000 \
    --save_interval 5000 \
    --eval_episodes 50 \
    --video_episodes 10 \
    --offline_steps 10000 \
    --save_dir $8 \
    --agent.num_critic 2 \
    --agent.alpha $4 \
    --agent.extract_method "ddpg" \
    --agent.mf_method "jit_mf" \
    --seed $6 \
    --agent.encoder "impala" \
    --p_aug=0.5 \
    --agent.use_DiT \
    --agent.size_DiT "small" \
    --log_interval 100 \
    --agent.critic_hidden_dims "(512, 512, 512, 512)" \
    --agent.latent_actor_hidden_dims "(512, 512)" \
    # --agent.weight_decay 0.1

# #-task$2-v0 \
