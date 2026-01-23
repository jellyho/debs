#!/bin/bash

MUJOCO_GL=egl
python main.py \
    --agent "agents/$2.py" \
    --project "debs_bandit" \
    --run_group=$2:$5 \
    --env_name="bandit-$1" \
    --horizon_length 1 \
    --agent.lr 3e-4 \
    --eval_interval 10000 \
    --video_episodes 1 \
    --offline_steps 100000 \
    --save_dir "exp/" \
    --agent.latent_dist $3 \
    --agent.alpha $4 \
    --agent.mf_method 'jit_mf' \
    # --seed $5 \
    # --agent.use_DiT
    # --agent.mf_method='jit_mf' \
    # --agent.rl_method='ddpg' \
    # --agent.extract_method='ddpg' \
    # --agent.num_critic=2 \