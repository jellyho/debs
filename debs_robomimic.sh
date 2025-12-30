#!/bin/bash

# source ~/.bashrc
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate debs  # 본인 환경 이름

export MUJOCO_GL="egl"
# export PYOPENGL_PLATFORM="egl"
# export LD_PRELOAD=MUJOCO_GL

# export CUDNN_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib
# export CUDART_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudart/lib
# export CUBLAS_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib

# # 위 경로들을 LD_LIBRARY_PATH '맨 앞'에 붙여줍니다.
# export LD_LIBRARY_PATH=$CUDNN_PATH:$CUDART_PATH:$CUBLAS_PATH:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python main.py \
    --agent "agents/$2.py" \
    --project "debs" \
    --run_group $2 \
    --task_name $1 \
    --env_name "$1-mh-low_dim" \
    --horizon_length 5 \
    --agent.lr 3e-4 \
    --eval_interval 100 \
    --eval_episodes 5 \
    --video_episodes 10 \
    --offline_steps 1000 \
    --save_dir "exp/" \
    --agent.mf_method='jit_mf' \
    --agent.rl_method='ddpg' \
    --agent.extract_method='ddpg' \
    --agent.num_critic=2 \
    --agent.latent_dist='sphere' \
    --agent.late_update=False \
    # --agent.alpha=1.0 \

# #-task$2-v0 \