#!/bin/bash

export MUJOCO_GL="egl"

# unset DISPLAY


# export LD_LIBRARY_PATH=$(python3 -c "import os; import site; print(os.pathsep.join([os.path.join(p, 'nvidia/cudnn/lib') for p in site.getsitepackages()] + [os.path.join(p, 'nvidia/cublas/lib') for p in site.getsitepackages()] + [os.path.join(p, 'nvidia/cuda_runtime/lib') for p in site.getsitepackages()]))"):$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64:/usr/local/nvidia/lib64
# # [중요] EGL이 NVIDIA 드라이버를 강제로 사용하도록 지정
# export __GLX_VENDOR_LIBRARY_NAME=nvidia

# # [중요] EGL 플랫폼을 'device'로 설정 (Slurm 환경에서 필수적인 경우가 많음)
# export EGL_PLATFORM=device
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu

# python egltest.py

python main.py \
    --agent "agents/$2.py" \
    --project "MFQ_DEBUG" \
    --run_group "$2" \
    --task_name $1 \
    --env_name "$1-mh-low_dim" \
    --horizon_length 5 \
    --agent.lr 3e-4 \
    --eval_interval 10000 \
    --eval_episodes 50 \
    --video_episodes 10 \
    --offline_steps 100000 \
    --save_dir "exp/" \
    --agent.mf_method "jit_mf" \
    --agent.rl_method "ddpg" \
    --agent.extract_method "ddpg" \
    --agent.num_critic 2 \
    --agent.latent_dist "sphere" \
    --agent.alpha 1.0
    # --agent.weight_decay 0.1

# #-task$2-v0 \