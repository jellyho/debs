#!/bin/bash

# 인자: gpu개수, job이름, single_pretrained_path, data_mix, learning_rate, lr_scheduler_type, batch_size
LEARNING_RATE=0.00002
LR_SCHEDULER_TYPE=cosine
BATCH_SIZE=8
NUM_GPU=1
ENV=$1

while :; do
    RDZV_PORT=$((10000 + RANDOM % 20000))
    (echo >/dev/tcp/localhost/$RDZV_PORT) &>/dev/null || break
done
#  -p suma_rtx4090 -q base_qos
#  -p suma_a100 -q a100_qos
srun --gres=gpu:$NUM_GPU --cpus-per-task=4 torchrun --rdzv_id=$SLURM_JOB_ID --rdzv_backend=static --master_port=$RDZV_PORT --nnodes 1 --nproc-per-node $NUM_GPU scripts/train_twinvla.py \
    --model_type "Eagle2_1BTwinVLA" \
    --singlevla_pretrained_path "/data5/jellyho/singlevla_checkpoints/Eagle2_1BVLA-oxe_magic_soup_plus_minus_100k" \
    --learning_rate "$LEARNING_RATE" \
    --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --adam_beta1 0.95 \
    --adam_beta2 0.999 \
    --max_grad_norm 1.0 \
    --gradient_accumulation_steps 1 \
    --adam_epsilon 1e-8 \
    --max_steps 100000 \
    --save_steps 20000 \
    --shuffle_buffer_size 50000 \
    --batch_size "$BATCH_SIZE" \
    --data_root_dir "/data5/jellyho/tabletop-simulation-rlds" \
    --data_mix $ENV \
    --output_dir "checkpoints/twinvla-lr-$LEARNING_RATE-$LR_SCHEDULER_TYPE-batchsize-$BATCH_SIZE-gpu-$NUM_GPU-$ENV-reproduce" \
    --image_aug false \
    --wandb_project "TWINVLA_PARAM_SEARCH" \
    --enable_autotune false \
    --freeze_vision_backbone true \
    --log_grad false \
    --bf16 true \
    --resume false

