#!/bin/bash

MUJOCO_GL=egl
python droid_eval.py \
    --checkpoint_path "exp/MFQ_DROID/meanflow:debug/droid_pnp_carrot/sd001s_18664.0.20260116_133937" \
    --checkpoint_step 1000 \
    --seed 100
