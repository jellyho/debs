#!/bin/bash
# --checkpoint_path "/home/rllab2/jellyho/droid_ckpts/lql/MFLQL_debug_v2" \
MUJOCO_GL=egl
python droid_eval.py \
    --checkpoint_path "/home/rllab2/jellyho/droid_ckpts/flow/FLOW_debug_v1" \
    --checkpoint_step 10000 \
    --seed 100
