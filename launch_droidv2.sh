#! /bin/bash


export PART='suma_A6000,gigabyte_A6000,ASUS_RTX6000ADA,TYAN_A6000,gigabyte_A5000,ASUS_A5000'
export QOS='base_qos'
export MUJOCO_GL='egl'
export JOBNAME="droid_20k"
export data_dir="/data5/jellyho/droid_rl/"
export save_dir="/data5/jellyho/lql_droid_20k/"
# export MODEL="meanflow"
# export lahd="(256,256)"
# export critic_agg="min"
# export extract_method="ddpg"


for task in "droid_eggplant_bin" "droid_plug_in_bulb"; do
    for model_name in "meanflowq" "dsrl" "flow" "meanflow"; do
        for seed in "100"; do
            echo "${JOBNAME}_${task}_${seed}"
            sh debs_droid.sh ${task} ${model_name} 1.0 1.0 ${JOBNAME} ${seed} ${data_dir} ${save_dir}
        done
    done
done