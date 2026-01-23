#! /bin/bash


export PART='suma_A6000,gigabyte_A6000,ASUS_RTX6000ADA,TYAN_A6000,gigabyte_A5000,ASUS_A5000'
export QOS='base_qos'
export MUJOCO_GL='egl'
export JOBNAME="droid_refill_tape"
export data_dir="/scratch/jellyho/"
# export MODEL="meanflow"
# export lahd="(256,256)"
# export critic_agg="min"
# export extract_method="ddpg"


for task in "droid_refill_tape"; do
    for model_name in "meanflowq" "dsrl" "flow" "meanflow"; do
        for seed in "100"; do
            echo "${JOBNAME}_${task}_${seed}"
            sbatch -p ${PART} -q ${QOS} --exclude=node19,node08,node16 --gres=gpu:1 -J ${JOBNAME}_${task}_${seed} -o ~/.slurm_logs/${JOBNAME}_${task}_${seed}.log debs_droid_compute.sh ${task} ${model_name} 1.0 1.0 ${JOBNAME} ${seed} ${data_dir}
        done
    done
done