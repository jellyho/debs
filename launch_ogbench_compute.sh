#! /bin/bash


export PART='suma_A6000,gigabyte_A6000,ASUS_RTX6000ADA,TYAN_A6000,gigabyte_A5000,ASUS_A5000'
export QOS='base_qos'
export MUJOCO_GL='egl'
export JOBNAME="ogbench_compute"
export data_dir="/data5/jellyho/droid_rl/"
export save_dir="/tmp"
# export MODEL="meanflow"
# export lahd="(256,256)"
# export critic_agg="min"
# export extract_method="ddpg"


for task in "cube-single-play"; do
    for model_name in "meanflowq" "fmlql"; do
        for seed in "100"; do
            echo "${JOBNAME}_${task}_${seed}"
            # sbatch -p ${PART} -q ${QOS} --exclude=node19,node08,node16 --gres=gpu:1 -J ${JOBNAME}_${task}_${seed} -o ~/.slurm_logs/${JOBNAME}_${task}_${seed}.log debs_droid.sh ${task} ${model_name} 1.0 1.0 ${JOBNAME} ${seed} ${data_dir}
            sh debs_ogbench_compute.sh ${task} ${model_name} 1.0 1.0 ${JOBNAME} ${seed} ${data_dir} ${save_dir}
        done
    done
done