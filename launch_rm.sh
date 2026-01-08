#! /bin/bash


export PART='base_suma_rtx3090,big_suma_rtx3090,suma_rtx4090,suma_A6000,gigabyte_A6000,gigabyte_A5000'
export QOS='big_qos'
export MUJOCO_GL='egl'
export JOBNAME="LQL_RM_chunk1"
export MODEL="meanflowq"


for latent in "sphere"; do
    for task in "can" "lift" "square"; do
        for seed in "100"; do
            echo "${JOBNAME}_${task_num}_${latent}_${task}_${seed}"
            sbatch -p ${PART} -q ${QOS} --gres=gpu:1 -J ${JOBNAME}_${task}_${seed} -o ~/.slurm_logs/${JOBNAME}_${task}_${seed}.log debs_robomimic.sh ${task} ${MODEL} ${latent} 1.0 ${JOBNAME} ${seed}
        done
    done
done