#! /bin/bash


export PART='base_suma_rtx3090'
export QOS='base_qos'
export MUJOCO_GL='egl'
export JOBNAME="jit_mf_time_dist"
export MODEL="meanflow"


for latent in "sphere"; do
    for task in "lift" "square" "can"; do
        for seed in "100"; do
            echo "${JOBNAME}_${task_num}_${latent}_${task}_${seed}"
            sbatch -p ${PART} -q ${QOS} --gres=gpu:1 -J ${JOBNAME}_${task}_${seed} -o ~/.slurm_logs/${JOBNAME}_${task}_${seed}.log debs_robomimic.sh ${task} ${MODEL} ${latent} 1.0 ${JOBNAME}
        done
    done
done