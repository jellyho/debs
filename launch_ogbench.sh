#! /bin/bash


export PART='big_suma_rtx3090'
export QOS='big_qos'
export MUJOCO_GL='egl'
export JOBNAME="jit_mf_large"
export MODEL="meanflowq"


for task_num in "1" "2" "3" "4" "5"; do
    for latent in "sphere"; do
        for task in "antmaze-large-navigate" "antmaze-giant-navigate" "humanoidmaze-medium-navigate" "humanoidmaze-large-navigate" "antsoccer-arena-navigate"; do
            for seed in "100" "200" "300"; do
                echo "${JOBNAME}_${task_num}_${latent}_${task}_${seed}"
                sbatch -p ${PART} -q ${QOS} --gres=gpu:1 -J ${JOBNAME}_${task}_${seed} -o ~/.slurm_logs/${JOBNAME}_${task}_${seed}.log debs_ogbench.sh ${task} ${MODEL} ${latent} 1.0 ${JOBNAME} ${task_num}
            done
        done
    done
done