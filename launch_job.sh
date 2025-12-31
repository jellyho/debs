#! /bin/bash

export PART='big_suma_rtx3090'
export QOS='big_qos'
export MUJOCO_GL='egl'
export JOBNAME="QCFQL_RM"
export MODEL="qcfql"


for alpha in "100.0" "1000.0" "10000.0"; do
    for latent in "sphere" "normal"; do
        for task in "lift" "can" "square"; do
            for seed in "100" "200" "300"; do
                echo "${JOBNAME}_${alpha}_${latent}_${task}_${seed}"
                sbatch -p ${PART} -q ${QOS} --gres=gpu:1 -J ${JOBNAME}_${task}_${seed} -o ~/.slurm_logs/${JOBNAME}_${task}_${seed}.log debs_robomimic.sh ${task} ${MODEL} ${latent} ${alpha} ${JOBNAME}
            done
        done
    done
done