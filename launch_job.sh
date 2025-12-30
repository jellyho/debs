#! /bin/bash

export PART='big_suma_rtx3090'
export QOS='big_qos'
export MUJOCO_GL='egl'
export JOBNAME="LQLTUNE"
export MODEL="meanflow"


for alpha in "1.0"; do
    for latent in "sphere" "normal"; do
        for task in "lift" "can" "square" "transport"; do
            for seed in "100" "200" "300" "400" "500"; do
                echo "${JOBNAME}_${alpha}_${latent}_${task}_${seed}"
                sbatch -p ${PART} -q ${QOS} --gres=gpu:1 -J ${JOBNAME}_${task}_${seed} -o ~/.slurm_logs/${JOBNAME}_${task}_${seed}.log debs_robomimic.sh ${task} ${MODEL} ${latent} ${alpha}
            done
        done
    done
done