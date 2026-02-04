#! /bin/bash


# export PART='big_suma_rtx3090'
# export QOS='big_qos'
export MUJOCO_GL='egl'
export JOBNAME="LPS_ONLINE"
export MODEL="meanflowq"


for task_num in "1" "2" "3" "4" "5"; do
    for latent in "sphere"; do
        for task in "cube-double-play" "puzzle-4x4-play"; do
            for seed in "$1"; do
                echo "${JOBNAME}_${task_num}_${latent}_${task}_${seed}"
                sh debs_ogbench_online.sh ${task} ${MODEL} ${latent} 1.0 ${JOBNAME} ${task_num} ${seed} "(256,256)" "ddpg"
            done
        done
    done
done
