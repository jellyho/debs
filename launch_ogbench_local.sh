#! /bin/bash


# export PART='big_suma_rtx3090'
# export QOS='big_qos'
export MUJOCO_GL='egl'
export JOBNAME="QCFQL_FINAL"
export MODEL="qcfql"


for task_num in "1" "2" "3" "4" "5"; do
    for latent in "normal" "sphere"; do
        for task in "cube-double-play"; do
            for seed in "100" "200" "300"; do
                echo "${JOBNAME}_${task_num}_${latent}_${task}_${seed}"
                sh debs_ogbench.sh ${task} ${MODEL} ${latent} 3.0 ${JOBNAME} ${task_num} ${seed}
            done
        done
    done
done
