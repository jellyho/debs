#! /bin/bash


# export PART='big_suma_rtx3090'
# export QOS='big_qos'
export MUJOCO_GL='egl'
export JOBNAME="QCFQL_TUNE"
export MODEL="qcfql"



for latent in "sphere" "normal"; do
    for task in "cube-double-play"; do
        for alpha in "0.01" "0.03" "0.1" "0.3" "1.0" "3.0" "10.0" "30.0" "100.0" "300.0"; do
            for seed in "100"; do
                echo "${JOBNAME}_${task_num}_${latent}_${task}_${seed}"
                sh debs_ogbench_tune.sh ${task} ${MODEL} ${latent} ${alpha} ${JOBNAME} ${seed}
            done
        done
    done
done
