#! /bin/bash


export PART='base_suma_rtx3090,big_suma_rtx3090,suma_rtx4090,suma_A6000,gigabyte_A6000,gigabyte_A5000'
export QOS='big_qos'
export MUJOCO_GL='egl'
export JOBNAME="BANDIT_DSRL_DEMO"
# export MODEL="meanflow"


for model in "meanflowq"; do
    # for mf_method in "jit_mf_nor" "mfql_nor" "imf" "jit_mf" "mfql"; do
    for latent in "sphere"; do
        for num in "4"; do
            for seed in "100"; do
                for alpha in "100.0" "10.0" "1.0" "0.1"; do
                # echo "${JOBNAME}_${task_num}_${latent}_${task}_${seed}"
                    sh bandit_debs.sh ${num} ${model} ${latent} ${alpha} ${JOBNAME}
                done
            done
        done
    done
done
