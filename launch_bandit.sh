#! /bin/bash


export PART='base_suma_rtx3090,big_suma_rtx3090,suma_rtx4090,suma_A6000,gigabyte_A6000,gigabyte_A5000'
export QOS='big_qos'
export MUJOCO_GL='egl'
export JOBNAME="BANDIT_CFM_TARGET_NEW"
# export MODEL="meanflow"


for model in "qcfql"; do
    # for mf_method in "jit_mf_nor" "mfql_nor" "imf" "jit_mf" "mfql"; do
    for latent in "normal"; do
        for alpha in "0.01" "0.1" "1.0" "10.0" "100.0"; do
            for num in "1"; do
                for seed in "100"; do
                    # echo "${JOBNAME}_${task_num}_${latent}_${task}_${seed}"
                    sbatch -p ${PART} -q ${QOS} --gres=gpu:1 -J ${JOBNAME}_${seed} -o ~/.slurm_logs/${JOBNAME}_${seed}.log bandit_debs.sh ${num} ${model} ${latent} ${alpha} ${JOBNAME}
                done
            done
        done
    done
done