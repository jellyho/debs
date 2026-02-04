#! /bin/bash


export PART='base_suma_rtx3090,big_suma_rtx3090,suma_rtx4090,suma_a6000,gigabyte_a6000,gigabyte_a5000'
export QOS='big_qos'
export MUJOCO_GL='egl'
export JOBNAME="BANDIT_test"
# export MODEL="meanflow"


for model in "cfgrl"; do
    # for mf_method in "jit_mf_nor" "mfql_nor" "imf" "jit_mf" "mfql"; do
    for latent in "normal"; do
        for num in "1" "2" "3" "4"; do
            for seed in "100"; do
                for alpha in "0.1"; do
                    # echo "${JOBNAME}_${task_num}_${latent}_${task}_${seed}"
                    sbatch -p ${PART} -q ${QOS} --gres=gpu:1 -J ${JOBNAME}_${seed}_${model}_${alpha} -o ~/.slurm_logs/${JOBNAME}_${seed}.log bandit_debs.sh ${num} ${model} ${latent} ${alpha} ${JOBNAME}
                done
            done
        done
    done
done
