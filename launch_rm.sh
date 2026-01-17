#! /bin/bash


export PART='base_suma_rtx3090,big_suma_rtx3090,suma_rtx4090,suma_A6000,gigabyte_A6000,gigabyte_A5000'
export QOS='big_qos'
export MUJOCO_GL='egl'
export JOBNAME="MFQ_RM_LQL_SMALL_MODEL"
export MODEL="meanflowq"

for mf_method in "jit_mf"; do
    for latent in "sphere"; do
        for task in "can" "lift" "square"; do
            for seed in "100" "200" "300"; do
                echo "${JOBNAME}_${task}_${latent}_${task}_${seed}"
                sbatch --exclude=node19,node16 -p ${PART} -q ${QOS} --gres=gpu:1 -J ${JOBNAME}_${task}_${seed} -o ~/.slurm_logs/${JOBNAME}_${task}_${seed}.log debs_robomimic.sh ${task} ${MODEL} ${latent} 1.0 ${JOBNAME} ${seed} ${mf_method}
            done
        done
    done
done