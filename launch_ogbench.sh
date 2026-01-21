#! /bin/bash


export PART='base_suma_rtx3090,big_suma_rtx3090,suma_rtx4090,suma_A6000,gigabyte_A6000,gigabyte_A5000'
export QOS='big_qos'
export MUJOCO_GL='egl'
export JOBNAME="mflql_debug"
export MODEL="meanflow"
export lahd="(256,256)"
export critic_agg="min"
export extract_method="ddpg"


for task_num in "3" "4" "5"; do
    for latent in "sphere"; do
        for task in "cube-single-play"; do
            for seed in "100"; do
                echo "${JOBNAME}_${task_num}_${latent}_${task}_${seed}"
                sbatch -p ${PART} -q ${QOS} --exclude=node19,node08,node16 --gres=gpu:1 -J ${JOBNAME}_${task}_${seed} -o ~/.slurm_logs/${JOBNAME}_${task}_${seed}.log debs_ogbench.sh ${task} ${MODEL} ${latent} 1.0 ${JOBNAME} ${task_num} ${seed} ${lahd} ${critic_agg} ${extract_method}
            done
        done
    done
done