#! /bin/bash


export PART='base_suma_rtx3090,big_suma_rtx3090,suma_rtx4090,suma_a6000,gigabyte_a6000,asus_6000ada,tyan_a6000,gigabyte_a5000,dell_rtx3090'
export QOS='big_qos'
export MUJOCO_GL='egl'
export JOBNAME="mflql_reform"
export MODEL="meanflowq"
export lahd="(256,256)"
export critic_agg="min"
export extract_method="ddpg"

#  "puzzle-3x3-play"
for task_num in "1"; do
    for latent in "truncated_normal"; do
        for task in "cube-single-play"; do
            for alpha in "1.0"; do
                for seed in "100"; do
                    echo "${JOBNAME}_${task_num}_${latent}_${task}_${seed}"
                    sbatch -p ${PART} -q ${QOS} --exclude=node19,node08,node16 --gres=gpu:1 -J ${JOBNAME}_${task}_${seed} -o ~/.slurm_logs/${JOBNAME}_${task}_${seed}.log debs_ogbench.sh ${task} ${MODEL} ${latent} ${alpha} ${JOBNAME} ${task_num} ${seed} ${lahd} ${critic_agg} ${extract_method}
                done
            done
        done
    done
done