#! /bin/bash


# export PART='big_suma_rtx3090'
# export QOS='big_qos'
export PART='asus_pro6000,base_suma_rtx3090,big_suma_rtx3090,suma_rtx4090,suma_a6000,gigabyte_a6000,asus_6000ada,tyan_a6000,gigabyte_a5000,asus_a5000'
export QOS='big_qos'

export MUJOCO_GL='egl'
export JOBNAME="LPS_ONLINE"
export MODEL="meanflowq"


for model in "meanflowq" "dsrl"; do
    for task_num in "1" "2" "3" "4" "5"; do
        for latent in "sphere"; do
            for task in "cube-double-play" "puzzle-4x4-play"; do
                for seed in "100" "200" "300"; do
                    echo "${JOBNAME}_${task_num}_${latent}_${task}_${seed}"
                    sbatch -p ${PART} -q ${QOS} --exclude=node19,node08,node16 --gres=gpu:1 -J ${JOBNAME}_${task}_${seed} -o ~/.slurm_logs/${JOBNAME}_${task}_${seed}.log debs_ogbench_online.sh ${task} ${model} ${latent} 1.0 ${JOBNAME} ${task_num} ${seed} "ddpg"
                done
            done
        done
    done
done
