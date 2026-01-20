#! /bin/bash


export PART='big_suma_rtx3090'
export QOS='big_qos'
export MUJOCO_GL='egl'
export JOBNAME="mfldql_debug"
export MODEL="mfldql"


for task_num in "1" "2" "3" "4" "5"; do
    for latent in "sphere"; do
        for task in "cube-single-play" "cube-double-play" "scene-play"; do
            for seed in "100"; do
                echo "${JOBNAME}_${task_num}_${latent}_${task}_${seed}"
                sbatch -p ${PART} -q ${QOS} --exclude=node19,node08,node16 --gres=gpu:1 -J ${JOBNAME}_${task}_${seed} -o ~/.slurm_logs/${JOBNAME}_${task}_${seed}.log debs_ogbench.sh ${task} ${MODEL} ${latent} 0.1 ${JOBNAME} ${task_num} ${seed}
            done
        done
    done
done