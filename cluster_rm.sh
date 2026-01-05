#! /bin/bash


export PART='big_suma_rtx3090'
export QOS='big_qos'
export MUJOCO_GL='egl'
export JOBNAME="jit_mf_time_dist"
export MODEL="meanflow"


for time_dist in "log_norm" "beta" "discrete" "uniform"; do
    for time_r_zero in "true"; do
        for latent in "sphere"; do
            for task in "can"; do
                for seed in "100"; do
                    echo "${JOBNAME}_${task_num}_${latent}_${task}_${seed}"
                    sbatch -p ${PART} -q ${QOS} --gres=gpu:1 -J ${JOBNAME}_${task}_${seed} -o ~/.slurm_logs/${JOBNAME}_${task}_${seed}.log debs_robomimic.sh ${task} ${MODEL} ${latent} 1.0 ${JOBNAME} ${time_dist} ${time_r_zero}
                done
            done
        done
    done
done