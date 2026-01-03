#! /bin/bash


export PART='base_suma_rtx3090'
export QOS='base_qos'
export MUJOCO_GL='egl'
export JOBNAME="jit_mf_time_dist_log_norm_rzero"
export MODEL="meanflow"

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# (선택 사항) 디버깅을 위해 경로 확인
echo "Current Conda Prefix: $CONDA_PREFIX"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

for latent in "sphere"; do
    for task in "lift" "square" "can"; do
        for seed in "100"; do
            echo "${JOBNAME}_${task_num}_${latent}_${task}_${seed}"
            sbatch --exclude=node100 --gres=gpu:1 -J ${JOBNAME}_${task}_${seed} -o ~/.slurm_logs/${JOBNAME}_${task}_${seed}.log debs_robomimic.sh ${task} ${MODEL} ${latent} 1.0 ${JOBNAME}
        done
    done
done