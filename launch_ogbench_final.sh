#! /bin/bash


export PART='base_suma_rtx3090,big_suma_rtx3090,suma_rtx4090,suma_A6000,gigabyte_A6000,gigabyte_A5000'
export QOS='big_qos'
export MUJOCO_GL='egl'
export JOBNAME="QCFQL_FINAL"
export MODEL="qcfql"


export TASK='cube-single-play'
# export ALPHA="30.0"
# export LATENT='normal'

# for task_num in "1" "2" "3" '4' "5"; do
#     for seed in "100" "200" "300"; do
#         echo "${JOBNAME}_${task_num}_${LATENT}_${task}_${seed}"
#         sbatch -p ${PART} -q ${QOS} --gres=gpu:1 -o ~/.slurm_logs/${JOBNAME}_${TASK}_${task_num}_${seed}.log debs_ogbench.sh ${TASK} ${MODEL} ${LATENT} ${ALPHA} ${JOBNAME} ${task_num} ${seed}
#     done
# done
export ALPHA="10.0"
export LATENT='sphere'

for task_num in "3"; do
    for seed in "100" "200" "300"; do
        echo "${JOBNAME}_${task_num}_${LATENT}_${task}_${seed}"
        sbatch -p ${PART} -q ${QOS} --gres=gpu:1 -o ~/.slurm_logs/${JOBNAME}_${TASK}_${task_num}_${seed}.log debs_ogbench.sh ${TASK} ${MODEL} ${LATENT} ${ALPHA} ${JOBNAME} ${task_num} ${seed}
    done
done

export TASK='cube-double-play'
export ALPHA="3.0"
export LATENT='normal'

for task_num in "1" "2" "3" '4' "5"; do
    for seed in "100" "200" "300"; do
        echo "${JOBNAME}_${task_num}_${LATENT}_${task}_${seed}"
        sbatch -p ${PART} -q ${QOS} --gres=gpu:1 -o ~/.slurm_logs/${JOBNAME}_${TASK}_${task_num}_${seed}.log debs_ogbench.sh ${TASK} ${MODEL} ${LATENT} ${ALPHA} ${JOBNAME} ${task_num} ${seed}
    done
done
export ALPHA="3.0"
export LATENT='sphere'

for task_num in "3"; do
    for seed in "100" "200" "300"; do
        echo "${JOBNAME}_${task_num}_${LATENT}_${task}_${seed}"
        sbatch -p ${PART} -q ${QOS} --gres=gpu:1 -o ~/.slurm_logs/${JOBNAME}_${TASK}_${task_num}_${seed}.log debs_ogbench.sh ${TASK} ${MODEL} ${LATENT} ${ALPHA} ${JOBNAME} ${task_num} ${seed}
    done
done

export TASK='scene-play'
export ALPHA="1.0"
export LATENT='normal'

for task_num in '4'; do
    for seed in "100" "200" "300"; do
        echo "${JOBNAME}_${task_num}_${LATENT}_${task}_${seed}"
        sbatch -p ${PART} -q ${QOS} --gres=gpu:1 -o ~/.slurm_logs/${JOBNAME}_${TASK}_${task_num}_${seed}.log debs_ogbench.sh ${TASK} ${MODEL} ${LATENT} ${ALPHA} ${JOBNAME} ${task_num} ${seed}
    done
done
# export ALPHA="3.0"
# export LATENT='sphere'

# for task_num in "1" "2" "3" '4' "5"; do
#     for seed in "100" "200" "300"; do
#         echo "${JOBNAME}_${task_num}_${LATENT}_${task}_${seed}"
#         sbatch -p ${PART} -q ${QOS} --gres=gpu:1 -o ~/.slurm_logs/${JOBNAME}_${TASK}_${task_num}_${seed}.log debs_ogbench.sh ${TASK} ${MODEL} ${LATENT} ${ALPHA} ${JOBNAME} ${task_num} ${seed}
#     done
# done

# export TASK='puzzle-3x3-play'
# export ALPHA="3.0"
# export LATENT='normal'

# for task_num in "1" "2" "3" '4' "5"; do
#     for seed in "100" "200" "300"; do
#         echo "${JOBNAME}_${task_num}_${LATENT}_${task}_${seed}"
#         sbatch -p ${PART} -q ${QOS} --gres=gpu:1 -o ~/.slurm_logs/${JOBNAME}_${TASK}_${task_num}_${seed}.log debs_ogbench.sh ${TASK} ${MODEL} ${LATENT} ${ALPHA} ${JOBNAME} ${task_num} ${seed}
#     done
# done
# export ALPHA="3.0"
# export LATENT='sphere'

# for task_num in "1" "2" "3" '4' "5"; do
#     for seed in "100" "200" "300"; do
#         echo "${JOBNAME}_${task_num}_${LATENT}_${task}_${seed}"
#         sbatch -p ${PART} -q ${QOS} --gres=gpu:1 -o ~/.slurm_logs/${JOBNAME}_${TASK}_${task_num}_${seed}.log debs_ogbench.sh ${TASK} ${MODEL} ${LATENT} ${ALPHA} ${JOBNAME} ${task_num} ${seed}
#     done
# done

# export TASK='puzzle-4x4-play'
# export ALPHA="10.0"
# export LATENT='normal'

# for task_num in "1" "2" "3" '4' "5"; do
#     for seed in "100" "200" "300"; do
#         echo "${JOBNAME}_${task_num}_${LATENT}_${task}_${seed}"
#         sbatch -p ${PART} -q ${QOS} --gres=gpu:1 -o ~/.slurm_logs/${JOBNAME}_${TASK}_${task_num}_${seed}.log debs_ogbench.sh ${TASK} ${MODEL} ${LATENT} ${ALPHA} ${JOBNAME} ${task_num} ${seed}
#     done
# done
# export ALPHA="3.0"
# export LATENT='sphere'

# for task_num in "1" "2" "3" '4' "5"; do
#     for seed in "100" "200" "300"; do
#         echo "${JOBNAME}_${task_num}_${LATENT}_${task}_${seed}"
#         sbatch -p ${PART} -q ${QOS} --gres=gpu:1 -o ~/.slurm_logs/${JOBNAME}_${TASK}_${task_num}_${seed}.log debs_ogbench.sh ${TASK} ${MODEL} ${LATENT} ${ALPHA} ${JOBNAME} ${task_num} ${seed}
#     done
# done