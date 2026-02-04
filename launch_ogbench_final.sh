#! /bin/bash


export PART='base_suma_rtx3090,big_suma_rtx3090,suma_rtx4090,suma_a6000,gigabyte_a6000,asus_6000ada,tyan_a6000,gigabyte_a5000,dell_rtx3090'
export QOS='big_qos'
export MUJOCO_GL='egl'
export JOBNAME="CFGRL_FINAL"
export MODEL="cfgrl"


export TASK='cube-single-play'
export cfg="1.25"
export LATENT='normal'

for task_num in "1" "2" "3" '4' "5"; do
    for seed in "100" "200" "300"; do
        echo "${JOBNAME}_${task_num}_${LATENT}_${task}_${seed}"
        sbatch --exclude=node19,node16 -p ${PART} -q ${QOS} --gres=gpu:1 -o ~/.slurm_logs/${JOBNAME}_${TASK}_${task_num}_${seed}.log cfgrl_ogbench.sh ${TASK} ${MODEL} ${LATENT} ${cfg} ${JOBNAME} ${task_num} ${seed}
    done
done

export TASK='cube-double-play'  
export cfg="2.0"
export LATENT='normal'

for task_num in "1" "2" "3" '4' "5"; do
    for seed in "100" "200" "300"; do
        echo "${JOBNAME}_${task_num}_${LATENT}_${task}_${seed}"
        sbatch --exclude=node19,node16 -p ${PART} -q ${QOS} --gres=gpu:1 -o ~/.slurm_logs/${JOBNAME}_${TASK}_${task_num}_${seed}.log cfgrl_ogbench.sh ${TASK} ${MODEL} ${LATENT} ${cfg} ${JOBNAME} ${task_num} ${seed}
    done
done

export TASK='scene-play'
export cfg="3.0"
export LATENT='normal'

for task_num in "1" "2" "3" "4" "5"; do
    for seed in "100" "200" "300"; do
        echo "${JOBNAME}_${task_num}_${LATENT}_${task}_${seed}"
        sbatch --exclude=node19,node16 -p ${PART} -q ${QOS} --gres=gpu:1 -o ~/.slurm_logs/${JOBNAME}_${TASK}_${task_num}_${seed}.log cfgrl_ogbench.sh ${TASK} ${MODEL} ${LATENT} ${cfg} ${JOBNAME} ${task_num} ${seed}
    done
done

export TASK='puzzle-3x3-play'
export cfg="1.5"
export LATENT='normal'

for task_num in "1" "2" "3" '4' "5"; do
    for seed in "100" "200" "300"; do
        echo "${JOBNAME}_${task_num}_${LATENT}_${task}_${seed}"
        sbatch --exclude=node19,node16 -p ${PART} -q ${QOS} --gres=gpu:1 -o ~/.slurm_logs/${JOBNAME}_${TASK}_${task_num}_${seed}.log cfgrl_ogbench.sh ${TASK} ${MODEL} ${LATENT} ${cfg} ${JOBNAME} ${task_num} ${seed}
    done
done

export TASK='puzzle-4x4-play'
export cfg="1.25"
export LATENT='normal'

for task_num in "1" "2" "3" '4' "5"; do
    for seed in "100" "200" "300"; do
        echo "${JOBNAME}_${task_num}_${LATENT}_${task}_${seed}"
        sbatch --exclude=node19,node16 -p ${PART} -q ${QOS} --gres=gpu:1 -o ~/.slurm_logs/${JOBNAME}_${TASK}_${task_num}_${seed}.log cfgrl_ogbench.sh ${TASK} ${MODEL} ${LATENT} ${cfg} ${JOBNAME} ${task_num} ${seed}
    done
done