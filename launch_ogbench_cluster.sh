#! /bin/bash

# base_suma_rtx3090*    up 14-00:00:0      7    mix node[01,03-06,10,15]
# base_suma_rtx3090*    up 14-00:00:0      1  alloc node02
# base_suma_rtx3090*    up 14-00:00:0      1   idle node07
# dell_rtx3090          up 14-00:00:0      2   idle node[22-23]
# big_suma_rtx3090      up 14-00:00:0      1 drain* node21
# big_suma_rtx3090      up 14-00:00:0     10    mix node[09,11-14,16-20]
# big_suma_rtx3090      up 14-00:00:0      1  alloc node08
# suma_A6000            up 14-00:00:0      2    mix node[25,30]
# suma_A6000            up 14-00:00:0      4  alloc node[26-29]
# suma_rtx4090          up 14-00:00:0      2    mix cs-gpu-01,node36
# suma_rtx4090          up 14-00:00:0      9  alloc node[31-35,37-40]
# suma_a100             up 14-00:00:0      2    mix node[42-43]
# suma_a100             up 14-00:00:0      1  alloc node41
# gigabyte_A6000        up 14-00:00:0      5    mix node[44,46-47,50-51]
# gigabyte_A6000        up 14-00:00:0      1  alloc node45
# gigabyte_A5000        up 14-00:00:0      2    mix node[48-49]
# ASUS_RTX6000ADA       up 14-00:00:0      1  alloc node52
# TYAN_A6000            up 14-00:00:0      1  alloc node53
# ASUS_A5000            up 14-00:00:0      1   idle node54
# dell_cpu              up 14-00:00:0      1    mix cnode01
# dell_cpu              up 14-00:00:0      2   idle cnode[02,04]
# "cube-single-play" "cube-double-play" "scene-play" "puzzle-3x3-play" "puzzle-4x4-play"


export PART='base_suma_rtx3090,big_suma_rtx3090,suma_rtx4090,suma_A6000,gigabyte_A6000,ASUS_RTX6000ADA,TYAN_A6000,gigabyte_A5000,ASUS_A5000'
export QOS='big_qos'
export MUJOCO_GL='egl'
export JOBNAME="ACFQLFINAL_BIG"
export MODEL="qclql"

for task_num in "1" "2" "3" "4" "5"; do
    for latent in "sphere" "normal"; do
        for task in "cube-single-play" "cube-double-play" "scene-play" "puzzle-3x3-play" "puzzle-4x4-play"; do
            for seed in "100" "200" "300"; do
                echo "${JOBNAME}_${task_num}_${latent}_${task}_${seed}"
                sbatch -p ${PART} -q ${QOS} --gres=gpu:1 -J ${JOBNAME}_${task}_${seed} -o ~/.slurm_logs/${JOBNAME}_${task}_${seed}.log debs_ogbench.sh ${task} ${MODEL} ${latent} ${alpha} ${JOBNAME} ${task_num} ${seed}
            done
        done
    done
done

# for latent in "sphere"; do
#     for task in "lift" "can" "square"; do
#         for seed in "100"; do
#             echo "${JOBNAME}_${task_num}_${latent}_${task}_${seed}"
#             sbatch -p ${PART} -q ${QOS} --gres=gpu:1 -J ${JOBNAME}_${task}_${seed} -o ~/.slurm_logs/${JOBNAME}_${task}_${seed}.log debs_robomimic.sh ${task} ${MODEL} ${latent} 1.0 ${JOBNAME} ${seed}
#         done
#     done
# done