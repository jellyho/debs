#! /bin/bash

# base_suma_rtx3090*    up 14-00:00:0      6 drain* node[02-07]
# base_suma_rtx3090*    up 14-00:00:0      3    mix node[01,10,15]
# dell_rtx3090          up 14-00:00:0      2 drain* node[22-23]
# big_suma_rtx3090      up 14-00:00:0      5 drain* node[16,18-21]
# big_suma_rtx3090      up 14-00:00:0      6    mix node[09,11-14,17]
# big_suma_rtx3090      up 14-00:00:0      1  alloc node08
# suma_A6000            up 14-00:00:0      2    mix node[25,28]
# suma_A6000            up 14-00:00:0      4  alloc node[26-27,29-30]
# suma_rtx4090          up 14-00:00:0     11  alloc cs-gpu-01,node[31-40]
# suma_a100             up 14-00:00:0      3    mix node[41-43]
# gigabyte_A6000        up 14-00:00:0      4    mix node[44,46-47,51]
# gigabyte_A6000        up 14-00:00:0      2  alloc node[45,50]
# gigabyte_A5000        up 14-00:00:0      1    mix node48
# gigabyte_A5000        up 14-00:00:0      1  alloc node49
# ASUS_RTX6000ADA       up 14-00:00:0      1 drain* node53
# TYAN_A6000            up 14-00:00:0      1 drain* node54
# ASUS_A5000            up 14-00:00:0      1 drain* node55
# dell_cpu              up 14-00:00:0      3   idle cnode[01-02,04]
# "cube-single-play" "cube-double-play" "scene-play" "puzzle-3x3-play" "puzzle-4x4-play"


export PART='suma_rtx4090,base_suma_rtx3090,big_suma_rtx3090,suma_A6000,gigabyte_A6000,gigabyte_A5000'
export QOS='base_qos'
export MUJOCO_GL='egl'
export JOBNAME="MFLDQL_VISUAL_FINAL"
export MODEL="meanflowq"

# "cube-double-play" "scene-play" "puzzle-3x3-play" "puzzle-4x4-play"; do
# "normal"
for task_num in "1"; do
    for latent in "normal"; do
        for task in "cube-single-play"; do
            for seed in "100" "200" "300"; do
                for alpha in "3.0"; do
                    echo "${JOBNAME}_${task_num}_${latent}_${task}_${seed}"
                    sbatch --exclude=node19,node16 -p ${PART} -q ${QOS} --gres=gpu:1 -J ${JOBNAME}_${task}_${seed} -o ~/.slurm_logs/${JOBNAME}_${task}_${seed}.log debs_visual_ogbench.sh ${task} ${MODEL} ${latent} ${alpha} ${JOBNAME} ${task_num} ${seed}
                done
            done
        done
    done
done

for task_num in "1"; do
    for latent in "normal"; do
        for task in "cube-double-play"; do
            for seed in "100" "200" "300"; do
                for alpha in "0.3"; do
                    echo "${JOBNAME}_${task_num}_${latent}_${task}_${seed}"
                    sbatch --exclude=node19,node16 -p ${PART} -q ${QOS} --gres=gpu:1 -J ${JOBNAME}_${task}_${seed} -o ~/.slurm_logs/${JOBNAME}_${task}_${seed}.log debs_visual_ogbench.sh ${task} ${MODEL} ${latent} ${alpha} ${JOBNAME} ${task_num} ${seed}
                done
            done
        done
    done
done
# for task_num in "1"; do
#     for latent in "normal"; do
#         for task in "cube-double-play"; do
#             for seed in "100" "200" "300"; do
#                 for alpha in "3.0"; do
#                     echo "${JOBNAME}_${task_num}_${latent}_${task}_${seed}"
#                     sbatch --exclude=node19,node16 -p W${PART} -q ${QOS} --gres=gpu:1 -J ${JOBNAME}_${task}_${seed} -o ~/.slurm_logs/${JOBNAME}_${task}_${seed}.log debs_visual_ogbench.sh ${task} ${MODEL} ${latent} ${alpha} ${JOBNAME} ${task_num} ${seed}
#                 done
#             done
#         done
#     done
# done

for task_num in "1"; do
    for latent in "normal"; do
        for task in "scene-play"; do
            for seed in "100" "200" "300"; do
                for alpha in "0.3"; do
                    echo "${JOBNAME}_${task_num}_${latent}_${task}_${seed}"
                    sbatch --exclude=node19,node16 -p ${PART} -q ${QOS} --gres=gpu:1 -J ${JOBNAME}_${task}_${seed} -o ~/.slurm_logs/${JOBNAME}_${task}_${seed}.log debs_visual_ogbench.sh ${task} ${MODEL} ${latent} ${alpha} ${JOBNAME} ${task_num} ${seed}
                done
            done
        done
    done
done

for task_num in "1"; do
    for latent in "normal"; do
        for task in "puzzle-3x3-play"; do
            for seed in "200" "300"; do
                for alpha in "10.0"; do
                    echo "${JOBNAME}_${task_num}_${latent}_${task}_${seed}"
                    sbatch --exclude=node19,node16 -p ${PART} -q ${QOS} --gres=gpu:1 -J ${JOBNAME}_${task}_${seed} -o ~/.slurm_logs/${JOBNAME}_${task}_${seed}.log debs_visual_ogbench.sh ${task} ${MODEL} ${latent} ${alpha} ${JOBNAME} ${task_num} ${seed}
                done
            done
        done
    done
done
# for task_num in "1"; do
#     for latent in "normal"; do
#         for task in "puzzle-3x3-play"; do
#             for seed in "100" "200" "300"; do
#                 for alpha in "0.3"; do
#                     echo "${JOBNAME}_${task_num}_${latent}_${task}_${seed}"
#                     sbatch --exclude=node19,node16 -p ${PART} -q ${QOS} --gres=gpu:1 -J ${JOBNAME}_${task}_${seed} -o ~/.slurm_logs/${JOBNAME}_${task}_${seed}.log debs_visual_ogbench.sh ${task} ${MODEL} ${latent} ${alpha} ${JOBNAME} ${task_num} ${seed}
#                 done
#             done
#         done
#     done
# done

for task_num in "1"; do
    for latent in "normal"; do
        for task in "puzzle-4x4-play"; do
            for seed in "100" "200" "300"; do
                for alpha in "1.0"; do
                    echo "${JOBNAME}_${task_num}_${latent}_${task}_${seed}"
                    sbatch --exclude=node19,node16 -p ${PART} -q ${QOS} --gres=gpu:1 -J ${JOBNAME}_${task}_${seed} -o ~/.slurm_logs/${JOBNAME}_${task}_${seed}.log debs_visual_ogbench.sh ${task} ${MODEL} ${latent} ${alpha} ${JOBNAME} ${task_num} ${seed}
                done
            done
        done
    done
done
##############################################################################