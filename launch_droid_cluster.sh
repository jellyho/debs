#! /bin/bash


export PART='asus_pro6000,suma_a6000,gigabyte_a6000,tyan_a6000,asus_6000ada'
export QOS='big_qos'
export MUJOCO_GL='egl'
export JOBNAME="droid_new"
export data_dir="/scratch/jellyho/"
export save_dir="/scratch/jellyho/lql_droid_new_zeroone/"
# export MODEL="meanflow"
# export lahd="(256,256)"
# export critic_agg="min"
# export extract_method="ddpg"


# for task in "droid_pnp_carrot" "droid_refill_tape" "droid_eggplant_bin" "droid_plug_in_bulb"; do
# for task in "droid_eggplant_bin"; do
for task in "droid_pnp_carrot" "droid_refill_tape" "droid_eggplant_bin" "droid_plug_in_bulb"; do
    for model_name in "meanflowq" "dsrl"; do
        for seed in "100"; do
            echo "${JOBNAME}_${task}_${seed}"
            sbatch --exclude=node19,node16 -p ${PART} -q ${QOS} --gres=gpu:1 -J ${JOBNAME}_${task}_${seed} -o ~/.slurm_logs/${JOBNAME}_${task}_${seed}.log debs_droid.sh ${task} ${model_name} 1.0 1.0 ${JOBNAME} ${seed} ${data_dir} ${save_dir}
        done
    done
done