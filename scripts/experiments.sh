#!/bin/bash

cd ..

declare -a AlgList=("Proposed")
declare -a BackboneList=("Resnet34")
declare -a DatasetList=("office_home")
declare -a DASettingList=("openpartial")

txt="txt"
sampling="kshot"
num_shots=3

for da_method in ${AlgList[@]}; do
    for backbone in ${BackboneList[@]}; do
        for dataset in ${DatasetList[@]}; do
            for da_setting in ${DASettingList[@]}; do
                expt_description="expt_run-${txt}-${backbone}-${dataset}-${da_setting}"
                run_description="expt-${da_method}-${sampling}-${num_shots}"
                python main.py  --experiment_description $expt_description  \
                                --run_description $run_description \
                                --da_setting $da_setting \
                                --da_method $da_method \
                                --dataset $dataset \
                                --backbone $backbone \
                                --num_seeds 3 \
                                --sampling $sampling \
                                --num_shots $num_shots \
                                --data_path "./data/${txt}" \
                                --data_root "./data" \
                                --wandb_dir "results/${expt_description}" \
                                --wandb_project $expt_description \
                                --wandb_entity ssda \
                                --wandb_tag debug
            done
        done
    done
done