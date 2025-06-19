#!/bin/bash

# Default device is GPU
device="gpu"

# Parse device argument
if [[ "$1" == "-c" ]]; then
    export CUDA_VISIBLE_DEVICES="-1"
    device="cpu"
    echo "Running on CPU only (CUDA disabled)"
else
    unset CUDA_VISIBLE_DEVICES
    device="gpu"
    echo "Running on GPU (CUDA enabled)"
fi

# Define the list of properties
# props=("mbj_bandgap" "Tc_supercon")
props=("mbj_bandgap")  # Add more if needed

# Training
# for prop in "${props[@]}"; do 
#     python dl_regressors_torch.py --config_file ./sample/example_alignn_matbert-base-cased_robo_prop_"$prop".config    
# done

# Inference
for prop in "${props[@]}"; do 
    if [[ "$device" == "cpu" ]]; then
        model_path="model/alignn_matbert-base-cased_robo_prop_${prop}/model_1024Rx4D-512Rx3D-256Rx3D-128Rx3D-64Rx2-32Rx1-1_20250614_193231.pt"
    else
        model_path="model/alignn_matbert-base-cased_robo_prop_${prop}/model_1024Rx4D-512Rx3D-256Rx3D-128Rx3D-64Rx2-32Rx1-1_20250614_184558.pt"
    fi

    python dl_regressors_torch_predict.py \
        --config_file ./sample/example_alignn_matbert-base-cased_robo_prop_"$prop".config \
        --model_path "$model_path"
done