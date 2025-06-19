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
# props=("mbj_bandgap")  # Add more if needed

# Training
# for prop in "${props[@]}"; do 
#     python dl_regressors_torch.py --config_file ./sample/example_alignn_matbert-base-cased_robo_prop_"$prop"_local.config    
# done

# Inference
# for prop in "${props[@]}"; do 
#     # cpu trained model
#     model_path="/scratch/yll6162/CrossPropertyTL/elemnet/model/alignn_matbert-base-cased_robo_prop_mbj_bandgap/model_1024Rx4D-512Rx3D-256Rx3D-128Rx3D-64Rx2-32Rx1-1.pt"
#     python dl_regressors_torch_predict.py \
#         --config_file ./sample/example_alignn_matbert-base-cased_robo_prop_"$prop"_local.config \
#         --model_path "$model_path"
#     # gpu trained model
#     model_path="model/alignn_matbert-base-cased_robo_prop_${prop}/model_1024Rx4D-512Rx3D-256Rx3D-128Rx3D-64Rx2-32Rx1-1_20250614_184558.pt"
#     python dl_regressors_torch_predict.py \
#         --config_file ./sample/example_alignn_matbert-base-cased_robo_prop_"$prop".config \
#         --model_path "$model_path"


# done

prop="mbj_bandgap"
model_paths=(
  "./models/leia/gpu/alignn_matbert-base-cased_robo_prop_mbj_bandgap/model_1024Rx4D-512Rx3D-256Rx3D-128Rx3D-64Rx2-32Rx1-1_20250614_184558.pt"
  "./models/leia/cpu/alignn_matbert-base-cased_robo_prop_mbj_bandgap/model_1024Rx4D-512Rx3D-256Rx3D-128Rx3D-64Rx2-32Rx1-1_20250614_193231.pt"
  "./models/luke/gpu/alignn_matbert-base-cased_robo_prop_mbj_bandgap/model_1024Rx4D-512Rx3D-256Rx3D-128Rx3D-64Rx2-32Rx1-1_20250619_152657.pt"
  "./models/luke/cpu/alignn_matbert-base-cased_robo_prop_mbj_bandgap/model_1024Rx4D-512Rx3D-256Rx3D-128Rx3D-64Rx2-32Rx1-1_20241103_131729.pt"
  "./models/colab/gpu/alignn_matbert-base-cased_robo_prop_mbj_bandgap/model_1024Rx4D-512Rx3D-256Rx3D-128Rx3D-64Rx2-32Rx1-1_20250616_015946.pt"
  "./models/colab/cpu/alignn_matbert-base-cased_robo_prop_mbj_bandgap/model_1024Rx4D-512Rx3D-256Rx3D-128Rx3D-64Rx2-32Rx1-1_20250615_200123.pt"
)


# Loop through each path
for model_path in "${model_paths[@]}"; do
    echo "Processing: $model_path"
    python dl_regressors_torch_predict.py \
        --config_file ./sample/example_alignn_matbert-base-cased_robo_prop_"$prop"_local.config \
        --model_path "$model_path"
done