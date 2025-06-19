unset CUDA_VISIBLE_DEVICES
# Define the list of properties
# props=("mbj_bandgap" "Tc_supercon")
props=("Tc_supercon")
# Loop through each property in the list
for prop in "${props[@]}"; do 
    # python dl_regressors_torch.py --config_file ./sample/example_alignn_"$prop"_local.config
    # python dl_regressors_torch.py --config_file ./sample/example_alignn_matbert-base-cased_robo_prop_"$prop"_local.config
    python dl_regressors_torch.py --config_file ./sample/example_alignn_matbert-base-cased_robo_prop_"$prop".config
done


for prop in "${props[@]}"; do 
    python dl_regressors_torch_predict.py --config_file ./sample/example_alignn_matbert-base-cased_robo_prop_mbj_bandgap.config --model_path model/alignn_matbert-base-cased_robo_prop_mbj_bandgap/model_1024Rx4D-512Rx3D-256Rx3D-128Rx3D-64Rx2-32Rx1-1_20250614_193231.pt    
done

