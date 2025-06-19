# Set CUDA to use CPU only
export CUDA_VISIBLE_DEVICES="-1"
# conda activate tl_torch
# Define the list of properties
# props=("mbj_bandgap" "Tc_supercon")
props=("mbj_bandgap")
# Loop through each property in the list
for prop in "${props[@]}"; do 
    # python dl_regressors_torch.py --config_file ./sample/example_alignn_"$prop"_local.config
    # python dl_regressors_torch.py --config_file ./sample/example_alignn_matbert-base-cased_robo_prop_"$prop"_local.config
    python dl_regressors_torch.py --config_file ./sample/example_alignn_matbert-base-cased_robo_prop_"$prop".config    
done
