#!/bin/bash

# Number of times to run the command
n=8  # Replace 5 with the desired number of iterations

# Loop to run the command n times
for ((i=1; i<=n; i++))
do
    echo "Running iteration $i of $n"
    # python dl_regressors_tf2.py --config_file sample/example_alignn_matbert-base-cased_robo_prop_mbj_bandgap.config

    python dl_regressors_tf2.py --config_file /scratch/yll6162/atomgpt_pub/ALIGNN-BERT-TL-crystal/CrossPropertyTL/elemnet/sample/example_alignn_mbj_bandgap.config

    
done

echo "Completed $n iterations."
