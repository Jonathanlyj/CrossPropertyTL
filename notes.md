

python dl_regressors_tf2.py --config_file ./sample/alignn_bert-base-uncased_chemnlp_prop_formation_energy_peratom.config; python dl_regressors_tf2.py --config_file ./sample/bert-base-uncased_robo_prop_formation_energy_peratom.config; python dl_regressors_tf2.py --config_file ./sample/alignn_bert-base-uncased_robo_prop_formation_energy_peratom.config






python dl_regressors_tf2.py --config_file /scratch/yll6162/CrossPropertyTL/elemnet/sample/matbert-base-cased_robo_prop_formation_energy_peratom.config;

python dl_regressors_tf2.py --config_file /scratch/yll6162/CrossPropertyTL/elemnet/sample/matbert-base-cased_chemnlp_prop_formation_energy_peratom.config;python dl_regressors_tf2.py --config_file /scratch/yll6162/CrossPropertyTL/elemnet/sample/alignn_matbert-base-cased_robo_prop_formation_energy_peratom.config; python dl_regressors_tf2.py --config_file /scratch/yll6162/CrossPropertyTL/elemnet/sample/alignn_matbert-base-cased_chemnlp_prop_formation_energy_peratom.config



python dl_regressors_tf2.py --config_file /scratch/yll6162/CrossPropertyTL/elemnet/sample/alignn_bert-base-uncased_robo_prop_ehull.config;python dl_regressors_tf2.py --config_file /scratch/yll6162/CrossPropertyTL/elemnet/sample/bert-base-uncased_chemnlp_prop_formation_energy_peratom.config;



python dl_regressors_tf2.py --config_file /scratch/yll6162/CrossPropertyTL/elemnet/sample/alignn_prop_magmom_outcar.config;

python dl_regressors_tf2.py --config_file /scratch/yll6162/CrossPropertyTL/elemnet/sample/alignn_matbert-base-cased_chemnlp_prop_mbj_bandgap.config;python dl_regressors_tf2.py --config_file /scratch/yll6162/CrossPropertyTL/elemnet/sample/alignn_matbert-base-cased_robo_prop_mbj_bandgap.config



python dl_regressors_tf2.py --config_file /scratch/yll6162/CrossPropertyTL/elemnet/sample/bert-base-uncased_chemnlp_prop_magmom_outcar.config; python dl_regressors_tf2.py --config_file /scratch/yll6162/CrossPropertyTL/elemnet/sample/bert-base-uncased_robo_prop_magmom_outcar.config; python dl_regressors_tf2.py --config_file /scratch/yll6162/CrossPropertyTL/elemnet/sample/alignn_bert-base-uncased_chemnlp_prop_magmom_outcar.config; python dl_regressors_tf2.py --config_file /scratch/yll6162/CrossPropertyTL/elemnet/sample/alignn_bert-base-uncased_robo_prop_magmom_outcar.config;python dl_regressors_tf2.py --config_file /scratch/yll6162/CrossPropertyTL/elemnet/sample/alignn_matbert-base-cased_chemnlp_prop_magmom_outcar.config; python dl_regressors_tf2.py --config_file /scratch/yll6162/CrossPropertyTL/elemnet/sample/alignn_matbert-base-cased_robo_prop_magmom_outcar.config; 

magmom_outcar


python dl_regressors_tf2.py --config_file /scratch/yll6162/CrossPropertyTL/elemnet/sample/alignn_prop_magmom_outcar.config


{"loss_type": "mae", "log_file": "mof_oxo_form_e_tl.log", 
"config_file": "sample/mof_oxo_form_e_job.config", "use_valid": true, 
"val_data_path": null, 
"test_metric": "mae",
 "test_data_path": null,
  "label": "Oxo Formation Energy", "project": "MOF",
    "train_data_path": "/scratch/yll6162/MOF-oxo/data/query_0_mof_form_e_embed_Oxo Formation Energy.csv", 
    "ext_train_data_path": "/scratch/yll6162/MOF-oxo/data/query_1_mof_form_e_embed_Oxo Formation Energy_new_only.csv", 
    "regressors": null, 
    "input_types": null,
    "paramsGrid": {"optimizer": "Adam", "learning_rate": 0.0001, "patience": 200,
     "dropouts": [0.8, 0.9, 0.7, 0.8], "EVAL_FREQUENCY": 1000}, 
     "architecture": "1024RBx3D-128RBx3D-64Rx2D-32Rx1-1", 
     "save_path": "sample/mof_oxo_form_e", 
      "log_folder": "log", "model_path": null, 
     "last_layer_with_weight": true, "keras_path":"model/mof_oxo_form_e",
"test_size" : 0.2}

python dl_regressors_tf2.py --config_file /scratch/yll6162/CrossPropertyTL/elemnet/sample/mof_h_racs_form_e_tl.config --kfold

python dl_regressors_tf2.py --config_file /scratch/yll6162/CrossPropertyTL/elemnet/sample/mof_oxo_racs_dband_tl.config --kfold

python dl_regressors_tf2.py --config_file /scratch/yll6162/CrossPropertyTL/elemnet/sample/mof_h_racs_dband_tl.config --kfold



python dl_regressors_tf2.py --config_file /scratch/yll6162/CrossPropertyTL/elemnet/sample/matbert-base-cased_chemnlp_prop_spillage.config; python dl_regressors_tf2.py --config_file /scratch/yll6162/CrossPropertyTL/elemnet/sample/matbert-base-cased_chemnlp_prop_mbj_bandgap.config; python dl_regressors_tf2.py --config_file /scratch/yll6162/CrossPropertyTL/elemnet/sample/matbert-base-cased_chemnlp_prop_magmom_outcar.config; python dl_regressors_tf2.py --config_file /scratch/yll6162/CrossPropertyTL/elemnet/sample/matbert-base-cased_chemnlp_prop_slme.config;


python dl_regressors_tf2.py --config_file /scratch/yll6162/CrossPropertyTL/elemnet/sample/matbert-base-cased_chemnlp_prop_slme.config;