# CrossPropertyTL for Deterministic Analysis

This repository contains the code for performing cross-property deep transfer learning to predict materials properties. The DL code has been optimized to provide deterministic result. The input embeddings used for training are extracted using <a href="https://github.com/Jonathanlyj/ALIGNN-BERT-TL-crystal"> Hybrid-LLM-GNN framework </a>. This repo provides the following functions: 

* Train a ElemNet model on a given dataset using Tensorflow or Pytorch framework
* Predict material properties of new compounds with a pre-trained ElemNet model
* Deterministic analysis based on prediction results



## Installation Requirements

The basic requirement for using the files are a Python 3.9 with the packages listed in `requirements.txt`

## Source Files
  
Here is a brief description about the folder content:

* [`elemnet/dl_regressors_tf2_dtm.py`](./elemnet/dl_regressors_tf2_dtm.py): code for training ElemNet model using Tensorflow framework from scratch with deterministic behavior controlled.

* [`elemnet/dl_regressors_torch.py`](./elemnet/dl_regressors_torch.py): code for training ElemNet model using PyTorch framework from scratch with deterministic behavior controlled.

* [`elemnet/dl_regressors_torch_predict.py`](./elemnet/dl_regressors_torch_predict.py): code for predict on unseen dataset using pretrained ElemNet model built with PyTorch framework with deterministic behavior controlled.

* [`CrossPropertyTL_cystal_run.ipynb`](CrossPropertyTL_cystal_run.ipynb): Example notebook demonstrating the full pipeline, including dataset download, environment setup, model training from scratch, and inference with pretrained models.  
  Also available on [Colab](https://colab.research.google.com/drive/1MVldJ7pcExAZbLBF7sTcCIG4raHRRPZY?usp=sharing). The convenient way to do a quick-run on different hardware setups.

* [`data`](./data): 39 different datasets used for training ElemNet model. Transfer learning datasets can be downloaded from <a href="https://figshare.com/articles/dataset/ALIGNN_BERT_TL_train_dataset/27569631">figshare</a> </br> </br> 

* [`determinstics_analysis.ipynb`](./determinstics_analysis.ipynb): Jupyter Notebook to perform analysis based on test set predictions.


### Run ElemNet model

The code to run the ElemNet model is provided in the [`elemnet`](./elemnet) folder. In order to run the model you can pass a sample config file to the dl_regressors_tf2.py from inside of your elemnet directory:

`python dl_regressors_torch.py --config_file elemnet/sample/example_alignn_matbert-base-cased_robo_prop_mbj_bandgap_local.config`

The config file defines all the related hyperparameters associated with the model training and model testing such as loss_type, training_data_path, val_data_path, test_data_path, label, input_type, etc. 

For model inference with pre-trained model, you need to set 'model_path' [e.g. `model/sample_model`]. 
After training, you will get the following files:

* The output log from the training will be saved in the [`log`](./elemnet/log) folder as `log/sample-run_example_tf2.log` file.

* The trained model will be saved in [`model`](./elemnet/model) folder.


Note: Your results should remain deterministic (bit-exact) on a single machine across repeated runs but may differ across machines due to GPU parallelism and hardware-specific variation in numerical execution.

## Developer Team

The code was developed by Youjia Li from the <a href="http://cucis.ece.northwestern.edu/">CUCIS</a> group at the Electrical and Computer Engineering Department at Northwestern University.

## Acknowledgements

The open-source implementation of ElemNet <a href="https://github.com/NU-CUCIS/ElemNet">here</a> provided significant initial inspiration for the structure of this code-base.

## Disclaimer

The research code shared in this repository is shared without any support or guarantee on its quality. However, please do raise an issue if you find anything wrong and I will try my best to address it.

email: youjia@northwestern.edu

Copyright (C) 2025, Northwestern University.

See COPYRIGHT notice in top-level directory.
