import argparse
import torch
import os
import numpy as np
from data_utils import *
from train_utils import *
from dl_regressors_torch import CustomDataset, ModelSlim, save_predictions, log_machine_info
import pandas as pd
pd.set_option("display.precision", 10)  # or more, e.g. 15

parser = argparse.ArgumentParser(description='Run inference on test set using saved PyTorch model')
parser.add_argument('--config_file', required=True, type=str, help='YAML config file used in training')
parser.add_argument('--model_path', required=True, type=str, help='Directory containing model_*.pt checkpoint')


args = parser.parse_args()
example_sample_id = 'JVASP-1151.vasp'  # Example sample ID for testing
# example_sample_id = 'JVASP-28397.vasp'  # Example sample ID for testing
# Load config
config = load_config(args.config_file)
architecture = config.get('architecture', 'infile')
dropouts = config['paramsGrid'].get('dropouts', [])
label_name = config['label']
save_path = config['save_path']

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# logger = Record_Results(os.path.join(config['log_folder'], log_file_ts))
logger = Record_Results('/dev/null')

# Load test data
ids, X, y = load_csv(
                     train_data_path=config['test_data_path'], # for comparing raw prediction for example sample 
                    # train_data_path=config['train_data_path'], # for comparing testset MAE value
                     val_data_path=None,
                     test_data_path=None,
                     label=label_name,
                     logger=logger,
                     full=True)

X = X.astype(np.float32).reshape(X.shape[0], -1)
y = y.astype(np.float32)

test_dataset = CustomDataset(X, y)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)

# Load model
input_size = X.shape[1]
model = ModelSlim(architecture, input_size, dropouts=dropouts).to(device)

log_machine_info(logger)
print(f"Loading model from {args.model_path}")
checkpoint = torch.load(args.model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()



def extract_timestamp(file_path: str) -> str:
    filename = os.path.basename(file_path)
    match = re.search(r'\d{8}_\d{6}', filename)
    return match.group(0) if match else ""

timestamp = extract_timestamp(args.model_path)
model_str = 'cpu_leia' if device.type == 'cpu' else 'gpu_leia'
run_id = "_".join([timestamp,model_str])
# Inference and save
df = save_predictions(test_loader, model, device, ids, save_path, timestamp=run_id, skip_save=False)
df = df.iloc[:] #
mae = np.mean(np.abs(df['predictions'] - df['labels']))
num_samples = len(df)

if example_sample_id in df['ids_test'].values:
    logger.fprint(f'Found {example_sample_id} in predictions, prediction: {df[df["ids_test"] == example_sample_id]["predictions"].values[0]}')
logger.fprint(f'Inference set size: {num_samples}')
logger.fprint(f'MAE on inference set: {mae}')