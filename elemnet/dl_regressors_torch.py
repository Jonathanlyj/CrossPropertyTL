"""
Train a neural network on the given dataset with given configuration

"""
import argparse
import math
import re
import sys
import traceback
import random
import numpy as np
import time
from data_utils import *
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from train_utils import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
parser = argparse.ArgumentParser(description='run ml regressors on dataset')
parser.add_argument('--train_data_path', help='path to the training dataset',default=None, type=str, required=False)
parser.add_argument('--val_data_path', help='path to the validation dataset',default=None, type=str, required=False)
parser.add_argument('--test_data_path', help='path to the test dataset', default=None, type=str,required=False)
parser.add_argument('--label', help='output variable', default=None, type=str,required=False)
parser.add_argument('--input', help='input attributes set', default=None, type=str, required=False)
parser.add_argument('--config_file', help='configuration file path', default=None, type=str, required=False)
parser.add_argument('--test_metric', help='test_metric to use', default=None, type=str, required=False)

parser.add_argument('--priority', help='priority of this job', default=0, type=int, required=False)
parser.add_argument('--seed', help='random seed', default=0, type=int, required=False)
parser.add_argument('--kfold', help='enable k-fold cross-validation', action='store_true')

args,_ = parser.parse_known_args()

hyper_params = {'batch_size':512, 'num_epochs':2000, 'EVAL_FREQUENCY':1000, 'SAVE_FREQUENCY':10, \
                'learning_rate':1e-7, 'momentum':0.9, 'lr_drop_rate':0.5, 'epoch_step':500, \
                'nesterov':True, 'reg_W':0., 'optimizer':'Adam', 'reg_type':None, \
                    'activation':'relu'}

SEED = 66478

# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

set_seed(SEED)



class ResNet(torch.nn.Module):
    def __init__(self, main_module, side_module=None):
        super().__init__()
        self.main_module = main_module
        self.side_module = side_module

    def forward(self, inputs):
        if not self.side_module:
            return self.main_module(inputs) + inputs
        else:
            return self.main_module(inputs) + self.side_module(inputs)
        
class DropoutBlock(torch.nn.Module):
    def __init__(self, module, dropout_ratio):
        super(DropoutBlock, self).__init__()
        self.module = module
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, inputs):
        x = self.module(inputs)
        x = self.dropout(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = {
            'data': torch.tensor(self.data[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }
        return sample
    

class ModelSlim(torch.nn.Module):
    def __init__(self, architecture, input_size, num_labels=1, activation='relu', dropouts=[]):
        super(ModelSlim, self).__init__()
        archs = architecture.strip().split('-')
        self.arch_layers = []
        prev_num_outputs = input_size
        prev_block_num_outputs = input_size
        for i in range(len(archs)):
            block_layer = []
            arch = archs[i]
            if 'x' in arch:
                arch = arch.split('x')
                num_outputs = int(re.findall(r'\d+',arch[0])[0])
                layers = int(re.findall(r'\d+',arch[1])[0])
                j = 0
                aux_layers = re.findall(r'[A-Z]',arch[0])
                for l in range(layers):
                    crt_layer = []
                    if aux_layers and aux_layers[0] == 'B':
                        if len(aux_layers)>1 and aux_layers[1]=='A':
                            print('adding fully connected layers with %d outputs followed by batch_norm and act' % num_outputs)
                            crt_layer.append(nn.Linear(prev_num_outputs, num_outputs))
                            crt_layer.append(nn.BatchNorm1d(num_outputs, affine=True))
                            crt_layer.append(nn.ReLU())
                        else:
                            print('adding fully connected layers with %d outputs followed by batch_norm' % num_outputs)
                            crt_layer.append(nn.Linear(prev_num_outputs, num_outputs))
                            crt_layer.append(nn.ReLU())
                            crt_layer.append(nn.BatchNorm1d(num_outputs, affine=True))
                    else:
                        print('adding fully connected layers with %d outputs' % num_outputs)
                        crt_layer.append(nn.Linear(prev_num_outputs, num_outputs))
                        crt_layer.append(nn.ReLU())
                    if 'R' in aux_layers:
                        if prev_num_outputs and prev_num_outputs==num_outputs:
                            print('adding residual, both sizes are same')
                            crt_layer = [ResNet(nn.Sequential(*crt_layer))]
                        else:
                            crt_layer = [ResNet(nn.Sequential(*crt_layer), nn.Linear(prev_num_outputs, num_outputs))]
                            print('adding residual with fc as the size are different')
                    prev_num_outputs = num_outputs
                    block_layer = block_layer + crt_layer
                aux_layers_sub = re.findall(r'[A-Z]', arch[1])
                if 'R' in aux_layers_sub:
                    if prev_block_num_outputs and prev_block_num_outputs == num_outputs:
                        print('adding residual to stub, both sizes are same')
                        block_layer = [ResNet(nn.Sequential(*block_layer))]
                    else:
                        print('adding residual to stub with fc as the size are different')
                        block_layer = [ResNet(nn.Sequential(*block_layer), nn.Linear(prev_block_num_outputs, num_outputs))]
                if 'D' in aux_layers_sub and num_labels == 1 and len(dropouts) > i:
                    #skip dropout for now
                    pass
                    # print('adding dropout', dropouts[i])
                    # block_layer = [DropoutBlock(nn.Sequential(*block_layer), dropout_ratio=dropouts[i])]
                prev_block_num_outputs = num_outputs
            else:
                # final layer
                print('adding final layer with ' + str(num_labels) + ' output')
                block_layer = [nn.Linear(prev_block_num_outputs, num_labels)]
                if 'R' in arch:
                    print('using ReLU at last layer')
                    block_layer.append(nn.ReLU())
            self.arch_layers += block_layer

        self.model = nn.Sequential(*self.arch_layers)
    
    def forward(self, x):
        return self.model(x)

def numpy_to_tensor(lst):
    """
    Convert numpy array to torch tensor
    """
    if type(lst) == np.ndarray:
        return torch.tensor(lst).to(dtype=torch.float32)

    else:
        out = [torch.tensor(item).to(dtype=torch.float32) for item in lst]
        return tuple(out)



# def error_rate(predictions, labels, step=0, dataset_partition=''):
#     return np.mean(np.absolute(predictions - labels))
# def error_rate_classification(predictions, labels, step=0, dataset_partition=''):
#     return 100.0 - (100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0])

def print_model(model):
    """ 
    A simple functon that prints out a PyTorch model's structural details
    """
    # Print the number of parameters in the model
    parameter_count =  sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("In total, this network has ", parameter_count, " parameters")

def eval_in_batches(model, dataloader, device, criterion=nn.L1Loss()):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            data, target = batch['data'], batch['label']
            if device.type == 'cuda':
                data, target = data.cuda(), target.cuda()
            output = model(data)
            predictions.append(output.cpu().numpy())
    return np.concatenate(predictions, axis=0)



def save_predictions(test_dataloader, model, device, ids_test, save_path):
    model.eval()
    raw_predictions = []
    actual_labels = []
    # Get test predictions
    with torch.no_grad():
        for batch in test_dataloader:
            data, target = batch['data'], batch['label']
            if device.type == 'cuda':
                data = data.cuda()
            output = model(data)
            raw_predictions.append(output.cpu().numpy())  # Move to CPU, then convert to numpy array
            actual_labels.append(target.cpu().numpy())    # Move to CPU, then convert to numpy array

    raw_predictions = np.concatenate(raw_predictions).flatten()
    actual_labels = np.concatenate(actual_labels).flatten()
    output_dir = "../prediction_torch"
    os.makedirs(output_dir, exist_ok=True)
    pred_path = os.path.join(output_dir, f"{save_path.split('/')[-1]}_pred_otf.csv")
    predictions_df = pd.DataFrame({
        'ids_test': ids_test, 
        'labels': actual_labels,
        'predictions': raw_predictions
    })

    # Save the predictions DataFrame as a CSV file
    predictions_df.to_csv(pred_path, index=False)

    print(f"Predictions saved at {pred_path}")
def run_regressors(train_X, train_y, valid_X, valid_y, test_X, test_y, save_pred, ids_test=None, logger=None, config=None):
    assert config is not None
    hyper_params.update(config['paramsGrid'])
    assert  logger is not None
    rr = logger

    device = torch.device('cuda') if not hasattr(args, 'use_cpu') and torch.cuda.is_available() else torch.device('cpu')
    # tf.compat.v1.reset_default_graph()
    train_X = train_X.reshape(train_X.shape[0], -1).astype("float32")
    valid_X = valid_X.reshape(valid_X.shape[0], -1).astype("float32")
    test_X = test_X.reshape(test_X.shape[0], -1).astype("float32")

    num_input = train_X.shape[1]
    batch_size = hyper_params['batch_size']
    learning_rate = hyper_params['learning_rate']
    optimizer = hyper_params['optimizer']
    architecture = config['architecture']
    num_epochs = hyper_params['num_epochs']
    model_path = config['model_path']
    patience = hyper_params['patience']
    save_path = config['save_path']
    loss_type = config['loss_type']
    keras_path = config['keras_path']
    if 'dropouts' in hyper_params:
        dropouts = hyper_params['dropouts']
    else:
        dropouts = []
    test_metric = mean_squared_error
    if config['test_metric']=='mae':
        test_metric = mean_absolute_error
    use_valid = config['use_valid']
    # EVAL_FREQUENCY = hyper_params['EVAL_FREQUENCY']
    SAVE_FREQUENCY = hyper_params['SAVE_FREQUENCY']

    train_y = train_y.reshape(train_y.shape[0]).astype("float32")
    valid_y = valid_y.reshape(valid_y.shape[0]).astype("float32")
    test_y = test_y.reshape(test_y.shape[0]).astype("float32")

    train_data = train_X
    train_labels = train_y
    test_data = test_X
    test_labels = test_y
    validation_data = valid_X
    validation_labels = valid_y

    train_set = CustomDataset(train_data, train_labels)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    valid_set = CustomDataset(validation_data, validation_labels)
    valid_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    test_set = CustomDataset(test_data, test_labels)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # architecture = "1024x4D-512x3D-256x3D-128x3D-64x2-32x1-1"
    rr.fprint("train matrix shape of train_X: ",train_X.shape, ' train_y: ', train_y.shape)
    rr.fprint("valid matrix shape of train_X: ",valid_X.shape, ' valid_y: ', valid_y.shape)
    rr.fprint("test matrix shape of valid_X:  ",test_X.shape, ' test_y: ', test_y.shape)
    rr.fprint('architecture is: ',architecture)
    rr.fprint('learning rate is ',learning_rate)

    # train_data_node = tf.placeholder(tf.float32, shape=(batch_size, num_input))
    ##train_data_node = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, num_input))
    # eval_data = tf.placeholder(tf.float32, shape=(batch_size, num_input))

    # logits,_ = model_slim(train_data_node, architecture, dropouts=dropouts)
    input_size = train_X.shape[1]
    model = ModelSlim(architecture, input_size, dropouts=dropouts).to(device)
    print(model)
    assert  loss_type == 'mae'
    if loss_type == 'mae':
        loss_function = nn.L1Loss()

    # batch = tf.Variable(0)

    assert optimizer=='Adam'
    if optimizer=='Adam':
        # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=batch)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # eval_prediction,_ = model_slim(eval_data, architecture,train=False, dropouts=dropouts)

    start_time = time.time()
    print('num_epochs is ', num_epochs)
    # sess = tf.Session()
    # sess.run(tf.initialize_all_variables())
    rr.fprint('Initialized')
    # train_writer = tf.summary.FileWriter('summary', graph_def=sess.graph_def)
    train_size = train_X.shape[0]
    best_val_error = 1e10
    best_epoch = 0
    start_epoch = 0
    if model_path and os.path.exists(model_path):
        model_load_path = os.path.join(model_path, f"model_{architecture}.pt")
        rr.fprint('Restoring model from %s' % model_load_path)
        checkpoint = torch.load(model_load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    rr.fprint('start training')

  

    num_batches = len(train_dataloader)
    rr.fprint('num_batches:', num_batches)
    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        for batch_idx, batch in enumerate(train_dataloader):
            model.train()
            data, target = batch['data'], batch['label']
            if device.type == 'cuda':
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            target = target.unsqueeze(1)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            running_loss += train_loss
            # if (step + 1) % EVAL_FREQUENCY == 0:
            if batch_idx == num_batches - 1:
                elapsed_time = time.time() - start_time
                if use_valid:
                    val_predictions = eval_in_batches(model, valid_dataloader, device)
                    val_error = test_metric(val_predictions, validation_labels)

                if not use_valid:
                    test_predictions = eval_in_batches(model, test_dataloader, device)
                    test_error = test_metric(test_predictions, test_labels)
                    val_error = test_error
                rr.fprint(
                    'Epoch %.2d, train loss: %.12f, validation error: %.12f, best validation error: %.12f' % (
                    epoch, train_loss, val_error, best_val_error))
                if best_val_error > val_error:
                    best_val_error = val_error
                    best_epoch = epoch

                if best_epoch + patience <= epoch:
                    rr.fprint('No improvement observed in last %d epochs, best error in validation set is %f'%(patience, best_val_error))
                    test_predictions = eval_in_batches(model, test_dataloader, device)
                    test_error = test_metric(test_predictions, test_labels)
                    rr.fprint('Test error is %.12f' % test_error)
                    if save_pred:
                        save_predictions(test_dataloader, model, device, ids_test, save_path)
                    return best_val_error

            sys.stdout.flush()
            start_time = time.time()

        if (epoch + 1) % SAVE_FREQUENCY == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            os.makedirs(keras_path, exist_ok=True)
            model_save_path = os.path.join(keras_path, f"model_{architecture}.pt")
            torch.save(checkpoint, model_save_path)
            rr.fprint(f'Checkpoint saved at {model_save_path} at epoch {epoch+1}')
    test_predictions = eval_in_batches(model, test_dataloader, device)
    test_error = test_metric(test_predictions, test_labels)
    rr.fprint('Test error is %f' % test_error)
    if save_pred:
        save_predictions(test_dataloader, model, device, ids_test, save_path)


    return best_val_error


if __name__=='__main__':
    args = parser.parse_args()
    config = {}
    config['train_data_path'] = args.train_data_path
    config['val_data_path'] = args.val_data_path
    config['test_data_path'] = args.test_data_path
    config['label'] = args.label
    config['input_type'] = args.input
    config['log_folder'] = 'logs_dl'
    config['log_file'] = 'dl_log_' + get_date_str() + '.log'
    config['test_metric'] = args.test_metric
    config['architecture'] = 'infile'
    config['model_seed'] = args.seed
    # config['ext_train_data_path'] = args.ext_train_data_path
    if args.config_file:
        config.update(load_config(args.config_file))
    if not os.path.exists(config['log_folder']):
        createDir(config['log_folder'])
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_ts = f"{os.path.splitext(config['log_file'])[0]}_{timestamp}{os.path.splitext(config['log_file'])[1]}"
    logger = Record_Results(os.path.join(config['log_folder'], log_file_ts))
    logger.fprint('job config: ' + str(config))
    small_constant = 1e-6
    if args.kfold:
        kf = KFold(n_splits = round(1/config['test_size']), shuffle=False)
        # kf = KFold(n_splits = round(1/config['test_size']), shuffle=True, random_state=seed + 10)
        datasets = []
        ids, X, y = load_csv(train_data_path=config['train_data_path'],
                                                                    #   val_data_path=config['val_data_path'],
                                                                    #   test_data_path=config['test_data_path'],
                                                                    test_size = config['test_size'],
                                                                    #   input_types = config['input_types'],
                                                                    label=config['label'], logger=logger,
                                                                    full = True)
        if config['ext_train_data_path'] is not None:
            ext_ids, ext_X, ext_y = load_csv(train_data_path=config['ext_train_data_path'],
                                                                        #   val_data_path=config['val_data_path'],
                                                                        #   test_data_path=config['test_data_path'],
                                                                        test_size = config['test_size'],
                                                                        #   input_types = config['input_types'],
                                                                        label=config['label'], logger=logger,
                                                                        full = True)

        
        mads = []
        maes = []
        for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
            train_X, valid_X = X[train_index], X[test_index]
            train_y, valid_y = y[train_index], y[test_index]
            _, test_ids = ids[train_index], ids[test_index]
            if config['ext_train_data_path'] is not None:
                train_X = np.concatenate((train_X, ext_X), axis=0)
                train_y = np.concatenate((train_y, ext_y), axis=0)
            test_X = valid_X
            test_y = valid_y
            # train_X = np.nan_to_num(train_X, nan=small_constant)
            # valid_X = np.nan_to_num(valid_X, nan=small_constant)
            # test_X = np.nan_to_num(test_X, nan=small_constant)
 
            assert not np.any(np.isnan(train_X))
            # train_X = np.where(train_X == 0, small_constant, train_X)
            # valid_X = np.where(valid_X == 0, small_constant, valid_X)
            # test_X = np.where(test_X == 0, small_constant, test_X)

            test_mae = run_regressors(train_X, train_y, valid_X, valid_y, valid_X, valid_y, logger=logger, config=config, save_pred=True, ids = test_ids)
            mad = mean_absolute_error(len(test_y) * [np.mean(train_y)], test_y)
            mads.append(mad)
            maes.append(test_mae)
            logger.fprint(f'MAD is {mad}')
        logger.fprint(f'Mean MAD is {np.mean(mads)}')
        logger.fprint(f'Mean MAE is {np.mean(maes)}')
        logger.fprint('done')
    else:
        train_ids, train_X, train_y,  valid_ids, valid_X, valid_y, test_ids, test_X, test_y = load_csv(train_data_path=config['train_data_path'],
                                                                                                    val_data_path=config['val_data_path'],
                                                                                                    test_data_path=config['test_data_path'],
                                                                                                    test_size = config['test_size'],
                                                                                                    val_size = config['val_size'],
                                                                                                    #   input_types = config['input_types'],
                                                                                                    label=config['label'], logger=logger,
                                                                                                    full = False,
                                                                                                    save_data = False,
                                                                                                    shuffle=True)
                                    

        # train_X = np.nan_to_num(train_X, nan=small_constant)
        # valid_X = np.nan_to_num(valid_X, nan=small_constant)
        # test_X = np.nan_to_num(test_X, nan=small_constant)

        assert not np.any(np.isnan(train_X))
        # train_X = np.where(train_X == 0, small_constant, train_X)
        # valid_X = np.where(valid_X == 0, small_constant, valid_X)
        # test_X = np.where(test_X == 0, small_constant, test_X)

        run_regressors(train_X, train_y, valid_X, valid_y, test_X, test_y, save_pred=True, ids_test = test_ids, logger=logger, config=config)
        logger.fprint('done')
