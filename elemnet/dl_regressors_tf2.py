"""
Train a neural network on the given dataset with given configuration

"""
# SEED = 66478
import numpy as np
# np.random.seed(args.SEED)
import tensorflow as tf
# tf.random.set_seed(SEED)
import random
# random.seed(SEED)

import argparse
import math
import re
import sys
import traceback

from tensorflow.keras import Input, Model, regularizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
import time
from data_utils import *
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from tensorflow.python import debug as tf_debug
from train_utils import *
import datetime
# import autosklearn.classification

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

hyper_params = {'batch_size':256, 'num_epochs':2000, 'EVAL_FREQUENCY':1000, \
                'learning_rate':1e-7, 'momentum':0.9, 'lr_drop_rate':0.5, 'epoch_step':500, \
                'nesterov':True, 'reg_W':0., 'optimizer':'Adam', 'reg_type':None, \
                    'activation':'relu', 'patience':100}

# NN architecture
SEED=1234567

np.random.seed(args.seed)
tf.random.set_seed(args.seed)
random.seed(args.seed)
tf.keras.utils.set_random_seed(args.seed)

SEED = args.seed
print(f"SEED set to {args.seed}")
#1024Rx4D-512Rx3D-256Rx3D-128Rx3D-64Rx2-32Rx1-1

def run_regressors(train_X, train_y, valid_X, valid_y, test_X, test_y, ids = None, ids_test = None, logger=None, config=None, save_pred=False):
    assert config is not None
    hyper_params.update(config['paramsGrid'])
    assert  logger is not None
    rr = logger

    def define_model(data, architecture, num_labels=1, activation='relu', dropouts=[]):
        assert '-' in architecture
        archs = architecture.strip().split('-')
        net = data
        pen_layer = net
        prev_layer = net
        prev_num_outputs = None
        prev_block_num_outputs = None
        prev_stub_output = net
        for i in range(len(archs)):
            arch = archs[i] # 1024Rx4D
            if 'x' in arch:
                arch = arch.split('x')
                num_outputs = int(re.findall(r'\d+',arch[0])[0]) # 1024
                layers = int(re.findall(r'\d+',arch[1])[0]) # 4
                j = 0
                aux_layers = re.findall(r'[A-Z]',arch[0]) # ['R']
                for l in range(layers):
                    if aux_layers and aux_layers[0] == 'B':
                        if len(aux_layers)>1 and aux_layers[1]=='A':
                            rr.fprint('adding fully connected layers with %d outputs followed by batch_norm and act' % num_outputs)

                            net = Dense(num_outputs, 
                                        name='fc' + str(i) + '_' + str(j),
                                        kernel_regularizer=regularizers.l2(0.01) if hyper_params['reg_type'] == 'L2' else None,
                                        activation=None)(net)
                            net = BatchNormalization(center=True, scale=True, name='fc_bn'+str(i)+'_'+str(j))(net)
                            if activation =='relu': net = Activation('relu')(net)
                        else:
                            rr.fprint('adding fully connected layers with %d outputs followed by batch_norm' % num_outputs)
                            net = Dense(num_outputs,
                                        name='fc' + str(i) + '_' + str(j),
                                        kernel_regularizer=regularizers.l2(0.01) if hyper_params['reg_type'] == 'L2' else None,
                                        activation=activation)(net)
                            net = BatchNormalization(center=True, scale=True,
                                             name='fc_bn' + str(i) + '_' + str(j))(net)

                    else:
                        rr.fprint('adding fully connected layers with %d outputs' % num_outputs)

                        net = Dense(num_outputs,
                                    name='fc' + str(i) + '_' + str(j), 
                                    kernel_regularizer=regularizers.l2(0.01) if hyper_params['reg_type'] == 'L2' else None,
                                    activation=activation)(net)

                    if 'R' in aux_layers:
                        if prev_num_outputs and prev_num_outputs==num_outputs:
                            rr.fprint('adding residual, both sizes are same')

                            net = net+prev_layer
                        else:
                            rr.fprint('adding residual with fc as the size are different')
                            net = net + Dense(num_outputs,
                                                name='fc' + str(i) + '_' +'dim_'+ str(j),
                                                activation=None)(prev_layer)
                    prev_num_outputs = num_outputs
                    j += 1
                    prev_layer = net
                aux_layers_sub = re.findall(r'[A-Z]', arch[1])
                if 'R' in aux_layers_sub:
                    if prev_block_num_outputs and prev_block_num_outputs == num_outputs:
                        rr.fprint('adding residual to stub, both sizes are same')
                        net = net + prev_stub_output
                    else:
                        rr.fprint('adding residual to stub with fc as the size are different')
                        net = net + Dense(num_outputs,
                                         name='fc' + str(i) + '_' + 'stub_dim_' + str(j),
                                         activation=None)(prev_stub_output)

                if 'D' in aux_layers_sub and (num_labels == 1) and len(dropouts) > i:
                    rr.fprint('adding dropout', dropouts[i])
                    #turn on training=True for psed
                    # net = Dropout(1.-dropouts[i], seed=SEED)(net, training=True)
                    net = Dropout(1.-dropouts[i], seed=SEED)(net, training=False)
                prev_stub_output = net
                prev_block_num_outputs = num_outputs
                prev_layer = net

            else:
                if 'R' in arch:
                    act_fun = 'relu'
                    rr.fprint('using ReLU at last layer') 
                elif 'T' in arch:
                    act_fun = 'tanh'
                    rr.fprint('using TanH at last layer')    
                else:
                    act_fun = None
                pen_layer = net
                rr.fprint('adding final layer with ' + str(num_labels) + ' output')
                net = Dense(num_labels, name='fc' + str(i),
                            activation=act_fun)(net)

        return net

    def error_rate(predictions, labels, step=0, dataset_partition=''):

        return np.mean(np.absolute(predictions - labels))

    def error_rate_classification(predictions, labels, step=0, dataset_partition=''):
        return 100.0 - (100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0])

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
    last_layer_with_weight = config['last_layer_with_weight']
    if 'dropouts' in hyper_params:
        dropouts = hyper_params['dropouts']
    else:
        dropouts = []
    test_metric = mean_squared_error
    if config['test_metric']=='mae':
        test_metric = mean_absolute_error
    if config['test_metric']=='accuracy':
        test_metric = accuracy_score    
    use_valid = config['use_valid']
    EVAL_FREQUENCY = hyper_params['EVAL_FREQUENCY']


    train_y = train_y.reshape(train_y.shape[0]).astype("float32")
    valid_y = valid_y.reshape(valid_y.shape[0]).astype("float32")
    test_y = test_y.reshape(test_y.shape[0]).astype("float32")

    train_data = train_X
    train_labels = train_y
    test_data = test_X
    test_labels = test_y
    validation_data = valid_X
    validation_labels = valid_y


    rr.fprint("train matrix shape of train_X: ",train_X.shape, ' train_y: ', train_y.shape)
    rr.fprint("valid matrix shape of train_X: ",valid_X.shape, ' valid_y: ', valid_y.shape)
    rr.fprint("test matrix shape of valid_X:  ",test_X.shape, ' test_y: ', test_y.shape)
    rr.fprint('architecture is: ',architecture)
    rr.fprint('learning rate is ',learning_rate)



    rr.fprint('model path is ', model_path)
    model = None

    inputs = Input(shape=(num_input,), name='elemental_fractions')
    outputs = define_model(inputs, architecture, dropouts=dropouts)
    model = Model(inputs=inputs, outputs=outputs, name= 'ElemNet')
    model.summary(print_fn=lambda x: rr.fprint(x))

    if model_path:
        rr.fprint('Restoring model from %s' % model_path)
        model_h5 = "%s.h5" % model_path
        model.load_weights(model_h5)
        if not last_layer_with_weight:
            rr.fprint('removing last layer to add model and adding dense layer without weight')
            newl16 = Dense(1, activation=None)(model.layers[-2].output)  
            model = Model(inputs=model.input, outputs=[newl16])

    assert optimizer == 'Adam' 

    if loss_type=='mae':
        model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=optimizers.Adam(learning_rate=learning_rate), metrics=['mean_absolute_error'])
    elif loss_type=='binary':
        model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=optimizers.Adam(learning_rate=learning_rate), metrics=[tf.keras.metrics.BinaryAccuracy()])

    class LossHistory(Callback):
        def on_epoch_end(self, epoch, logs={}):
            #rr.fprint(
            #    'Step %d (epoch %.2d), %.1f s minibatch loss: %.5f, validation error: %.5f, test error: %.5f, best validation error: %.5f' % (
            #        step, int(step * batch_size) / train_size,
            #        elapsed_time, l_, val_error, test_error, best_val_error))

            rr.fprint('{}: Current epoch: {}, loss: {}, validation loss: {}'.format(datetime.datetime.now(), epoch, logs['loss'], logs['val_loss']))

    rr.fprint('start training')

    early_stopping = EarlyStopping(patience=patience, restore_best_weights=True, monitor='val_loss')
    checkpointer = ModelCheckpoint(filepath=save_path, verbose=0, save_best_only=True, save_freq='epoch', save_format='tf', period=10)
    history = model.fit(train_X, train_y, verbose=2, batch_size=batch_size, epochs=num_epochs, validation_data=(valid_X, valid_y), callbacks=[early_stopping, LossHistory(), checkpointer])
    # history = model.fit(train_X, train_y, verbose=2, batch_size=batch_size, epochs=num_epochs, validation_data=(valid_X, valid_y), callbacks=[LossHistory(), checkpointer])
    if use_valid:
        test_result = model.evaluate(test_X, test_y, batch_size=32)
        rr.fprint('the test error is ',test_result)

    # rr.fprint(history.history)
    model.save(save_path, save_format='tf')

    filename_json = "%s.json" % keras_path
    filename_h5 = "%s.h5" % keras_path

    model_json = model.to_json()
    with open(filename_json, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(filename_h5)

    rr.fprint('saved model to '+save_path)
    if save_pred:

        raw_predictions = model.predict(test_X)
        actual_labels = test_y.flatten()


        # Create a new DataFrame with predictions and actual labels
        # output_dir = "../pred/mof/query_3"
        output_dir = "../prediction"
        pred_path = os.path.join(output_dir, f"{save_path.split('/')[-1]}_pred_otf.csv")
        predictions_df = pd.DataFrame({'ids_test': ids_test,'labels': actual_labels, 'predictions': raw_predictions.flatten()})
        # if not os.path.exists(pred_path):
        #     predictions_df = pd.DataFrame({'ids_test': ids,'labels': actual_labels, 'predictions': raw_predictions.flatten()})
        # else:
        #     predictions_df = pd.read_csv(pred_path, index_col = 0)
        #     new_df = pd.DataFrame({'ids_test': ids,'labels': actual_labels, 'predictions': raw_predictions.flatten()})
        #     predictions_df = pd.concat([predictions_df, new_df])
        predictions_df.to_csv(pred_path)


    return test_result[1]


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
        train_ids, train_X, train_y, valid_ids, valid_X, valid_y, test_ids, test_X, test_y = load_csv(train_data_path=config['train_data_path'],
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

        run_regressors(train_X, train_y, valid_X, valid_y, test_X, test_y, save_pred=True, ids_test= test_ids, logger=logger, config=config)
        logger.fprint('done')
