import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import pandas as pd
from data_processing import MAXLIFE

from tsfresh import select_features, extract_features
from tsfresh.utilities.dataframe_functions import impute
from itertools import chain

class TsConf(object):
    train_fids = [101]
    test_fids = [201]
    sequence_length = 30
    shift=1
    batch_size=1
    path_checkpoint = './save/save_lstm/lstm_2_layers'
    input_url = "input/dataset.csv"
    plot = False    
    learning_rate = 2*10e-5  # 0.0001
    epochs = 1000
    ann_hidden = 16
    lstm_size = 48  # Number LSTM units
    num_layers = 2  # 2  # Number of layers
    alpha = 0 # regularization coef    
    target = 'RUL'
    groupids = ['FILEID', 'ENGINEID']
    timestamp = 'TIMECYCLE'
    
    def __init__(self, columns):
        """Set values of computed attributes."""
        self.features = columns[~columns.isin(self.groupids+[self.timestamp]+[self.target])]
        self.n_channels = len(self.features)

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

class TsSeries(pd.Series):
    @property
    def _constructor(self):
        return TsSeries
    @property
    def _constructor_expanddim(self):
        return TsDataFrame

class TsDataFrame(pd.DataFrame):
    """
    Class for reshaping dataset to list of timeseries of (n_sequences, sequence_length, num_x_sensors): overlapping/non-overlapping; batch generation: random/sequential/ 
    """
    # normal properties
    _metadata = ['target', 'groupids', 'timestamp']
    
    @property
    def _constructor(self):
        return TsDataFrame

    @property
    def _constructor_sliced(self):
        return TsSeries
    
    #def featGet(self):
        #return self.columns[~self.columns.isin(self.groupids+[self.timestamp]+[self.target])]
    
    #features = property(featGet)
        
    def extract_target(self):
        #extract target
        #print(self)        
        X = self[self.columns[~self.columns.isin([self.target])]]
        y = self[self.target] if self.target else None
        return X, y
    
    def series_to_supervised(self, n_in=1, n_out=1, periods=5, dropnan=True):
        """
        Frame a time series as a supervised learning dataset.
        Arguments:
            data: Sequence of observations as a list or NumPy array.
            n_in: Number of lag observations as input (X).
            n_out: Number of observations as output (y).
            dropnan: Boolean whether or not to drop rows with NaN values.
        Returns:
            Pandas DataFrame of series framed for supervised learning.
        """
        n_vars = 1 if type(self) is list else self.shape[1]
        df = pd.DataFrame(self)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in//periods, 0, -1):
            cols.append(df.shift(i*periods)-df)
            names += [('var%d(t-%d)' % (j+1, i*periods)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out//periods):
            cols.append(df.shift(-i*periods)-df)
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i*periods)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg
    
    def add_lag_roll(self, feat_names, lag=1, step=1, roll=0):
        x = series_to_supervised(self[feat_names], n_in=lag, n_out=1, periods=step)    
        return pd.concat([df[lag:], x], axis=1)
           
    def tsfresh(self):
        return self
    
    def lstm_sampling (self, timesteps=30, lag=1, random = False):        
        # split into samples 
        n = self.shape[0]
        if n ==0:
            return samples, targets
        
        samples, targets = list(), list()        
        X, y = self.extract_target()
        for i in range(0, n-timesteps+1, lag):
            sample = X[i:i+timesteps]
            samples.append(sample)
            if y is not None:
                target = y[i:i+timesteps]
                targets.append(target)
        
        #print("Samples length: ",len(samples),", Targets length: ",len(targets))
        
        # convert list of arrays into 2d array
        if samples:
            data = np.stack(samples)
        else:
            data = None
        if targets:
            y = np.stack(targets)
            
        return data, y

    def batch_generator(self, sequence_length=30, online_shift=1, batch_size=1,  online=True):
        """
        Generator function for creating sequential batches of training-data
        """
        x_train, y_train = self.lstm_sampling(sequence_length, online_shift, batch_size)
        
        num_x_sensors = x_train.shape[2]
        num_train = x_train.shape[0]
        idx = 0

        while idx <= num_train - batch_size:

            # Allocate a new array for the batch of input-signals.
            x_shape = (batch_size, sequence_length, num_x_sensors)
            x_batch = np.zeros(shape=x_shape, dtype=np.float32)
            # Allocate a new array for the batch of output-signals.
            y_shape = (batch_size, sequence_length)
            y_batch = np.zeros(shape=y_shape, dtype=np.float32)
            # Fill the batch with random sequences of data.
            for i in range(batch_size):
                x_batch[i] = x_train[idx+i]
                y_batch[i] = y_train[idx+i]

            if online:
                idx = idx + online_shift  # check if its nee to be idx=idx+1
            #print("num_train %s, idx %s, x_batch.shape %s" % (num_train, idx, x_batch.shape))
            yield (x_batch[:,:,0:3], x_batch[:,:,3:], y_batch)
        
    def endless_batch(self, timesteps=30, lag=1, batch_size=1,  online=True):
        feats_target = self.columns[~self.columns.isin(self.groupids+[self.timestamp])]
        gen = iter(())
        for name, group in self.groupby(self.groupids):
            gen = chain(gen, group.batch_generator(timesteps, lag, batch_size))
        return(gen)            


def dense_layer(x, size,activation_fn, batch_norm = False,phase=False, drop_out=False, keep_prob=None, scope="fc_layer"):
    """
    Helper function to create a fully connected layer with or without batch normalization or dropout regularization

    :param x: previous layer
    :param size: fully connected layer size
    :param activation_fn: activation function
    :param batch_norm: bool to set batch normalization
    :param phase: if batch normalization is set, then phase variable is to mention the 'training' and 'testing' phases
    :param drop_out: bool to set drop-out regularization
    :param keep_prob: if drop-out is set, then to mention the keep probability of dropout
    :param scope: variable scope name
    :return: fully connected layer
    """
    with tf.variable_scope(scope):
        if batch_norm:
            dence_layer = tf.contrib.layers.fully_connected(x, size, activation_fn=None)
            dence_layer_bn = BatchNorm(name="batch_norm_" + scope)(dence_layer, train=phase)
            return_layer = activation_fn(dence_layer_bn)
        else:
            return_layer = tf.layers.dense(x, size,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation=activation_fn)
        if drop_out:
            return_layer = tf.nn.dropout(return_layer, keep_prob)

        return return_layer


def get_RNNCell(cell_types, keep_prob, state_size, build_with_dropout=True):
    """
    Helper function to get a different types of RNN cells with or without dropout wrapper
    :param cell_types: cell_type can be 'GRU' or 'LSTM' or 'LSTM_LN' or 'GLSTMCell' or 'LSTM_BF' or 'None'
    :param keep_prob: dropout keeping probability
    :param state_size: number of cells in a layer
    :param build_with_dropout: to enable the dropout for rnn layers
    :return:
    """
    cells = []
    for cell_type in cell_types:
        if cell_type == 'GRU':
            cell = tf.contrib.rnn.GRUCell(num_units=state_size,
                                          bias_initializer=tf.zeros_initializer())  # Or GRU(num_units)
        elif cell_type == 'LSTM':
            cell = tf.contrib.rnn.LSTMCell(num_units=state_size, use_peepholes=True, state_is_tuple=True,
                                           initializer=tf.contrib.layers.xavier_initializer())
        elif cell_type == 'LSTM_LN':
            cell = tf.contrib.rnn.LayerNormBasicLSTMCell(state_size)
        elif cell_type == 'GLSTMCell':
            cell = tf.contrib.rnn.GLSTMCell(num_units=state_size, initializer=tf.contrib.layers.xavier_initializer())
        elif cell_type == 'LSTM_BF':
            cell = tf.contrib.rnn.LSTMBlockFusedCell(num_units=state_size, use_peephole=True)
        else:
            cell = tf.nn.rnn_cell.BasicRNNCell(state_size)

        if build_with_dropout:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
        cells.append(cell)

    cell = tf.contrib.rnn.MultiRNNCell(cells)

    if build_with_dropout:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

    return cell


class BatchNorm(object):
    """
    usage : dence_layer_bn = BatchNorm(name="batch_norm_" + scope)(previous_layer, train=is_train)
    """
    def __init__(self, epsilon=1e-5, momentum=0.999, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)


def model_summary(learning_rate,batch_size,lstm_layers,lstm_layer_size,fc_layer_size,sequence_length,n_channels,path_checkpoint,spacial_note=''):
    path_checkpoint=path_checkpoint + ".txt"
    if not os.path.exists(os.path.dirname(path_checkpoint)):
        os.makedirs(os.path.dirname(path_checkpoint))

    with open(path_checkpoint, "w") as text_file:
        variables = tf.trainable_variables()

        print('---------', file=text_file)
        print(path_checkpoint, file=text_file)
        print(spacial_note, file=text_file)
        print('---------', '\n', file=text_file)

        print('---------', file=text_file)
        #print('MAXLIFE: ', MAXLIFE,'\n',  file=text_file)
        print('learning_rate: ', learning_rate, file=text_file)
        print('batch_size: ', batch_size, file=text_file)
        print('lstm_layers: ', lstm_layers, file=text_file)
        print('lstm_layer_size: ', lstm_layer_size, file=text_file)
        print('fc_layer_size: ', fc_layer_size, '\n', file=text_file)
        print('sequence_length: ', sequence_length, file=text_file)
        print('n_channels: ', n_channels, file=text_file)
        print('---------', '\n', file=text_file)

        print('---------', file=text_file)
        print('Variables: name (type shape) [size]', file=text_file)
        print('---------', '\n', file=text_file)
        total_size = 0
        total_bytes = 0
        for var in variables:
            # if var.num_elements() is None or [] assume size 0.
            var_size = var.get_shape().num_elements() or 0
            var_bytes = var_size * var.dtype.size
            total_size += var_size
            total_bytes += var_bytes
            print(var.name, slim.model_analyzer.tensor_description(var), '[%d, bytes: %d]' %
                      (var_size, var_bytes), file=text_file)

        print('\nTotal size of variables: %d' % total_size, file=text_file)
        print('Total bytes of variables: %d' % total_bytes, file=text_file)


def scoring_func(error_arr):
    '''

    :param error_arr: a list of errors for each training trajectory
    :return: standered score value for RUL
    '''
    import math
    # print(error_arr)
    pos_error_arr = error_arr[error_arr >= 0]
    neg_error_arr = error_arr[error_arr < 0]

    score = 0
    # print(neg_error_arr)
    for error in neg_error_arr:
        score = math.exp(-(error / 13)) - 1 + score
        # print(math.exp(-(error / 13)),score,error)

    # print(pos_error_arr)
    for error in pos_error_arr:
        score = math.exp(error / 10) - 1 + score
        # print(math.exp(error / 10),score, error)
    return score


def get_state_variables(batch_size, cell):
    # For each layer, get the initial state and make a variable out of it
    # to enable updating its value.
    state_variables = []
    for state_c, state_h in cell.zero_state(batch_size, tf.float32):
        state_variables.append(tf.contrib.rnn.LSTMStateTuple(
            tf.Variable(state_c, trainable=False),
            tf.Variable(state_h, trainable=False)))
    # Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
    return tuple(state_variables)


def get_state_update_op(state_variables, new_states):
    # Add an operation to update the train states with the last state tensors
    update_ops = []
    for state_variable, new_state in zip(state_variables, new_states):
        # Assign the new state to the state variables on this layer
        update_ops.extend([state_variable[0].assign(new_state[0]),
                           state_variable[1].assign(new_state[1])])
    # Return a tuple in order to combine all update_ops into a single operation.
    # The tuple's actual value should not be used.
    return tf.tuple(update_ops)

