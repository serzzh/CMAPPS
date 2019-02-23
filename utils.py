import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import pandas as pd
from data_processing import MAXLIFE
import time
import datetime
from matplotlib import pyplot as plt


from tsfresh import select_features, extract_features
from tsfresh.utilities.dataframe_functions import impute
from itertools import chain

class TsConf(object):
    train_fids = [102]
    test_fids = [202]
    cmapss_url = "input/dataset.csv"
    cmapss_n_clusters=6
    ewm = 20
    target = 'RUL'
    timestamp = 'TIMECYCLE'

    groupids = ['FILEID', 'ENGINEID']
    norm_groupids = ['FILEID','CLUSTER']
    opset = ["Alt", "Mach", "TRA"]
    sequence_length = 30
    shift=30
    batch_size=30
    trim_left = True # trim left rows for each group to end exactly at the final of the group?
    random = False # using random shuffle in batch generator?

    path_checkpoint = './save/save_lstm/lstm_2_layers'
    stateful = False # using stateful lstm?
    learning_rate = 2*10e-5  # 0.0001
    epochs = 1000
    ann_hidden = 16
    lstm_size = 48  # Number LSTM units
    num_layers = 2  # 2  # Number of layers
    alpha = 0 # regularization coef    
    restore_model = True #restore model when training
    
    plot = False    
    
    columns_old = ["ENGINEID", "TIMECYCLE", "Alt", "Mach", "TRA", "Total temp at fan in (T2)", "Total temp at LPC out (T24)", "Total temp at HPC out (T30)", "Total temp at LPT out (T50)", 
    "Pres at fan in (P2)", "Total pres in bypass-duct (P15)", "Total pres at HPC out (P30)", "Physical fan speed (Nf)", 
    "Physical core speed (Nc)", "Engine pres ratio (epr=P50/P2)", "Static pres at HPC out (Ps30)", "Ratio of fuel flow to Ps30 (phi)",
    "Corrected fan speed (NRf)", "Corrected core speed (NRc)", "Bypass Ratio (BPR)", "Burner fuel-air ratio (farB)", 
    "Bleed Enthalpy (htBleed)", "Demanded fan speed (Nf_dmd)", "Demanded corrected fan speed (PCNfR_dmd)", "HPT coolant bleed (W31)",
    "LPT coolant bleed (W32)", "FILEID","RUL"]
    columns_new = ["FILEID","ENGINEID", "TIMECYCLE", "Alt", "Mach", "TRA", "Total temp at fan in (T2)", "Total temp at LPC out (T24)", "Total temp at HPC out (T30)", "Total temp at LPT out (T50)", 
    "Pres at fan in (P2)", "Total pres in bypass-duct (P15)", "Total pres at HPC out (P30)", "Physical fan speed (Nf)", 
    "Physical core speed (Nc)", "Engine pres ratio (epr=P50/P2)", "Static pres at HPC out (Ps30)", "Ratio of fuel flow to Ps30 (phi)",
    "Corrected fan speed (NRf)", "Corrected core speed (NRc)", "Bypass Ratio (BPR)", "Burner fuel-air ratio (farB)", 
    "Bleed Enthalpy (htBleed)", "Demanded fan speed (Nf_dmd)", "Demanded corrected fan speed (PCNfR_dmd)", "HPT coolant bleed (W31)",
    "LPT coolant bleed (W32)", "RUL"]
    
    select_feat = ["Total temp at LPC out (T24)", 
               "Total temp at HPC out (T30)", 
               "Total temp at LPT out (T50)", 
               "Physical core speed (Nc)", 
               "Static pres at HPC out (Ps30)", 
               "Corrected core speed (NRc)", 
               "Bypass Ratio (BPR)", 
               "Bleed Enthalpy (htBleed)"]


    
    def __init__(self):
        """Set values of computed attributes."""
        self.features = [x for x in self.columns_new if x not in (self.groupids+[self.timestamp]+[self.target]+self.opset)]        
        self.n_channels = len(self.select_feat)

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")



class TsLSTM():
    """Class for training LSTM network.
    """

    def __init__(self, config):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        self.config = config
        self.model_dir = config.path_checkpoint
        #self.set_log_dir()
        self.model = self._build_lstm(config=config)

    def _build_lstm(self, config):
        print ("Building LSTM...")
        
        self.X = tf.placeholder(tf.float32, [None, config.sequence_length, config.n_channels], name='inputs')
        self.Y = tf.placeholder(tf.float32, [None, config.sequence_length], name='labels')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')
        
        is_train = tf.placeholder(dtype=tf.bool, shape=None, name="is_train")

        input_layer = self.X

        shape = input_layer.get_shape().as_list()
        print('My Conv Shape:',shape)
        input_flat = tf.reshape(input_layer, [-1, shape[1] * shape[2]])

        dence_layer_1 = dense_layer(input_flat, size=config.sequence_length * config.n_channels, 
                                    activation_fn=tf.nn.relu, batch_norm=False,
                                    phase=is_train, drop_out=True, keep_prob=keep_prob,
                                    scope="fc_1")
        lstm_input = tf.reshape(dence_layer_1, [-1, config.sequence_length, config.n_channels])

        cell = get_RNNCell(['LSTM'] * config.num_layers, keep_prob=keep_prob, state_size=config.lstm_size)
        init_states = cell.zero_state(config.batch_size, tf.float32)

        # For each layer, get the initial state. states will be a tuple of LSTMStateTuples.
        states = get_state_variables(config.batch_size, cell)

        # Unroll the LSTM
        rnn_output, new_states = tf.nn.dynamic_rnn(cell, lstm_input, dtype=tf.float32, initial_state=states)

        # Add an operation to update the train states with the last state tensors.
        self.update_op = get_state_update_op(states, new_states) if config.stateful else get_state_update_op(states, init_states)
        self.reset_op = get_state_update_op(states, init_states)

        stacked_rnn_output = tf.reshape(rnn_output, [-1, config.lstm_size])  # change the form into a tensor

        dence_layer_2 = dense_layer(stacked_rnn_output, size=config.ann_hidden, activation_fn=tf.nn.relu, batch_norm=False,
                                    phase=is_train, drop_out=True, keep_prob=keep_prob,
                                    scope="fc_2")

        dence_layer_3 = dense_layer(dence_layer_2, size=config.ann_hidden, activation_fn=tf.nn.relu, batch_norm=False,
                                    phase=is_train, drop_out=True, keep_prob=keep_prob,
                                    scope="fc_2_2")

        self.output = dense_layer(dence_layer_3, size=1, activation_fn=None, batch_norm=False, phase=is_train, drop_out=False,
                             keep_prob=keep_prob,
                             scope="fc_3_output")

        prediction = tf.reshape(self.output, [-1])
        y_flat = tf.reshape(self.Y, [-1])

        self.h = prediction - y_flat

        tv = tf.trainable_variables()
        regularization_cost = tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])

        cost_function = tf.reduce_sum(tf.square(self.h)) + config.alpha*regularization_cost
        RMSE = tf.sqrt(tf.reduce_mean(tf.square(self.h)))
        self.optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost_function)
        self.keep_prob = keep_prob
        self.learning_rate_ = learning_rate_ 
        
        
    def train(self, train, test, config):
        
        with tf.Session() as session:
            
            saver = tf.train.Saver()
            
            keep_prob = self.keep_prob
            learning_rate_ = self.learning_rate_ 

            model_summary(learning_rate=config.learning_rate, batch_size=config.batch_size, lstm_layers=config.num_layers, 
                          lstm_layer_size=config.lstm_size, fc_layer_size=config.ann_hidden, sequence_length=config.sequence_length, 
                          n_channels=config.n_channels, path_checkpoint=config.path_checkpoint, spacial_note='')


            tf.global_variables_initializer().run()

            if config.restore_model:
                saver.restore(session, config.path_checkpoint)
                print("Model restored from file: %s" % config.path_checkpoint)

            cost = []
            plot_x = []
            plot_y1 = []
            plot_y2 = []
            iter_train = int(train.shape[0]/config.shift)
            iter_test = int(test.shape[0]/config.shift)
            print("Training set MSE")
            print("No epoches: ", config.epochs, "No itr: ", iter_train)
            __start = time.time()
            for ep in range(config.epochs):
                session.run(self.reset_op)
                training_generator = train.batch_generator(config)
                testing_generator = test.batch_generator(config)

                h1 = []
                t1 = []
                engine_id = 1

                try:
                    old_engine_id = 0
                    while True:
                        ## training ##
                        train_gen = next(training_generator)
                        new_engine_id = train_gen[0][0,0,1]
                        if (old_engine_id != new_engine_id) :
                            session.run(self.reset_op)
                            #if (old_engine_id != 0):
                                #print ("eng_ids: ",old_engine_id, new_engine_id, "  RMSE train:", np.sqrt(np.mean(np.square(h1))))
                            old_engine_id = new_engine_id
                        batch_x, batch_y = train_gen[1], train_gen[2]                   
                        session.run([self.optimizer, self.update_op],
                                    feed_dict={self.X: batch_x, self.Y: batch_y, keep_prob: 0.7, learning_rate_: config.learning_rate})
                        h_i = self.h.eval(feed_dict={self.X: batch_x, self.Y: batch_y, 
                                                     keep_prob: 1.0, learning_rate_: config.learning_rate})
                        cost.append(np.square(h_i))
                        h1.append(h_i)
                except StopIteration:
                    pass

                rmse_train = np.sqrt(np.mean(np.square(h1)))

                y_pred = []

                try:
                    old_engine_id = 0
                    while True:
                        test_gen = next(testing_generator)
                        new_engine_id = test_gen[0][0,0,1]
                        if old_engine_id != new_engine_id:
                            session.run(self.reset_op)
                            #if (old_engine_id != 0):
                                #print ("eng_ids: ",old_engine_id, new_engine_id, "  RMSE train:", np.sqrt(np.mean(np.square(t1))))
                            old_engine_id = new_engine_id
                        x_test_batch, y_test_batch = test_gen[1], test_gen[2]
                        h_i, u = session.run([self.h, self.update_op], feed_dict={self.X: x_test_batch, self.Y: y_test_batch, 
                                                                                  keep_prob: 1.0, learning_rate_: config.learning_rate})
                        t1.append(h_i)
                except StopIteration:
                    pass

                rmse_test = np.sqrt(np.mean(np.square(t1)))

                plot_x.append(ep)
                plot_y1.append(rmse_train)
                plot_y2.append(rmse_test)

                time_per_ep = (time.time() - __start)
                time_remaining = ((config.epochs - ep) * time_per_ep) / 3600
                print("LSTM", "epoch:", ep, "RMSE-train:", rmse_train, "RMSE-test", rmse_test, "lr", config.learning_rate,
                      "\ttime/epoch:", round(time_per_ep, 2), "\ttime_remaining: ",
                      int(time_remaining), " hr:", round((time_remaining % 1) * 60, 1), " min", "\ttime_stamp: ",
                      datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S"))
                __start = time.time()

                if ep % 20 == 0 and ep != 0:
                    save_path = saver.save(session, config.path_checkpoint)
                    if os.path.exists(config.path_checkpoint + '.meta'):
                        print("Model saved to file: %s" % config.path_checkpoint)
                    else:
                        print("NOT SAVED!!!", config.path_checkpoint)
                        plt.plot(plot_x, plot_y1, 'bo', plot_x, plot_y2, 'go')
                        plt.show()

                if ep % 100 == 0 and ep != 0: 
                    config.learning_rate = config.learning_rate / 2
                    

                #plt.plot(plot_x, plot_y1, 'bo', plot_x, plot_y2, 'go')
                #plt.show()

    def predict(self, predict, config):   
   
        with tf.Session() as session:
            
            keep_prob = self.keep_prob
            saver = tf.train.Saver()
            saver.restore(session, config.path_checkpoint)
            print("Model restored from file: %s" % config.path_checkpoint)

            print("Prediction for submit...")
            x_predict = predict

            full_prediction = []

            print("#of validation points:", x_predict.shape[0], "#datapoints covers from minibatch:",
                  config.batch_size * config.sequence_length)

            predict_generator = x_predict.batch_generator(config)

            try:
                old_engine_id = 0                
                while True:
                    test_gen = next(predict_generator)                    
                    new_engine_id = test_gen[0][0,0,1]
                    if old_engine_id != new_engine_id:
                        session.run(self.reset_op)
                        old_engine_id = new_engine_id
                    x_validate_batch, y_validate_batch = test_gen[1], test_gen[2]
                    __y_pred, u = session.run([self.output, self.update_op], feed_dict={self.X: x_validate_batch, self.Y: y_validate_batch, 
                                                                                        keep_prob: 1.0})
                    
                    full_prediction.append(__y_pred[0])
            except StopIteration:
                pass        

            full_prediction = np.array(full_prediction)
            full_prediction = full_prediction.ravel()
        
        return full_prediction

    
    
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

