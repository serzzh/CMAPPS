import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from data_processing import MAXLIFE



def lstm_sampling (data, y=None, timesteps=30, lag=1):
    # split into samples 
    samples = list()
    targets = list()
    length = timesteps
    n = data.shape[0]
    

    for i in range(0, n-length+1, lag):
        sample = data[i:i+length]
        samples.append(sample)
        if y is not None:
            target = y[i:i+length]
            targets.append(target)
    
    print("Samples length",len(samples))
    print("Targets length",len(targets))
    
    # convert list of arrays into 2d array
    data = np.stack(samples)
    if y is not None:
        y = np.stack(targets)
        
    return data, y

def batch_generator(x_train, y_train, batch_size, sequence_length, online=False, online_shift=1):
    """
    Generator function for creating sequential batches of training-data
    """
    num_x_sensors = x_train.shape[2]
    num_train = x_train.shape[0]
    idx = 0

    # Infinite loop.
    while True:
        if idx > num_train - batch_size+1:
            idx = 0
        #if online is False:
        #    idx = np.random.randint(num_train - batch_size+1)
        
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_x_sensors)
        x_batch = np.zeros(shape=x_shape, dtype=np.float32)
        #print("x_shape %s, idx %s" % (x_shape, idx))
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
        yield (x_batch, y_batch)



