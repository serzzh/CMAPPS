import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import random
#from utils import TsConf

MAXLIFE = 130
SCALE = 1
RESCALE = 1
true_rul = []
test_engine_id = 0
training_engine_id = 0

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
    
    def lstm_sampling(self, config):        
        # split into samples groupwise

        samples, targets = list(), list() 
        for name, group in self.groupby(self.groupids):
            n = group.shape[0]
            timesteps = config.sequence_length
            lag = config.shift           

            if n ==0:
                continue

            X, y = group.extract_target()
            
            start = n % timesteps if config.trim_left else 0
            
            for i in range(start, n-timesteps+1, lag):
                sample = X[i:i+timesteps]
                samples.append(sample)
                if y is not None:
                    target = y[i:i+timesteps]
                    targets.append(target)
        
        print("Samples length: ",len(samples),", Targets length: ",len(targets))
        
        # convert list of arrays into 2d array
        if samples:
            data = np.stack(samples)
        else:
            data = None
        if targets:
            y = np.stack(targets)
            
        return data, y

    def batch_generator(self, config):
        """
        Generator function for creating sequential batches of training-data
        """
        sequence_length = config.sequence_length
        batch_size = config.batch_size
        
        x_train, y_train = self.lstm_sampling(config)
        
        num_x_sensors = x_train.shape[2]
        num_train = x_train.shape[0]
        idx = 0
        #print("num_train %s, idx %s, x_batch.shape %s" % (num_train, idx, batch_size))

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

            if not config.random:
                idx = idx + 1  # check if its nee to be idx=idx+1
            #print("num_train %s, idx %s, x_batch.shape %s" % (num_train, idx, x_batch.shape))
            yield (x_batch[:,:,0:3], x_batch[:,:,3:], y_batch)
    
    def cluster(self, config):
        self["CLUSTER"] = KMeans(config.cmapss_n_clusters, random_state=0).fit_predict(self[config.opset])
        return self
    
    def data_norm(self, config):
        ### Standard Normal ###
        self[config.features] = self.groupby(config.norm_groupids)[config.features].transform(
                lambda x: (x - x.mean()) / (x.std() if x.std()>0 else 1))        
        return self
    
    def ewma(self, config):
        self[config.select_feat] = self.groupby(config.groupids)[config.select_feat].transform(lambda x: x.ewm(com=20).mean())
        return self


def kink_RUL(cycle_list, max_cycle):
    '''
    Piecewise linear function with zero gradient and unit gradient

            ^
            |
    MAXLIFE |-----------
            |            \
            |             \
            |              \
            |               \
            |                \
            |----------------------->
    '''
    knee_point = max_cycle - MAXLIFE
    kink_RUL = []
    stable_life = MAXLIFE
    for i in range(0, len(cycle_list)):
        if i < knee_point:
            kink_RUL.append(MAXLIFE)
        else:
            tmp = kink_RUL[i - 1] - (stable_life / (max_cycle - knee_point))
            kink_RUL.append(tmp)

    return kink_RUL


def compute_rul_of_one_id(FD00X_of_one_id, max_cycle_rul=None):
    '''
    Enter the data of an engine_id of train_FD001 and output the corresponding RUL (remaining life) of these data.
    type is list
    '''

    cycle_list = FD00X_of_one_id['cycle'].tolist()
    if max_cycle_rul is None:
        max_cycle = max(cycle_list)  # Failure cycle
    else:
        max_cycle = max(cycle_list) + max_cycle_rul
        # print(max(cycle_list), max_cycle_rul)

    # return kink_RUL(cycle_list,max_cycle)
    return kink_RUL(cycle_list, max_cycle)


def compute_rul_of_one_file(FD00X, id='engine_id', RUL_FD00X=None):
    '''
    Input train_FD001, output a list
    '''
    rul = []
    # In the loop train, each id value of the 'engine_id' column
    if RUL_FD00X is None:
        for _id in set(FD00X[id]):
            rul.extend(compute_rul_of_one_id(FD00X[FD00X[id] == _id]))
        return rul
    else:
        rul = []
        for _id in set(FD00X[id]):
            # print("#### id ####", int(RUL_FD00X.iloc[_id - 1]))
            true_rul.append(int(RUL_FD00X.iloc[_id - 1]))
            rul.extend(compute_rul_of_one_id(FD00X[FD00X[id] == _id], int(RUL_FD00X.iloc[_id - 1])))
        return rul


    





