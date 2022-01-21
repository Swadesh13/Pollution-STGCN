from utils.math_utils import z_score
import os
import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d

class Dataset(object):
    def __init__(self, data, stats):
        self.__data = data
        self.mean = stats['mean']
        self.std = stats['std']

    def get_data(self, type):
        return self.__data[type]

    def get_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def get_len(self, type):
        return len(self.__data[type])

    def z_inverse(self, type):
        return self.__data[type] * self.std + self.mean


def create_data_matrix(files):
    '''
    Create the Data matrix containing data from the intersection of stations of the filenames.
    :param files: list, filenames.
    :return: np.array, data matrix
    '''
    data = []
    for filename in files:
        data.append(pd.read_csv(os.path.join("dataset", filename+".csv")))

    cols = data[0].columns
    for d in data:
        cols = list(set(cols) & set(d.columns))
    print(f"Selecting data for the following {len(cols)} stations:", cols)
    
    mat = np.zeros((len(data[0]), len(cols)*len(files)), np.float)
    ind = 0
    for col in cols:
        for d in data:
            mat[:, ind] = d[col].values
            ind += 1
    
    return mat, cols


def seq_gen(len_seq, data_seq, offset, n_frame, n_route, C_0=1):
    '''
    Generate data in the form of standard sequence unit.
    :param len_seq: int, the length of target date sequence.
    :param data_seq: np.ndarray, source data / time-series.
    :param offset:  int, the starting index of different dataset type.
    :param n_frame: int, the number of frame within a standard sequence unit, which contains n_his = 12 and n_pred = 9 (3 /15 min, 6 /30 min & 9 /45 min).
    :param n_route: int, the number of routes in the graph.
    :param C_0: int, the size of input channel.
    :return: np.ndarray, [len_seq, n_frame, n_route, C_0].
    '''
    n_slot = len_seq - n_frame + 1

    tmp_seq = np.zeros((n_slot, n_frame, n_route, C_0))
    for i in range(n_slot):
        sta = i + offset
        end = sta + n_frame
        d = data_seq[sta:end, :]
        tmp_seq[i, :, :, :] = np.reshape(d, (*(d.shape[:-1]), n_route, C_0))
    return tmp_seq

#* Training data only. Only test??
def data_gen(data_seq, n_route, n_frame=21, C_0=1):
    '''
    Source file load and dataset generation.
    :param data_seq: np.array, 2D data matrix.
    :param n_route: int, the number of routes in the graph.
    :param n_frame: int, the number of frame within a standard sequence unit, which contains n_his = 12 and n_pred = 9 (3 /15 min, 6 /30 min & 9 /45 min).
    :param C_0: int, the size of input channel.
    :return: dict, dataset that contains training, validation and test with stats.
    '''
    # generate training, validation and test data

    l = data_seq.shape[0]
    n_train, n_val, n_test = int(l*.92), int(l*.08), int(l*.08)

    seq_train = seq_gen(n_train, data_seq, 0, n_frame, n_route, C_0)
    seq_val = seq_gen(n_val, data_seq, n_train, n_frame, n_route, C_0)
    seq_test = seq_gen(n_test, data_seq, n_train, n_frame, n_route, C_0)

    # x_stats: dict, the stats for the train dataset, including the value of mean and standard deviation.
    x_stats = {'mean': np.mean(data_seq[:len(seq_train)]), 'std': np.std(data_seq[:len(seq_train)])}

    # x_train, x_val, x_test: np.array, [sample_size, n_frame, n_route, channel_size].
    x_train = z_score(seq_train, x_stats['mean'], x_stats['std'])
    x_val = z_score(seq_val, x_stats['mean'], x_stats['std'])
    x_test = z_score(seq_test, x_stats['mean'], x_stats['std'])

    x_data = {'train': x_train, 'val': x_val, 'test': x_test}
    dataset = Dataset(x_data, x_stats)
    return dataset


def gen_batch(inputs, batch_size, dynamic_batch=False, shuffle=False):
    '''
    Data iterator in batch.
    :param inputs: np.ndarray, [len_seq, n_frame, n_route, C_0], standard sequence units.
    :param batch_size: int, the size of batch.
    :param dynamic_batch: bool, whether changes the batch size in the last batch if its length is less than the default.
    :param shuffle: bool, whether shuffle the batches.
    '''
    len_inputs = len(inputs)

    if shuffle:
        idx = np.arange(len_inputs)
        np.random.shuffle(idx)

    for start_idx in range(0, len_inputs, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > len_inputs:
            if dynamic_batch:
                end_idx = len_inputs
            else:
                break
        if shuffle:
            slide = idx[start_idx:end_idx]
        else:
            slide = slice(start_idx, end_idx)

        yield inputs[slide]
