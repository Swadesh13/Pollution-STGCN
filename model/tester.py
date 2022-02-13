from data_loader.data_utils import gen_batch
from utils.math_utils import evaluation, custom_loss

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import time


def multi_pred(model, seq, batch_size, n_his, n_pred, dynamic_batch=True):
    '''
    Multi_prediction function.
    :param model: tf.keras Model.
    :param y_pred: placeholder.
    :param seq: np.ndarray, [len_seq, n_frame, n_route, C_0].
    :param batch_size: int, the size of batch.
    :param n_his: int, size of historical records for training.
    :param n_pred: list, time steps of prediction. Eg: [1,3] - print prediction evaluation at 1 and 3 time_steps.
    :param dynamic_batch: bool, whether changes the batch size in the last one if its length is less than the default.
    :return pred_array : np.ndarray, [len(seq), max(n_pred), n_route, channels].
    '''
    pred_array = np.zeros((seq.shape[0], max(n_pred), *seq.shape[2:]))
    batch_size = min(batch_size, len(seq))
    for i, data in enumerate(gen_batch(seq, batch_size, dynamic_batch=dynamic_batch)):
        # Note: use np.copy() to avoid the modification of source data.
        test = np.copy(data[:, 0:n_his, :, :])
        for j in range(max(n_pred)):
            pred = model(test).numpy()
            pred = np.reshape(pred, (test.shape[0], test.shape[2], test.shape[3]))
            test[:, 0:n_his - 1, :, :] = test[:, 1:n_his, :, :]
            test[:, n_his - 1, :, :] = pred
            pred_array[i*batch_size:(i+1)*batch_size, j, :, :] = pred
    return pred_array


def model_inference(model, inputs, batch_size, n_his, n_pred):
    '''
    Model inference function.
    :param model: tf Model.
    :param inputs: instance of class Dataset, data source for inference.
    :param batch_size: int, the size of batch.
    :param n_his: int, the length of historical records for training.
    :param n_pred: list, time steps of prediction. Eg: [1,3] - print prediction evaluation at 1 and 3 time_steps.
    :return eval_test: dict, evaluation results per data per time step.
    :return y_test: npp.ndarray, model prediction for max(n_pred) steps.
    '''
    x_test, x_stats = inputs.get_data('test'), inputs.get_stats()

    y_test = multi_pred(model, x_test, batch_size, n_his, n_pred)
    eval_test = {}
    for i in n_pred:
        eval_test[i] = []
        for j in range(y_test.shape[-1]):
            eval_test[i].append(evaluation(x_test[:, n_his+i-1, :, j], y_test[:, i-1, :, j], x_stats).tolist())
    return y_test, eval_test


def model_test(inputs, batch_size, n_his, n_pred, load_path, columns):
    '''
    Load and test saved model from the checkpoint.
    :param inputs: instance of class Dataset, data source for test.
    :param batch_size: int, the size of batch.
    :param n_his: int, the length of historical records for training.
    :param n_pred: list, time steps of prediction. Eg: [1,3] - print prediction evaluation at 1 and 3 time_steps.
    :param load_path: str, the path of loaded model.
    :param columns: list, same as args.datafiles.
    '''
    start_time = time.time()
    model = keras.models.load_model(load_path, {"custom_loss": custom_loss})
    print(f'>> Loading saved model from {load_path} ...')

    x_test, x_stats = inputs.get_data('test'), inputs.get_stats()

    y_test = multi_pred(model, x_test, batch_size, n_his, n_pred)
    eval_test = {}
    for i in n_pred:
        eval_test[i] = []
        for j in range(y_test.shape[-1]):
            eval_test[i].append(evaluation(x_test[:, n_his+i-1, :, j], y_test[:, i-1, :, j], x_stats).tolist())

    for key in eval_test.keys():
        mets = eval_test[key]
        for col, met in zip(columns, met):
            print("%s\tMAPE %.4f, MAE %.4f, RMSE %.4f, Corr %.4f, R2 %.4f" % (col, *mets))
    
    print(f'Model Test Time {time.time() - start_time:.3f}s')
    print('Testing model finished!')
    return y_test, eval_test