import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

def z_score(x, mean, std):
    '''
    Z-score normalization function: $z = (X - \mu) / \sigma $,
    where z is the z-score, X is the value of the element,
    $\mu$ is the population mean, and $\sigma$ is the standard deviation.
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score normalized array.
    '''
    return (x - mean) / std


def z_inverse(x, mean, std):
    '''
    The inverse of function z_score().
    :param x: np.ndarray, input to be recovered.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score inverse array.
    '''
    return x * std + mean


def MAPE(v, v_):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAPE averages on all elements of input.
    '''
    return np.mean(np.abs(v_ - v) / (v + 1e-5))*100


def RMSE(v, v_):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, RMSE averages on all elements of input.
    '''
    return np.sqrt(np.mean((v_ - v) ** 2))


def MAE(v, v_):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAE averages on all elements of input.
    '''
    return np.mean(np.abs(v_ - v))


def Pearsonr(v, v_):
    '''
    Pearson correlation.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, Pearson's r over all elements of input.
    '''
    return pearsonr(v.flatten(), v_.flatten())[0]


def Rsquared(v, v_):
    '''
    R-squared value.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, R-squared value over all elements of input.
    '''
    return r2_score(v.flatten(), v_.flatten())


def evaluation(y, y_, x_stats) -> np.ndarray:
    '''
    Evaluation function: interface to calculate MAPE, MAE, RMSE, Pearson's r and R-squared between ground truth and prediction.
    Extended version: multi-step prediction can be calculated by self-calling.
    :param y: np.ndarray or int, ground truth.
    :param y_: np.ndarray or int, prediction.
    :param x_stats: dict, paras of z-scores (mean & std).
    :return: np.ndarray, averaged metric values (MAPE, MAE, RMSE).
    '''
    v = z_inverse(y, x_stats['mean'], x_stats['std'])
    v_ = z_inverse(y_, x_stats['mean'], x_stats['std'])
    return np.array([MAPE(v, v_), MAE(v, v_), RMSE(v, v_), Pearsonr(v, v_), Rsquared(v, v_)])


def custom_loss(y_true, y_pred) -> tf.Tensor:
    # return tf.reduce_mean(tf.math.squared_difference(y_true, y_pred))
    return tf.nn.l2_loss(y_true - y_pred)