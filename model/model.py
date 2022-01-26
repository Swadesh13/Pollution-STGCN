import tensorflow as tf
import tensorflow.keras as keras
from .layers import SConvBlock, TConvBlock, STConvBlock, OutputLayer

class STGCNA_Model(keras.Model):
    '''
    Spatio-Temporal Graph Convolutional Neural Model.
    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param input_shape: list, [time_step, n_route, c_in].
    :param graph_kernel: tensor, [n_route, Ks*n_route].
    :param n_his: int, size of historical records for training.
    :param Ks: int, kernel size of spatial convolution.
    :param Kt: int, kernel size of temporal convolution.
    :param blocks: list, channel configs of st_conv blocks.
    :param act_func: str, activation function.
    :param norm: str, normalization function.
    :param dropout: float, dropout ratio.
    :param pad: string, Temporal layer padding - VALID or SAME.
    :return: tensor, [batch_size, time_step, n_route, c_out].
    '''
    def __init__(self, input_shape, graph_kernel, n_his, Ks, Kt, blocks, act_func, norm, dropout, pad = "VALID", **kwargs):
        super(STGCNA_Model, self).__init__(name = "STGCN" ,**kwargs)
        self.n_his = n_his
        Ko = n_his
        assert len(blocks) == 2, "STGCN-A model allows only 1 group of Spatio-Conv layers & 1 group of Temporal-Conv layers"
        # Format - All spatio layers, followed by all temporal layers - blocks contains output units for each layer
        self.sconv_block = SConvBlock(graph_kernel, Ks, input_shape[-1], blocks[0], norm, dropout)
        self.tconv_block = TConvBlock(Kt, blocks[0][-1], blocks[1], act_func, norm, dropout, pad)
        if pad=='VALID':
            Ko -= len(blocks[1])*(Kt - 1)
        # Output Layer
        if Ko > 1:
            self.output_layer = OutputLayer(Ko, input_shape[1], blocks[-1][-1], input_shape[-1], act_func, norm)
        else:
            raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{Ko}".')

    @tf.function
    def call(self, x:tf.Tensor):
        x = tf.cast(x, tf.float64)
        x = self.sconv_block(x)
        x = self.tconv_block(x)
        y = self.output_layer(x)
        return y

    def model(self, shape): # To get brief summary
        x = keras.Input(shape=shape, batch_size=32)
        return keras.Model(inputs=[x], outputs=self.call(x))


class STGCNB_Model(keras.Model):
    '''
    Spatio-Temporal Graph Convolutional Neural Model.
    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param input_shape: list, [time_step, n_route, c_in].
    :param graph_kernel: tensor, [n_route, Ks*n_route].
    :param n_his: int, size of historical records for training.
    :param Ks: int, kernel size of spatial convolution.
    :param Kt: int, kernel size of temporal convolution.
    :param blocks: list, channel configs of st_conv blocks.
    :param act_func: str, activation function.
    :param norm: str, normalization function.
    :param dropout: float, dropout ratio.
    :param pad: string, Temporal layer padding - VALID or SAME.
    :return: tensor, [batch_size, time_step, n_route, c_out].
    '''
    def __init__(self, input_shape, graph_kernel, n_his, Ks, Kt, blocks, act_func, norm, dropout, pad = "VALID", **kwargs):
        super(STGCNB_Model, self).__init__(name = "STGCN" ,**kwargs)
        self.n_his = n_his
        self.stconv_blocks = []
        Ko = n_his
        # ST Blocks
        for channels in blocks:
            self.stconv_blocks.append(STConvBlock(graph_kernel, Ks, Kt, channels, act_func, norm, dropout, pad))
            if pad == "VALID":
                Ko -= 2 * (Kt - 1)
        # Output Layer
        if Ko > 1:
            self.output_layer = OutputLayer(Ko, input_shape[1], blocks[-1][-1], input_shape[-1], act_func, norm)
        else:
            raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{Ko}".')

    @tf.function
    def call(self, x:tf.Tensor):
        x = tf.cast(x, tf.float64)
        for block in self.stconv_blocks:
            x = block(x)
        y = self.output_layer(x)
        return y

    def model(self, shape): # To get brief summary
        x = keras.Input(shape=shape, batch_size=32)
        return keras.Model(inputs=[x], outputs=self.call(x))


class STGCNC_Model(keras.Model):
    '''
    Spatio-Temporal Graph Convolutional Neural Model.
    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param input_shape: list, [time_step, n_route, c_in].
    :param graph_kernel: tensor, [n_route, Ks*n_route].
    :param n_his: int, size of historical records for training.
    :param Ks: int, kernel size of spatial convolution.
    :param Kt: int, kernel size of temporal convolution.
    :param blocks: list, channel configs of st_conv blocks.
    :param act_func: str, activation function.
    :param norm: str, normalization function.
    :param dropout: float, dropout ratio.
    :return: tensor, [batch_size, time_step, n_route, c_out].
    '''
    def __init__(self, input_shape, graph_kernel, n_his, Ks, Kt, blocks, act_func, norm, dropout, **kwargs):
        super(STGCNC_Model, self).__init__(name = "STGCNB" ,**kwargs)
        self.n_his = n_his
        self.stconv_blocks = []
        # ST Blocks
        self.stconv_blocks.append([TConvBlock(Kt, input_shape[-1], blocks[0], act_func, norm, dropout, 'SAME'), SConvBlock(graph_kernel, Ks, input_shape[-1], blocks[0], norm, dropout)])
        for i in range(1, len(blocks)):
            # c_in - twice due to concat
            self.stconv_blocks.append([TConvBlock(Kt, 2*blocks[i-1][-1], blocks[i], act_func, norm, dropout, 'SAME'), SConvBlock(graph_kernel, Ks, 2*blocks[i-1][-1], blocks[i], norm, dropout)])
        # Output Layer
        self.output_layer = OutputLayer(n_his, input_shape[1], blocks[-1][-1]*2, input_shape[-1], act_func, norm)

    @tf.function
    def call(self, x:tf.Tensor):
        x = tf.cast(x, tf.float64)
        for block in self.stconv_blocks:
            x1 = block[0](x)
            x2 = block[1](x)
            x = tf.concat([x1, x2], axis=-1)
        y = self.output_layer(x)
        return y

    def model(self, shape): # To get brief summary
        x = keras.Input(shape=shape, batch_size=32)
        return keras.Model(inputs=[x], outputs=self.call(x))