from data_loader.data_utils import Dataset, gen_batch
from model.model import STGCN_Model, STGCNB_Model
from os.path import join as pjoin
from utils.math_utils import evaluation, MAPE, MAE, RMSE

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import time
import tqdm
import math


def custom_loss(y_true, y_pred) -> tf.Tensor:
    # return tf.reduce_mean(tf.math.squared_difference(y_true, y_pred))
    return tf.nn.l2_loss(y_true - y_pred)

def model_train(inputs: Dataset, graph_kernel, blocks, args, sum_path='./output/tensorboard'):
    '''
    Train the base model.
    :param inputs: instance of class Dataset, data source for training.
    :param blocks: list, channel configs of st_conv blocks.
    :param args: instance of class argparse, args for training.
    '''
    n_his, n_pred = args.n_his, args.n_pred
    Ks, Kt = args.ks, args.kt
    batch_size, epochs, opt = args.batch_size, args.epochs, args.opt
    train_data = inputs.get_data("train")
    val_data = inputs.get_data("val")
    steps_per_epoch = math.ceil(train_data.shape[0]/batch_size)

    model = STGCN_Model(train_data.shape[1:], batch_size, graph_kernel, n_his, Ks, Kt, blocks, "GLU", "layer", 0.1)
    lr_func = keras.optimizers.schedules.PiecewiseConstantDecay(
        [int(epochs/4)*steps_per_epoch, int(2*epochs/4)*steps_per_epoch, int(3*epochs/4)*steps_per_epoch],
        [args.lr, 0.7*args.lr, 0.4*args.lr, 0.1*args.lr]
    )
    if opt == "RMSprop":
        optimizer = keras.optimizers.RMSprop(lr_func)
    elif opt == "Adam":
        optimizer = keras.optimizers.Adam(lr_func)
    else:
        raise NotImplementedError(f'ERROR: optimizer "{opt}" is not implemented')

    model.compile(optimizer=optimizer, loss=custom_loss)

    print("Training Model on Data")
    best_val_mae = np.inf
    weights_file = f"best_nblocks_{len(blocks)}_nhis_{args.n_his}_opt_{opt}_ks_{Ks}_kt_{Kt}_lr_{args.lr}.h5"
    for epoch in range(1, epochs+1):
        print(f"\nEpoch {epoch} / {epochs}")
        train_loss = 0
        start_time = time.time()
        for batch in tqdm.tqdm(gen_batch(train_data, batch_size, dynamic_batch=True, shuffle=True), total=steps_per_epoch):
            with tf.GradientTape() as tape:
                y_pred = model(batch[:, :n_his, :, :], training=True)
                loss = custom_loss(batch[:, n_his:n_his+1, :, :], y_pred)
                gradients = tape.gradient(loss, model.trainable_weights)
            train_loss += loss
            model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        print(f"Epoch {epoch} finished!", "Training Time:", f"{time.time()-start_time}s")
        print("Train L2 Loss: ", train_loss.numpy())

        val_train = val_data[:, :n_his, :, :]
        val_preds = model(val_train, training=False)
        val_loss = custom_loss(val_data[:, n_his:n_his+1, :, :], val_preds)
        print("Val L2 Loss: ", val_loss.numpy())
        for i, col in enumerate(args.datafiles):
            v = val_data[:, n_his:n_his+1, :, i:i+1]*inputs.std+inputs.mean
            v_ = val_preds[:, :, :, i:i+1]*inputs.std+inputs.mean
            print(f"{col}\t MAE:", "%.4f" % MAE(v, v_), end="\t")
            print(f"MAPE:", "%.4f" % MAPE(v, v_), end="\t")
            print(f"RMSE:", "%.4f" % RMSE(v, v_))

        v = val_data[:, n_his:n_his+1, :, :1]*inputs.std+inputs.mean
        v_ = val_preds[:, :, :, :1]*inputs.std+inputs.mean
        mae = MAE(v, v_)

        if mae < best_val_mae:
            print(f"Saving weights for model! (Based on MAE of {args.datafiles[0]})")
            best_val_mae = mae
            model.save_weights(weights_file)
        
    model.load_weights(weights_file)
    x_test = inputs.get_data("test")[:, :n_his, :, :]
    y_test = inputs.get_data("test")[:, n_his:n_his+1, :, :]
    preds = model(x_test)

    # print(np.array(inputs.get_data("test")[:1, n_his:n_his+1, :, :], dtype=np.float)*inputs.std+inputs.mean)
    # print(model(inputs.get_data("test")[:1, :n_his, :, :])*inputs.std+inputs.mean)

    print("\nTEST")
    for i, col in enumerate(args.datafiles):
        test_m = evaluation(y_test[:,:,:,i:i+1], preds[:,:,:,i:i+1], inputs.get_stats())
        print(f"{col} (MAPE, MAE, RMSE):", test_m)
    return float(test_m[1])