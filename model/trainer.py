from data_loader.data_utils import Dataset, gen_batch
from model.model import STGCNA_Model, STGCNB_Model, STGCNC_Model
from utils.math_utils import custom_loss, MAPE, MAE, RMSE, Pearsonr, Rsquared
from model.tester import model_inference

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import time
import tqdm
import math
import os

def model_train(inputs: Dataset, graph_kernel, blocks, args):
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
    train_length = train_data.shape[0]
    val_length = val_data.shape[0]

    train_log_dir = os.path.join(args.log_dir, 'train')
    test_log_dir = os.path.join(args.log_dir, 'test')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    if args.retrain:
        model = keras.models.load_model(args.model_path, custom_objects={'custom_loss': custom_loss})
    elif args.model == 'A':
        model = STGCNA_Model(train_data.shape[1:], graph_kernel, n_his, Ks, Kt, blocks, "GLU", "layer", 0.1)
    elif args.model == 'B':
        model = STGCNB_Model(train_data.shape[1:], graph_kernel, n_his, Ks, Kt, blocks, "GLU", "layer", 0.1)
    elif args.model == 'C':
        model = STGCNC_Model(train_data.shape[1:], graph_kernel, n_his, Ks, Kt, blocks, "GLU", "layer", 0.1)
    else:
        raise NotImplementedError('STGCN model has only A, B and C types.')

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
    for epoch in range(1, epochs+1):
        print(f"\nEpoch {epoch} / {epochs}")
        train_loss = 0
        start_time = time.time()
        for batch in tqdm.tqdm(gen_batch(train_data, batch_size, dynamic_batch=True, shuffle=True), total=steps_per_epoch):
            with tf.GradientTape() as tape:
                y_pred = model(batch[:, :n_his, :, :], training=True)
                loss = model.loss(batch[:, n_his:n_his+1, :, :], y_pred)
                gradients = tape.gradient(loss, model.trainable_weights)
            train_loss += loss
            model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        print(f"Epoch {epoch} finished!", "Training Time:", f"{time.time()-start_time:.4f}s")
        print("Train L2 Loss: %.4f" % (train_loss.numpy()))
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.numpy()*2/train_length, step=epoch)

        val_train = val_data[:, :n_his, :, :]
        val_preds = model(val_train, training=False).numpy()
        val_loss = custom_loss(val_data[:, n_his:n_his+1, :, :], val_preds)
        print("Val L2 Loss: %.4f" % (val_loss.numpy()))
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss.numpy()*2/val_length, step=epoch)
        for i, col in enumerate(args.datafiles):
            v = val_data[:, n_his:n_his+1, :, i:i+1]*inputs.std+inputs.mean
            v_ = val_preds[:, :, :, i:i+1]*inputs.std+inputs.mean
            print(col, end='\t')
            for m in zip(['MAE', 'MAPE', 'RMSE', 'Corr', 'R2'], [MAE(v, v_), MAPE(v, v_), RMSE(v, v_), Pearsonr(v, v_), Rsquared(v, v_)]):
                print(m[0], "%.4f" % m[1], end="\t")
                with test_summary_writer.as_default():
                    tf.summary.scalar(f'{col}_{m[0]}', m[1], step=epoch)
            print()

        v = val_data[:, n_his:n_his+1, :, 0]*inputs.std+inputs.mean
        v_ = val_preds[:, :, :, 0]*inputs.std+inputs.mean
        mae = MAE(v, v_)
        if mae < best_val_mae:
            print(f"Saving weights for model! (Based on MAE of {args.datafiles[0]})")
            best_val_mae = mae
            model.save(args.model_path)

    keras.backend.clear_session()
    model = keras.models.load_model(args.model_path, custom_objects={'custom_loss':custom_loss})
    y_test, test_evaluation = model_inference(model, inputs, batch_size, n_his, n_pred)

    print("\nTEST evaluations:")
    for key in test_evaluation.keys():
        mets = test_evaluation[key]
        print("\nTime Step", key)
        for col, met in zip(args.datafiles, mets):
            print("%s\tMAPE %.4f%%, MAE %.4f, RMSE %.4f, Corr %.4f, R2 %.4f" % (col, *met))

    np.save(os.path.join(args.output_dir, "test.npy"), inputs.get_data('test'))
    np.save(os.path.join(args.output_dir, "pred.npy"), y_test)
    return y_test, test_evaluation