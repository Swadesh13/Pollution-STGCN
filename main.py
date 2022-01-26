import argparse
import os
import datetime
from model.trainer import model_train
# from model.tester import model_test
from data_loader.data_utils import *
from utils.math_graph import *
import tensorflow as tf

if tf.test.is_built_with_cuda():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--n_his', type=int, default=12)
parser.add_argument('--n_pred', nargs='+', type=int, default=[9], help='List of all times at which prediction is required')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--ks', type=int, default=3)
parser.add_argument('--kt', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='RMSprop')
parser.add_argument('--datafiles', nargs='+', default=['PM2.5'], help="list of all data filenames. \
    Assumes file ends with .csv and 1st row contains station names.")
parser.add_argument('--coords', type=str, default='coords.json')
parser.add_argument('--model', type=str, default='B', choices=['A', 'B', 'C'])
parser.add_argument('--output', type=str, default='output', help='output dir')

args = parser.parse_args()
print(f'Training configs: {args}')

channels = len(args.datafiles)
Ks, Kt = args.ks, args.kt
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
args.output_dir = os.path.join(args.output, current_time)
args.model_path = os.path.join(args.output_dir, 'model')
args.log_dir = os.path.join(args.output_dir, 'logs')
os.makedirs(args.model_path, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)

# blocks: settings of channel size in st_conv_blocks / bottleneck design
if args.model == 'A':
    blocks = [[channels, 32, 64, 128], [64, 64, 128]] # for STGCN-A
elif args.model == 'B':
    blocks = [[channels, 32, 64], [64, 32, 128]] # for STGCN-B
else:
    blocks = [[channels, 32, 32], [64, 128, 128]] # for STGCN-C

mat, cols = create_data_matrix(args.datafiles)
n = len(cols)

W = create_weight_matrix(args.coords, cols, np.inf)
# Calculate graph kernel
L = scaled_laplacian(W)
# Alternative approximation method: 1st approx - first_approx(W, n).
Lk = cheb_poly_approx(L, Ks, n)

PeMS = data_gen(mat, n, args.n_his + max(args.n_pred), channels)
# print(f'>> Loading Train dataset with Mean: {PeMS.mean:.2f}, STD: {PeMS.std:.2f}')
print("Training Data shape:", PeMS.get_data("train").shape)
print("Validation Data shape:", PeMS.get_data("val").shape)

if __name__ == '__main__':
    model_train(PeMS, Lk, blocks, args)
    # model_test(PeMS, PeMS.get_len('test'), n_his, n_pred, args.inf_mode)
