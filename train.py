import argparse
import yaml
import os
from utils import str2bool, set_seed, create_dirs_if_not_exist
from core.train_solver import TrainSolver


parser = argparse.ArgumentParser()

# basic opts.
parser.add_argument('--seed', type=int, default=2408, help='random seed')
parser.add_argument('--use_gpu', type=str2bool, default='True', help='whether to use gpu')
parser.add_argument('--log_dir', type=str, default='logs/')
parser.add_argument('--checkpoint_dir', type=str, default='models/')
parser.add_argument('--sample_dir', type=str, default='samples/')
parser.add_argument('--result_root', type=str, default='results/')
# 3dmm opts
parser.add_argument('--gpmm_model_path', type=str, default='data/BFM/BFM_model_front.mat')
parser.add_argument('--gpmm_delta_bs_path', type=str, default='data/BFM/mean_delta_blendshape.npy')
# data opts.
parser.add_argument('--dataset', type=str, default='full', choices=['full', 'voxceleb2'])
parser.add_argument('--voxceleb2_root', type=str, default=None)
parser.add_argument('--feafa_root', type=str, default=None)
parser.add_argument('--lp_300w_root', type=str, default=None)
parser.add_argument('--input_size', type=int, default=224, help='input data size')
parser.add_argument('--input_channel', type=int, default=3, help='input data channel')
parser.add_argument('--n_frames', type=int, default=4, help='the number of frames')
parser.add_argument('--batch_size', type=int, default=8, help='mini-batch size')
parser.add_argument('--num_workers', type=int, default=8, help='num_workers to load data.')
# for loading data fast, I save the file path in pickle file in first time, and use data file later
# when run at the second time ,set --use_data_files=True --save_pickle False
parser.add_argument('--use_data_files', type=str2bool, default='False', help='dataset')
parser.add_argument('--save_pickle', type=str2bool, default='True', help='dataset')
# model opts.
parser.add_argument('--conv_dim', type=int, default=32, help='number of conv filters in the first layer of G')
parser.add_argument('--network_type', default='ResNet50', choices=['ResNet50', 'mobilenet-v2'])

# Training opts.
parser.add_argument('--num_iters', type=int, default=300000, help='number of total iterations for training D')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
parser.add_argument('--log_step', type=int, default=5)
parser.add_argument('--eval_step', type=int, default=1000)
parser.add_argument('--model_save_step', type=int, default=10000)
# loss weight
parser.add_argument('--PHOTO_LOSS_W', type=float, default=1.9, help='Photometric Loss')
parser.add_argument('--IDENTITY_LOSS_W', type=float, default=0.2, help='Perception Loss')
parser.add_argument('--LM_LOSS_W', type=float, default=0.1, help='Landmark Loss')
parser.add_argument('--REG_W', type=float, default=0.0001, help='Regularization Loss')
parser.add_argument('--SP_LOSS_W', type=float, default=0.1, help='Regularization Loss')
parser.add_argument('--ID_CON_LOSS_W', type=float, default=1000, help='Identity-consistent Loss')
parser.add_argument('--CONFLICT_LOSS_W', type=float, default=10, help='Expression-exclusive Loss')


opts = parser.parse_args()
opts.log_dir = os.path.join(opts.result_root, opts.log_dir)
opts.checkpoint_dir = os.path.join(opts.result_root, opts.checkpoint_dir)
opts.sample_dir = os.path.join(opts.result_root, opts.sample_dir)
# Create directories if not exist.
create_dirs_if_not_exist([opts.log_dir, opts.checkpoint_dir, opts.sample_dir])




if __name__ == '__main__':

    # set random seed
    set_seed(opts.seed)

    # log opts.
    with open(os.path.join(opts.result_root, 'opts.yaml'), 'w') as f:
        f.write(yaml.dump(vars(opts)))

    # initialize trainer
    solver = TrainSolver(opts)

    # train
    solver.train()





