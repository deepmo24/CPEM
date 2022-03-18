import argparse
import os
from utils import str2bool, set_seed, create_dirs_if_not_exist
from core.test_solver import TestSolver


parser = argparse.ArgumentParser()

# basic opts.
parser.add_argument('--seed', type=int, default=2408, help='random seed')
parser.add_argument('--use_gpu', type=str2bool, default='True', help='whether to use gpu')
parser.add_argument('--log_dir', type=str, default='logs/')
parser.add_argument('--checkpoint_dir', type=str, default='models/')
parser.add_argument('--sample_dir', type=str, default='samples/')
parser.add_argument('--result_root', type=str, default='results/')
parser.add_argument('--checkpoint_path', type=str, default=None, help='load pretrained model from checkpoint_path')
# 3dmm opts
parser.add_argument('--gpmm_model_path', type=str, default='data/BFM/BFM_model_front.mat')
parser.add_argument('--gpmm_delta_bs_path', type=str, default='data/BFM/mean_delta_blendshape.npy')
# data opts.
parser.add_argument('--input_size', type=int, default=224, help='input data size')
parser.add_argument('--input_channel', type=int, default=3, help='input data channel')
parser.add_argument('--n_frames', type=int, default=4, help='the number of frames')
parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size')
# model opts.
parser.add_argument('--conv_dim', type=int, default=32, help='number of conv filters in the first layer of G')
parser.add_argument('--network_type', default='ResNet50', choices=['ResNet50', 'mobilenet-v2'])

# Test opts.
parser.add_argument('--mode', type=str, default='demo', choices=['demo', 'retarget', 'render_shape'])
parser.add_argument('--test_iter', type=int, default=300000, help='test model from this step, **only use when checkpoint_path is None')
parser.add_argument('--image_path', type=str, default=None, help='image directory or image path')
parser.add_argument('--save_path', type=str, default=None, help='result save path for video')
parser.add_argument('--onnx', type=str2bool, default='False', help='use onnx to accelerate')
parser.add_argument('--detect_type', type=str, default='box', help='box or fan')
parser.add_argument('--save_mesh', type=str2bool, default='False', help='whether save mesh')
parser.add_argument('--source_coeff_path', type=str, default=None, help='coefficiet directory or path')
parser.add_argument('--target_image_path', type=str, default=None, help='retargeted image path')

opts = parser.parse_args()
opts.log_dir = os.path.join(opts.result_root, opts.log_dir)
opts.checkpoint_dir = os.path.join(opts.result_root, opts.checkpoint_dir)
opts.sample_dir = os.path.join(opts.result_root, opts.sample_dir)

if __name__ == '__main__':

    # set random seed
    set_seed(opts.seed)

    # initialize trainer
    solver = TestSolver(opts)

    if opts.detect_type == 'box':
        from FaceBoxes import FaceBoxes # face box detector
        if opts.onnx:
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            os.environ['OMP_NUM_THREADS'] = '4'

            from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
            face_detector = FaceBoxes_ONNX()
        else:
            face_detector = FaceBoxes()
    else:
        from utils import FAN # landmark detector
        face_detector = FAN()


    if opts.mode == 'demo':
        solver.infer_from_image_paths(opts.image_path, face_detector)
    elif opts.mode == 'retarget':
        solver.run_facial_motion_retargeting(opts.source_coeff_path, opts.target_image_path, face_detector)
    elif opts.mode == 'render_shape':
        solver.render_shape(opts.image_path, face_detector)

    print('done!')


