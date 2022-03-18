'''
Estimate yaw angles according to facial landmarks
'''

import numpy as np
import os
import glob
import argparse
from multiprocessing import Pool
from itertools import cycle
from tqdm import tqdm
import pickle as pkl
# from preprocess.utils import estimate_yaw_angle
from utils import estimate_yaw_angle


# read standard face 3D landmarks
std_lm_3d = np.loadtxt('3d_points.txt')
std_lm_3d = std_lm_3d.reshape([3, 68]).T
# normal to [0,1]
min_v = std_lm_3d.min()
max_v = std_lm_3d.max()
std_lm_3d[:, 0] = (std_lm_3d[:, 0] - min_v) / max_v
std_lm_3d[:, 1] = (std_lm_3d[:, 1] - min_v) / max_v
std_lm_3d[:, 2] = (std_lm_3d[:, 2] - min_v) / max_v
# flip the y-axis
std_lm_3d[:, 1] = 1 - std_lm_3d[:, 1]


def process_person_id(person_id, args):
    # different video clips of a person
    video_ids = os.listdir(os.path.join(args.landmark_root, person_id))

    chunks_data = []
    for v_i, video_id in enumerate(video_ids):
        video_path = os.path.join(args.landmark_root, person_id, video_id)
        lm_paths = glob.glob(os.path.join(video_path, '*.txt'))

        for c_i in range(len(lm_paths)):
            # load 2d landmark
            lm_path = lm_paths[c_i]
            frame_name = os.path.basename(lm_path)[:-4]
            landmark = np.loadtxt(lm_path)  # 68x2

            #### estimate yaw angle and determine whether the face is front or not
            yaw_angle = estimate_yaw_angle(std_lm_3d, landmark)
            
            yaw_angle = abs(yaw_angle)
            if yaw_angle <= 15:
                front_flag = 1
            else:
                front_flag = 0

            chunks_data += [
                {'person_id': person_id, 'video_id': video_id, 'frame_name': frame_name, 'front_face_flag': front_flag}]

    return chunks_data


def process_SingleProcess(args):
    person_ids = os.listdir(args.landmark_root)
    person_ids = [person_id for person_id in person_ids if os.path.isdir(os.path.join(args.landmark_root, person_id))]
    person_ids.sort()
    print('There are total {} identities.'.format(len(person_ids)))

    if args.start is not None and args.end is not None:
        person_ids = person_ids[args.start:args.end]
        print('Interval [{},{}], process {} identities'.format(args.start, args.end, len(person_ids)))

        args.chunks_metadata = os.path.join(save_path,
                                            'Pos_{}_{}_{}'.format(args.start, args.end, args.chunks_metadata))
    else:
        args.chunks_metadata = os.path.join(save_path, args.chunks_metadata)


    with open(args.chunks_metadata, 'w') as f:

        line = "{person_id},{video_id},{frame_name},{front_face_flag}"
        print(line.replace('{', '').replace('}', ''), file=f)

        length = len(person_ids)
        for i, person_id in enumerate(person_ids):
            print('processing identity [{},{}]'.format(i + 1, length))

            chunks_data = process_person_id(person_id, args)

            for data in chunks_data:
                print(line.format(**data), file=f)
                f.flush()


### parallel processing
def scheduler(data_list, fn, args):

    with Pool(processes=args.workers) as pool:
        args_list = cycle([args])
        f = open(args.chunks_metadata, 'w')
        line = "{person_id},{video_id},{frame_name},{front_face_flag}"
        print(line.replace('{', '').replace('}', ''), file=f)
        for chunks_data in tqdm(pool.imap_unordered(fn, zip(data_list, args_list))):
            for data in chunks_data:
                print(line.format(**data), file=f)
                f.flush()
        f.close()

def run(params):
    person_id, args = params
    return process_person_id(person_id, args)


def process_MultiProcess(args):
    person_ids = os.listdir(args.landmark_root)
    person_ids = [person_id for person_id in person_ids if os.path.isdir(os.path.join(args.landmark_root, person_id))]
    person_ids.sort()
    print('There are total {} identities.'.format(len(person_ids)))

    args.chunks_metadata = os.path.join(save_path, args.chunks_metadata)

    scheduler(person_ids, run, args)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--landmark_root', type=str, default='./data/voxceleb2/landmarks')
    parser.add_argument('--save_path', type=str, default='./data/voxceleb2')
    parser.add_argument("--workers", default=16, type=int, help='Number of parallel workers')
    parser.add_argument("--chunks_metadata", default='front_face_flag.csv', help='File with metadata')
    parser.add_argument('--start', type=int, default=None, help='start index')
    parser.add_argument('--end', type=int, default=None, help='end index')
    parser.add_argument('--multi_process', action='store_true', default=False)
    args = parser.parse_args()


    save_path = args.save_path
    if args.start is not None and args.end is not None:
        save_path = save_path + '_s{}_e{}'.format(args.start, args.end)
    else:
        save_path = save_path
    args.save_path = save_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.multi_process:
        pass
        process_MultiProcess(args)
    else:
        process_SingleProcess(args)
