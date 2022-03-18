'''
Construct 300W_LP training dataset.
'''

import os
import shutil
import argparse
from scipy.io import loadmat
import numpy as np


def make_300W_LP_dataset(dataset_root, save_root):
    # --------------------------------------------------
    # Read dataset
    id_img_dict = {}
    image_path = []
    sub_datasets_name = ['AFW', 'HELEN', 'IBUG', 'LFPW']
    for subset_name in sub_datasets_name:
        subset_image_path = os.listdir(os.path.join(dataset_root, subset_name))
        image_path.extend(subset_image_path)

    print("total number of image in 300W_LP dataset: {}".format(len(image_path)))  # 122450

    for path in image_path:
        if not path.endswith('.jpg'):
            continue
        else:
            image = path.strip()
            image_split = os.path.splitext(path)[0].split("_")

            if not len(image_split) > 2:
                continue
            # assert len(image_split) > 2

            id = image_split[0]
            for i in range(1, len(image_split) - 1):
                id = id + "_" + image_split[i]

            if id not in id_img_dict:
                id_img_dict[id] = [image]
            else:
                id_img_dict[id].append(image)

    print("number of identity in 300W_LP: {}".format(len(id_img_dict.keys())))


    # Read landmarks from dataset
    landmark_root = os.path.join(dataset_root, 'landmarks')
    landmark_paths = []
    for subset_name in sub_datasets_name:
        subset_landmark_path = os.listdir(os.path.join(landmark_root, subset_name))
        landmark_paths.extend(subset_landmark_path)


    # --------------------------------------------------
    # Construct training dataset
    data_root = os.path.join(save_root, 'data')
    lm_root = os.path.join(save_root, 'landmarks')
    lm2d_root = os.path.join(save_root, 'landmarks2d')

    if not os.path.exists(data_root):
        os.makedirs(data_root)
    if not os.path.exists(lm_root):
        os.makedirs(lm_root)
    if not os.path.exists(lm2d_root):
        os.makedirs(lm2d_root)


    num_person = len(id_img_dict.keys())
    for index, (id, frame_names) in enumerate(id_img_dict.items()):
        print('process person [{}, {}]'.format(index+1, num_person))

        subset_dir = id.split("_")[0]
        # sort
        frame_names.sort()
        frame_paths = [os.path.join(dataset_root, subset_dir, name) for name in frame_names]
        lm_paths = [os.path.join(landmark_root, subset_dir, os.path.basename(item)[:-4] + '_pts.mat') for item in frame_paths]

        #### process images
        curr_data_path = os.path.join(data_root, id, '001')
        if not os.path.exists(curr_data_path):
            os.makedirs(curr_data_path)

        for frame_path in frame_paths:
            target_save_path = os.path.join(curr_data_path, os.path.basename(frame_path))
            shutil.copy(frame_path, target_save_path)

        #### process landmarks
        curr_lm_path = os.path.join(lm_root, id, '001')
        if not os.path.exists(curr_lm_path):
            os.makedirs(curr_lm_path)

        curr_lm2d_path = os.path.join(lm2d_root, id, '001')
        if not os.path.exists(curr_lm2d_path):
            os.makedirs(curr_lm2d_path)

        for lm_path in lm_paths:
            lm_data = loadmat(lm_path)

            lm3d = lm_data['pts_3d']
            lm2d = lm_data['pts_2d']

            base_name = os.path.basename(lm_path)[:-8] + '.txt'
            np.savetxt(os.path.join(curr_lm_path, base_name), lm3d)
            np.savetxt(os.path.join(curr_lm2d_path, base_name), lm2d)



def str2bool(v):
    return v.lower() in ('true')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='./data/300W_LP', help='raw dataset root')
    parser.add_argument('--save_root', default='./300W_LP_Train', help='constructed dataset root')
    args = parser.parse_args()

    make_300W_LP_dataset(args.dataset_root, args.save_root)