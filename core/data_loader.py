import torch
from torch.utils import data
import cv2
import numpy as np
from core.data_utils import *


# -----------------------------------------------------------------------------------------------------------------
class Voxceleb2Dataset(data.Dataset):

    def __init__(self, dataset_root, n_frames, img_size=224, shuffle=True, use_data_files=False, save_pickle=True):

        if use_data_files:
            print('load data from data files.')
            data_list, mask_list, lm_list, lm2d_list, face_flag_dict = make_dataset_from_files(dataset_root, n_frames)
        else:
            print('load data from directories.')
            data_list, mask_list, lm_list, lm2d_list, face_flag_dict = make_dataset_voxceleb2(dataset_root, n_frames, save_pickle=save_pickle)

        self.data_list = data_list
        self.face_mask_list = mask_list
        self.lm_list = lm_list
        self.lm2d_list = lm2d_list
        self.face_flag_dict = face_flag_dict
        self.n_frames = n_frames
        self.img_size = img_size
        self.shuffle = shuffle

    def __getitem__(self, index):
        frame_paths = self.data_list[index]
        face_mask_paths = self.face_mask_list[index]
        lm_paths = self.lm_list[index]
        lm2d_paths = self.lm2d_list[index]

        frames = []
        landmarks = []
        face_masks = []
        front_flags = []
        for i in range(len(frame_paths)):
            frame_path = frame_paths[i]
            face_mask_path = face_mask_paths[i]
            lm_path = lm_paths[i]
            lm2d_path = lm2d_paths[i]

            # read files
            image = cv2.imread(frame_path)
            face_mask = cv2.imread(face_mask_path, cv2.IMREAD_GRAYSCALE)
            landmark = np.loadtxt(lm_path)  # 68x2
            landmark2d = np.loadtxt(lm2d_path)  # 68x2

            # crop
            new_img, new_lm, new_lm2d = crop_and_resize_by_bbox(image, landmark, landmark2d, self.img_size)
            # BGRtoRGB
            image = new_img[:, :, ::-1].copy()
            # [0,255] -> [0,1]
            image = image / 255.
            # [h,w,c] -> [c,h,w]
            image = image.transpose(2, 0, 1)
            image_tensor = torch.from_numpy(image).float()
            frames.append(image_tensor[None, :])

            # face mask w.r.t. the cropped image
            # resize
            face_mask = cv2.resize(face_mask, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            # [0,255] -> [0,1]
            face_mask = face_mask / 255.
            face_mask_tensor = torch.from_numpy(face_mask)
            face_masks.append(face_mask_tensor[None, :])

            # combine 2d and 3d landmarks according to front face flag
            front_flag = self.face_flag_dict[frame_path[:-4]]
            front_flags.append(front_flag)
            composed_lm = combine_2d_3d_landmarks(front_flag, new_lm, new_lm2d)

            lm_tensor = torch.tensor(composed_lm)
            landmarks.append(lm_tensor[None, :])

        frames_tensor = torch.cat(frames)  # [n_frames, c, h, w]
        lms_tensor = torch.cat(landmarks)  # [n_frames, 68, 2]
        face_masks_tensor = torch.cat(face_masks)  # [n_frames, h, w]
        front_flag_tensor = torch.tensor(front_flags).view(-1, 1)  # [n_frames, 1]

        # [N, c, h, w], [N, 68, 2], [N, h, w], [N,1]
        return (frames_tensor, lms_tensor, face_masks_tensor, front_flag_tensor)

    def __len__(self):
        return len(self.data_list)


def get_voxceleb2_loader(data_root, n_frames, img_size, batch_size=4, is_train=True, num_workers=4, use_data_files=False, save_pickle=True):
    """
    Build and return a data loader.
    """

    dataset = Voxceleb2Dataset(data_root, n_frames, img_size, shuffle=is_train, use_data_files=use_data_files, save_pickle=save_pickle)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  drop_last=True,
                                  shuffle=is_train,
                                  num_workers=num_workers,
                                  pin_memory=True)
    return data_loader



# -----------------------------------------------------------------------------------------------------------------
class TrainDataset(data.Dataset):

    def __init__(self, voxceleb2_root, feafa_root, lp_300w_root, n_frames, img_size=224, shuffle=True, use_data_files=False, save_pickle=True):

        if use_data_files:
            print('load data from data files.')
            data_list_vox, mask_list_vox, lm_list_vox, lm2d_list_vox, face_flag_dict_vox = make_dataset_from_files(voxceleb2_root, n_frames)
            data_list_feafa, mask_list_feafa, lm_list_feafa, lm2d_list_feafa, face_flag_dict_feafa = make_dataset_from_files(feafa_root, n_frames)
            data_list_300w_lp, mask_list_300w_lp, lm_list_300w_lp, lm2d_list_300w_lp, face_flag_dict_300w_lp = make_dataset_from_files(lp_300w_root, n_frames)
        else:
            print('load data from directories.')
            data_list_vox, mask_list_vox, lm_list_vox, lm2d_list_vox, face_flag_dict_vox = make_dataset_voxceleb2(voxceleb2_root, n_frames, save_pickle=save_pickle)
            data_list_feafa, mask_list_feafa, lm_list_feafa, lm2d_list_feafa, face_flag_dict_feafa = make_dataset_video(feafa_root, n_frames, save_pickle=save_pickle)
            data_list_300w_lp, mask_list_300w_lp, lm_list_300w_lp, lm2d_list_300w_lp, face_flag_dict_300w_lp = make_dataset_video(lp_300w_root, n_frames, save_pickle=save_pickle)
            if save_pickle:
                print('[*] you can Load data from cache files next time by setting --use_data_files=True')

        data_list = data_list_vox + data_list_feafa + data_list_300w_lp
        mask_list = mask_list_vox + mask_list_feafa + mask_list_300w_lp
        lm_list = lm_list_vox + lm_list_feafa + lm_list_300w_lp
        lm2d_list = lm2d_list_vox + lm2d_list_feafa + lm2d_list_300w_lp

        face_flag_dict = face_flag_dict_vox
        for key in face_flag_dict_feafa.keys():
            face_flag_dict[key] = face_flag_dict_feafa[key]
        for key in face_flag_dict_300w_lp.keys():
            face_flag_dict[key] = face_flag_dict_300w_lp[key]

        self.data_list = data_list
        self.face_mask_list = mask_list
        self.lm_list = lm_list
        self.lm2d_list = lm2d_list
        self.face_flag_dict = face_flag_dict
        self.n_frames = n_frames
        self.img_size = img_size
        self.shuffle = shuffle

    def __getitem__(self, index):
        frame_paths = self.data_list[index]
        face_mask_paths = self.face_mask_list[index]
        lm_paths = self.lm_list[index]
        lm2d_paths = self.lm2d_list[index]

        frames = []
        landmarks = []
        face_masks = []
        front_flags = []
        for i in range(len(frame_paths)):
            frame_path = frame_paths[i]
            face_mask_path = face_mask_paths[i]
            lm_path = lm_paths[i]
            lm2d_path = lm2d_paths[i]

            image = cv2.imread(frame_path)
            face_mask = cv2.imread(face_mask_path, cv2.IMREAD_GRAYSCALE)
            landmark = np.loadtxt(lm_path)  # 68x2
            landmark2d = np.loadtxt(lm2d_path)  # 68x2

            # crop
            new_img, new_lm, new_lm2d = crop_and_resize_by_bbox(image, landmark, landmark2d, self.img_size)
            # BGRtoRGB
            image = new_img[:, :, ::-1].copy()
            # [0,255] -> [0,1]
            image = image / 255.
            # [h,w,c] -> [c,h,w]
            image = image.transpose(2, 0, 1)
            image_tensor = torch.from_numpy(image).float()
            frames.append(image_tensor[None, :])

            # face mask w.r.t. the cropped image
            # resize
            face_mask = cv2.resize(face_mask, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            # [0,255] -> [0,1]
            face_mask = face_mask / 255.
            face_mask_tensor = torch.from_numpy(face_mask)
            face_masks.append(face_mask_tensor[None, :])

            # combine 2d and 3d landmarks according to front face flag
            front_flag = self.face_flag_dict[frame_path[:-4]]
            front_flags.append(front_flag)
            composed_lm = combine_2d_3d_landmarks(front_flag, new_lm, new_lm2d)

            lm_tensor = torch.tensor(composed_lm)
            landmarks.append(lm_tensor[None, :])

        frames_tensor = torch.cat(frames)  # [n_frames, c, h, w]
        lms_tensor = torch.cat(landmarks)  # [n_frames, 68, 2]
        face_masks_tensor = torch.cat(face_masks)  # [n_frames, h, w]
        front_flag_tensor = torch.tensor(front_flags).view(-1, 1)  # [n_frames, 1]

        # [N, c, h, w], [N, 68, 2], [N, h, w], [N,1]
        return (frames_tensor, lms_tensor, face_masks_tensor, front_flag_tensor)

    def __len__(self):
        return len(self.data_list)


def get_trainset_loader(voxceleb2_root, feafa_root, lp_300w_root, n_frames, img_size, batch_size=4, is_train=True, num_workers=4, use_data_files=False, save_pickle=True):
    """
    Build and return a data loader.
    """

    dataset = TrainDataset(voxceleb2_root, feafa_root, lp_300w_root, n_frames, img_size, shuffle=is_train, use_data_files=use_data_files, save_pickle=save_pickle)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  drop_last=True,
                                  shuffle=is_train,
                                  num_workers=num_workers,
                                  pin_memory=True)
    return data_loader




