import os
import cv2
import glob
import pickle
import random
import torch
import numpy as np


#----------------------- functions for preprocessig image

# From 3DDFA-v2
def crop_img(img, roi_box):
    h, w = img.shape[:2]

    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
    dh, dw = ey - sy, ex - sx
    if len(img.shape) == 3:
        res = np.zeros((dh, dw, 3), dtype=np.uint8)
    else:
        res = np.zeros((dh, dw), dtype=np.uint8)
    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
    return res

# From 3DDFA-v2
def similar_transform_2d(pts2d, roi_box, size):
    '''
    transform the original landmarks to the new-size image coordinate
    '''
    sx, sy, ex, ey = roi_box
    scale_x = (ex - sx) / size
    scale_y = (ey - sy) / size
    pts2d[:, 0] = pts2d[:, 0] - sx
    pts2d[:, 1] = pts2d[:, 1] - sy
    pts2d[:, 0] = pts2d[:, 0] / scale_x
    pts2d[:, 1] = pts2d[:, 1] / scale_y

    return np.array(pts2d, dtype=np.float32)

# From 3DDFA-v2
def similar_transform_crop_to_origin(pts2d, roi_box, size):
    '''
    transform the landmarks to the original image coordinate
    '''
    sx, sy, ex, ey = roi_box
    scale_x = (ex - sx) / size
    scale_y = (ey - sy) / size
    pts2d[:, 0] = pts2d[:, 0] * scale_x + sx
    pts2d[:, 1] = pts2d[:, 1] * scale_y + sy
    return pts2d


# From 3DDFA-v2
def similar_transform_3d(pts3d, roi_box, size):
    '''
    transform the original landmarks to the new-size image coordinate
    '''
    # already done in face_decoder.decode_face()
    # pts3d[1, :] = size - 1 - pts3d[1, :]

    sx, sy, ex, ey = roi_box
    scale_x = (ex - sx) / size
    scale_y = (ey - sy) / size
    pts3d[:, 0] = pts3d[:, 0] * scale_x + sx
    pts3d[:, 1] = pts3d[:, 1] * scale_y + sy
    # s = (scale_x + scale_y) / 2
    # pts3d[:, 2] *= s
    # pts3d[:, 2] -= pts3d[:, 2].min()
    return pts3d



def overlying_image_origin(roi_box, composed_img, rendered_img, rendered_mask):
    '''
    overlay the rendered image on the origin input image
    *** Drawback: there are obvious gaps at the edges of the rendered mask
    '''
    h, w = composed_img.shape[:2]
    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
    dh, dw = ey - sy, ex - sx
    resized_img = cv2.resize(rendered_img, (dw, dh))  # h,w,3
    resized_mask = cv2.resize(rendered_mask, (dw, dh))  # h,w,3

    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    composed_img[sy:ey, sx:ex] = resized_img[dsy:dey, dsx:dex] * resized_mask[dsy:dey, dsx:dex] + \
                                 composed_img[sy:ey, sx:ex] * (1 - resized_mask[dsy:dey, dsx:dex])

    return composed_img


def overlying_image_resize(roi_box, composed_img, rendered_img, rendered_mask):
    '''
    overlay the resized rendered image on the resized input image
    *** better than overlying image to origin image plane
    '''
    h, w = composed_img.shape[:2]
    sh, sw = rendered_img.shape[:2]

    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
    dh, dw = ey - sy, ex - sx

    scale_h = sh / dh
    scale_w = sw / dw

    size_h = int(round(h * scale_h))
    size_w = int(round(w * scale_w))

    composed_img = cv2.resize(composed_img, (size_w, size_h))  # h,w,3

    # scale
    sx, sy, ex, ey = sx*scale_w, sy*scale_h, ex*scale_w, ey*scale_h
    sx, sy, ex, ey = int(round(sx)), int(round(sy)), int(round(ex)), int(round(ey))
    dh, dw = sh, sw
    h, w = int(round(h*scale_h)), int(round(w*scale_w))

    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    composed_img[sy:ey, sx:ex] = rendered_img[dsy:dey, dsx:dex] * rendered_mask[dsy:dey, dsx:dex] + \
                                 composed_img[sy:ey, sx:ex] * (1 - rendered_mask[dsy:dey, dsx:dex])

    return composed_img


def parse_roi_box_from_landmark_box(bbox):
    """calc roi box from landmark"""
    left, top, right, bottom = bbox[:4]
    old_size = (right - left + bottom - top) / 2 * 1.1
    center_x = right - (right - left) / 2.0
    center_y = bottom - (bottom - top) / 2.0

    size = int(old_size * 1.25)
    # size = int(old_size * 1.5)

    roi_box = [0] * 4
    roi_box[0] = center_x - size / 2
    roi_box[1] = center_y - size / 2
    roi_box[2] = roi_box[0] + size
    roi_box[3] = roi_box[1] + size

    return roi_box


def parse_roi_box_from_bbox(bbox):
    """calc roi box from bounding box"""
    left, top, right, bottom = bbox[:4]
    old_size = (right - left + bottom - top) / 2
    center_x = right - (right - left) / 2.0
    center_y = bottom - (bottom - top) / 2.0 + old_size * 0.12
    size = int(old_size * 1.25)
    # size = int(old_size * 1.5)

    roi_box = [0] * 4
    roi_box[0] = center_x - size / 2
    roi_box[1] = center_y - size / 2
    roi_box[2] = roi_box[0] + size
    roi_box[3] = roi_box[1] + size

    return roi_box


def crop_and_resize_by_bbox(image, landmark, landmark2d, img_size):
    # get bbox from landmarks
    left = np.min(landmark[:, 0])
    right = np.max(landmark[:, 0])
    top = np.min(landmark[:, 1])
    bottom = np.max(landmark[:, 1])
    bbox = [left, top, right, bottom]

    roi_box = parse_roi_box_from_landmark_box(bbox)
    img_cropped = crop_img(image, roi_box)
    img_resized = cv2.resize(img_cropped, dsize=(img_size, img_size), interpolation=cv2.INTER_LINEAR)
    landmark_resized = similar_transform_2d(landmark.copy(), roi_box, img_size)
    landmark2d_resized = similar_transform_2d(landmark2d.copy(), roi_box, img_size)

    return img_resized, landmark_resized, landmark2d_resized


def Preprocess(image, target_size, bbox, detect_type, return_box=False):
    '''
    preprocess the image for the network input
    :param image: image of size [h,w,c] read from cv2
    :param target_size: target size to resize
    :param bbox: face bounding box
    :param detect_type: box or fan
    :return: image_tensor: [1, c, h, w]
    '''
    if detect_type == 'box':
        roi_box = parse_roi_box_from_bbox(bbox)
    else:
        roi_box = parse_roi_box_from_landmark_box(bbox)

    img_cropped = crop_img(image, roi_box)
    img_resized = cv2.resize(img_cropped, dsize=(target_size, target_size), interpolation=cv2.INTER_LINEAR)

    # BGRtoRGB
    image = img_resized[:, :, ::-1].copy()
    # [0,255] -> [0,1]
    image = image / 255.
    # [h,w,c] -> [c,h,w]
    image = image.transpose(2, 0, 1)
    image_tensor = torch.from_numpy(image).float()
    # [c,h,w] -> [1,c,h,w]
    image_tensor = image_tensor[None,:]

    if not return_box:
        return image_tensor
    else:
        return image_tensor, roi_box


#----------------------- functions for preparing the datasets

def read_face_flag_file(data_root, flag_path):
    f = open(flag_path)
    flag_data = f.readlines()[1:]

    face_flag_dict = {}
    for line in flag_data:
        person_id, video_id, frame_name, flag = line.strip().split(',')
        flag = int(flag)
        frame_path = os.path.join(os.path.join(data_root, person_id, video_id, frame_name))
        face_flag_dict[frame_path] = flag

    return face_flag_dict


def combine_2d_3d_landmarks(front_flag, new_lm, new_lm2d):
    inner_idx = list(range(17, 68))
    outer_idx = list(range(17))

    composed_lm = np.zeros_like(new_lm)
    if front_flag == 1:
        composed_lm[inner_idx] = new_lm2d[inner_idx]
        composed_lm[outer_idx] = new_lm[outer_idx]
    else:
        composed_lm = new_lm

    return composed_lm

# voxceleb2
def make_dataset_voxceleb2(dataset_root, num_frames, save_pickle=False):
    '''
    data_root/person_id/video_id/{xxx.jpg}
    mask_root/person_id/video_id/{xxx.jpg}
    landmark_root/person_id/video_id/{xxx.txt}
    landmark2d_root/person_id/video_id/{xxx.txt}
    front_face_flag.csv
    '''
    if dataset_root is None or not os.path.exists(dataset_root):
        return [], [], [], [], {}

    data_root = os.path.join(dataset_root, 'data')
    mask_root = os.path.join(dataset_root, 'face_mask')
    landmark_root = os.path.join(dataset_root, 'landmarks')
    landmark2d_root = os.path.join(dataset_root, 'landmarks2d')

    person_ids = os.listdir(data_root)
    person_ids = [person_id for person_id in person_ids if os.path.isdir(os.path.join(data_root, person_id))]
    person_ids.sort()

    data_list = []
    mask_list = []
    lm_list = []
    lm2d_list = []
    for person_id in person_ids:
        video_ids = os.listdir(os.path.join(data_root, person_id))
        video_ids.sort()
        for video_id in video_ids:
            video_path = os.path.join(data_root, person_id, video_id)
            mask_video_path = os.path.join(mask_root, person_id, video_id)
            lm_video_path = os.path.join(landmark_root, person_id, video_id)
            lm2d_video_path = os.path.join(landmark2d_root, person_id, video_id)

            lm2d_paths = glob.glob(os.path.join(lm2d_video_path, '*.txt'))
            # shuffle the inputs of one video clip (different video clips belong to one person from different videos)
            random.shuffle(lm2d_paths)

            frame_paths = [os.path.join(video_path, os.path.basename(item)[:-4] + '.jpg') for item in lm2d_paths]
            mask_paths = [os.path.join(mask_video_path, os.path.basename(item)[:-4] + '.jpg') for item in lm2d_paths]
            lm_paths = [os.path.join(lm_video_path, os.path.basename(item)) for item in lm2d_paths]

            # construct samples
            frame_length = len(frame_paths)
            for i in range(0, frame_length, num_frames):
                if i + num_frames <= frame_length:
                    data_list.append(frame_paths[i:(i + num_frames)])
                    mask_list.append(mask_paths[i:(i + num_frames)])
                    lm_list.append(lm_paths[i:(i + num_frames)])
                    lm2d_list.append(lm2d_paths[i:(i + num_frames)])

    # read front face flag
    flag_path = os.path.join(dataset_root, 'front_face_flag.csv')
    face_flag_dict = read_face_flag_file(data_root, flag_path)

    if save_pickle:
        dataset_dict = {}
        dataset_dict['data'] = data_list
        dataset_dict['mask'] = mask_list
        dataset_dict['lm2d'] = lm2d_list
        dataset_dict['lm'] = lm_list

        save_path = os.path.join(dataset_root, 'dataset_n{}.pkl'.format(num_frames))
        f = open(save_path, 'wb')
        pickle.dump(dataset_dict, f)
        f.close()

        print('save data in {}.'.format(save_path))

    return data_list, mask_list, lm_list, lm2d_list, face_flag_dict


# feafa and 300w-lp
def make_dataset_video(dataset_root, num_frames, save_pickle=False):
    '''
    data_root/person_id/video_id/{xxx.jpg}
    mask_root/person_id/video_id/{xxx.jpg}
    landmark_root/person_id/video_id/{xxx.txt}
    landmark2d_root/person_id/video_id/{xxx.txt}
    front_face_flag.csv
    '''

    if dataset_root is None or not os.path.exists(dataset_root):
        return [], [], [], [], {}

    data_root = os.path.join(dataset_root, 'data')
    mask_root = os.path.join(dataset_root, 'face_mask')
    landmark_root = os.path.join(dataset_root, 'landmarks')
    landmark2d_root = os.path.join(dataset_root, 'landmarks2d')

    person_ids = os.listdir(data_root)
    person_ids = [person_id for person_id in person_ids if os.path.isdir(os.path.join(data_root, person_id))]
    person_ids.sort()


    data_list = []
    mask_list = []
    lm_list = []
    lm2d_list = []
    for person_id in person_ids:
        video_ids = os.listdir(os.path.join(data_root, person_id))
        video_ids.sort()

        lm2d_paths_list = []
        frame_paths_list = []
        mask_paths_list = []
        lm_paths_list = []
        for video_id in video_ids:
            video_path = os.path.join(data_root, person_id, video_id)
            mask_video_path = os.path.join(mask_root, person_id, video_id)
            lm_video_path = os.path.join(landmark_root, person_id, video_id)
            lm2d_video_path = os.path.join(landmark2d_root, person_id, video_id)

            lm2d_paths = glob.glob(os.path.join(lm2d_video_path, '*.txt'))
            frame_paths = [os.path.join(video_path, os.path.basename(item)[:-4] + '.jpg') for item in lm2d_paths]
            mask_paths = [os.path.join(mask_video_path, os.path.basename(item)[:-4] + '.jpg') for item in lm2d_paths]
            lm_paths = [os.path.join(lm_video_path, os.path.basename(item)) for item in lm2d_paths]

            lm2d_paths_list.extend(lm2d_paths)
            frame_paths_list.extend(frame_paths)
            mask_paths_list.extend(mask_paths)
            lm_paths_list.extend(lm_paths)

        # shuffle the inputs of one person (all the video clips belong to one person from one video)
        total_data_list = list(zip(*[frame_paths_list, mask_paths_list, lm_paths_list, lm2d_paths_list]))
        random.shuffle(total_data_list)
        shuffle_data_list = list(zip(*total_data_list))

        frame_paths_list = list(shuffle_data_list[0])
        mask_paths_list = list(shuffle_data_list[1])
        lm_paths_list = list(shuffle_data_list[2])
        lm2d_paths_list = list(shuffle_data_list[3])

        # construct samples
        frame_length = len(frame_paths_list)
        for i in range(0, frame_length, num_frames):
            if i+num_frames <= frame_length:
                data_list.append(frame_paths_list[i:(i+num_frames)])
                mask_list.append(mask_paths_list[i:(i + num_frames)])
                lm_list.append(lm_paths_list[i:(i + num_frames)])
                lm2d_list.append(lm2d_paths_list[i:(i + num_frames)])

    # read front face flag
    flag_path = os.path.join(dataset_root, 'front_face_flag.csv')
    face_flag_dict = read_face_flag_file(data_root, flag_path)

    if save_pickle:
        dataset_dict = {}
        dataset_dict['data'] = data_list
        dataset_dict['mask'] = mask_list
        dataset_dict['lm2d'] = lm2d_list
        dataset_dict['lm'] = lm_list

        save_path = os.path.join(dataset_root, 'dataset_n{}.pkl'.format(num_frames))
        f = open(save_path, 'wb')
        pickle.dump(dataset_dict, f)
        f.close()

        print('save data in {}.'.format(save_path))

    return data_list, mask_list, lm_list, lm2d_list, face_flag_dict


def make_dataset_from_files(dataset_root, num_frames):

    dataset_path = os.path.join(dataset_root, 'dataset_n{}.pkl'.format(num_frames))

    if os.path.exists(dataset_path):
        f = open(dataset_path, 'rb')

        dataset_dict = pickle.load(f)

        data_list = dataset_dict['data']
        mask_list = dataset_dict['mask']
        lm_list = dataset_dict['lm']
        lm2d_list = dataset_dict['lm2d']

        # read front face flag
        flag_path = os.path.join(dataset_root, 'front_face_flag.csv')
        data_root = os.path.join(dataset_root, 'data')
        face_flag_dict = read_face_flag_file(data_root, flag_path)

        return data_list, mask_list, lm_list, lm2d_list, face_flag_dict

    else:
        assert False, 'files not exist!'