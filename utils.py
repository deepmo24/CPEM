import os
import cv2
import random
import numpy as np
import torch
from torch.backends import cudnn

def str2bool(v):
    return v.lower() in ('true')

def create_dirs_if_not_exist(dir_list):
    if isinstance(dir_list, list):
        for dir in dir_list:
            if not os.path.exists(dir):
                os.makedirs(dir)
    else:
        if not os.path.exists(dir_list):
            os.makedirs(dir_list)


def set_seed(seed):
    # For fast training.
    cudnn.benchmark = True

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() == 1:
        torch.cuda.manual_seed(seed)
    else:
        torch.cuda.manual_seed_all(seed)


def read_obj_file(path): 
    """
    % Reads a .obj mesh file and outputs the vertex and face list
    % assumes a 3D triangle mesh and ignores everything but:
    % v x y z and f i j k lines
    % Input:
    %  filename  string of obj file's path
    %
    % Output:
    %  V  number of vertices x 3 array of vertex positions
    %  F  number of faces x 3 array of face indices
    %
    % Referred this site : http://www.alecjacobson.com/weblog/?p=917
    """
    if path.endswith('.obj'):
        f = open(path, 'r')
        lines = f.readlines()
        vertices = []
        faces = []

        for line in lines:
            # read vertices
            if line.startswith('v') and not line.startswith('vt') and not line.startswith('vn'): # 'v %f %f %f'
                line_split = line.split()
                ver = line_split[1:4]
                ver = [float(v) for v in ver]
                # print(ver)
                vertices.append(ver)
            else:
                # read faces
                if line.startswith('f'):
                    line_split = line.split()
                    if '/' in line: # 'f %d//%d %d//%d %d//%d' or 'f %d/%d/%d %d/%d/%d %d/%d/%d'
                        tmp_faces = line_split[1:]
                        f = []
                        for tmp_face in tmp_faces:
                            f.append(int(tmp_face.split('/')[0]))
                        faces.append(f)
                    else: # 'f %d %d %d'
                        face = line_split[1:]
                        face = [int(fa) for fa in face]
                        faces.append(face)
        return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)

    else:
        print('Input file format is not correct!')
        return


def write_obj(obj_name, vertices, triangles, colors=None):
    ''' Save 3D face model.
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        triangles: shape = (ntri, 3)
        colors: shape = (nver, 3)
    '''
    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'

    # write obj
    with open(obj_name, 'w') as f:

        # write vertices
        for i in range(vertices.shape[0]):
            if colors is None:
                s = 'v {} {} {} \n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2])
            else:
                s = 'v {} {} {} {} {} {} \n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2],
                                                colors[i, 0], colors[i, 1], colors[i, 2])
            f.write(s)

        # write f
        for i in range(triangles.shape[0]):
            s = 'f {} {} {} \n'.format(triangles[i, 0], triangles[i, 1], triangles[i, 2])
            f.write(s)


def save_frames_from_video(video_path, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

        cap = cv2.VideoCapture(video_path)

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('total frames: {}'.format(n_frames))

        i = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                file_name = '%05d.jpg'%(i)
                cv2.imwrite(os.path.join(save_path, file_name), frame)

                if (i+1) % 100 == 0:
                    print('process %d images.' % (i+1))
                i += 1
            else:
                break


class FAN(object):
    def __init__(self):
        import face_alignment
        print('use 3d detector')
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)

    def detect(self, image):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        landmarks = self.model.get_landmarks(image)
        if landmarks is None:
            return None, None
        else:
            list_size_detected_face = []
            for i in range(len(landmarks)):
                l_hf = landmarks[i]
                min_bbox = l_hf.min(axis=0)
                max_bbox = l_hf.max(axis=0)
                bbox_size = np.max(max_bbox - min_bbox)
                list_size_detected_face.append(bbox_size)

            list_size_detected_face = np.array(list_size_detected_face)
            idx_max = np.argmax(list_size_detected_face)

            landmark = landmarks[idx_max]

            kpt = landmark.squeeze()
            left = np.min(kpt[:,0]); right = np.max(kpt[:,0])
            top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
            bbox = [left, top, right, bottom]
            return bbox, kpt[:,:2]
