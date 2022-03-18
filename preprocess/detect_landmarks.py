'''
Detect 3D facial landmarks by face-alignmet detector.
'''

import face_alignment
import cv2
import argparse
import os
import numpy as np
from multiprocessing import Pool
from itertools import cycle
from tqdm import tqdm
import glob



def process_person_id(person_id, args):

    # initialize
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)


    video_ids = os.listdir(os.path.join(args.video_root, person_id))
    chunks_data = []
    for v_i, video_id in enumerate(video_ids):

        curr_video_path = os.path.join(args.video_root, person_id, video_id)
        curr_save_path = os.path.join(save_path, person_id, video_id)
        if not os.path.exists(curr_save_path):
            os.makedirs(curr_save_path)

        preds = fa.get_landmarks_from_directory(curr_video_path, show_progress_bar=False)

        count = 0
        for frame_path, lm_3d_list in preds.items():
            # No face detected(None),  No landmark detected([])
            if lm_3d_list is not None and len(lm_3d_list) != 0:
                # Only use the first face
                lm_3d = lm_3d_list[0]
                lm_2d = lm_3d[:,:2]

                base_output_name = os.path.basename(frame_path)[:-4] + '.txt'
                np.savetxt(os.path.join(curr_save_path, base_output_name), lm_2d)

                count += 1

        chunks_data += [
            {'person_id': person_id, 'video_id': video_id, 'success_count': count}]

    return chunks_data


def process_SingleProcess(args):
    person_ids = os.listdir(args.video_root)
    person_ids = [person_id for person_id in person_ids if os.path.isdir(os.path.join(args.video_root, person_id))]
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

        line = "{person_id},{video_id},{success_count}"
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
    device_ids = args.device_ids.split(",")
    with Pool(processes=args.workers) as pool:
        args_list = cycle([args])
        f = open(args.chunks_metadata, 'w')
        line = "{person_id},{video_id},{success_count}"
        print(line.replace('{', '').replace('}', ''), file=f)
        for chunks_data in tqdm(pool.imap_unordered(fn, zip(data_list, cycle(device_ids), args_list))):
            for data in chunks_data:
                print(line.format(**data), file=f)
                f.flush()
        f.close()

def run(params):
    person_id, device_id, args = params
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    return process_person_id(person_id, args)



def process_MultiProcess(args):
    person_ids = os.listdir(args.video_root)
    person_ids = [person_id for person_id in person_ids if os.path.isdir(os.path.join(args.video_root, person_id))]
    person_ids.sort()
    print('There are total {} identities.'.format(len(person_ids)))

    if args.start is not None and args.end is not None:
        person_ids = person_ids[args.start:args.end]
        print('Interval [{},{}], process {} identities'.format(args.start, args.end, len(person_ids)))
        args.chunks_metadata = os.path.join(save_path,
                                            'Pos_{}_{}_{}'.format(args.start, args.end, args.chunks_metadata))
    else:
        args.chunks_metadata = os.path.join(save_path, args.chunks_metadata)

    args.chunks_metadata = os.path.join(save_path, args.chunks_metadata)

    scheduler(person_ids, run, args)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_root', type=str, default='./data/voxceleb2/data')
    parser.add_argument('--save_path', type=str, default='./data/voxceleb2/landmarks')
    parser.add_argument("--chunks_metadata", default='landmark_metadata.csv', help='File with metadata')
    parser.add_argument('--start', type=int, default=None, help='start index')
    parser.add_argument('--end', type=int, default=None, help='end index')
    parser.add_argument('--multi_process', action='store_true', default=False)
    parser.add_argument("--workers", default=2, type=int, help='Number of parallel workers')
    parser.add_argument("--device_ids", default="0,1", help="Names of the devices comma separated.")
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
        process_MultiProcess(args)
    else:
        process_SingleProcess(args)

