import os
import torch
import glob
import cv2
from skimage import draw
from scipy.io import savemat, loadmat
import numpy as np

from core.data_utils import Preprocess, similar_transform_crop_to_origin, similar_transform_3d, overlying_image_resize
from model.resnet import ResNet50_3DMM
from model.mobilenetv2 import MobilenetV2_3DMM
from core.face_decoder import FaceDecoder

from utils import write_obj


class TestSolver(object):

    def __init__(self, opts):
        self.opts = opts
        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # face decoder
        self.FaceDecoder = FaceDecoder(opts.gpmm_model_path, opts.gpmm_delta_bs_path, batch_size=self.opts.batch_size, device=self.device, img_size=opts.input_size)

        # build model
        self.init_model()

        # load trained model
        self.restore_model(self.opts.test_iter)


    def init_model(self):
        print('Network Type: ', self.opts.network_type)
        if self.opts.network_type == 'ResNet50':
            self.network = ResNet50_3DMM(n_id=self.FaceDecoder.facemodel.n_id_para,
                                         n_bs=self.FaceDecoder.facemodel.n_bs_para,
                                         n_tex=self.FaceDecoder.facemodel.n_tex_para,
                                         n_rot=self.FaceDecoder.facemodel.n_rot_para,
                                         n_light=self.FaceDecoder.facemodel.n_light_para,
                                         n_tran=self.FaceDecoder.facemodel.n_tran_para)
        elif self.opts.network_type == 'mobilenet-v2':
            self.network = MobilenetV2_3DMM(n_id=self.FaceDecoder.facemodel.n_id_para,
                                            n_bs=self.FaceDecoder.facemodel.n_bs_para,
                                            n_tex=self.FaceDecoder.facemodel.n_tex_para,
                                            n_rot=self.FaceDecoder.facemodel.n_rot_para,
                                            n_light=self.FaceDecoder.facemodel.n_light_para,
                                            n_tran=self.FaceDecoder.facemodel.n_tran_para)

        self.network.to(self.device)


    def restore_model(self, resume_iters):
        """
        Restore the trained generators and discriminator.
        """
        if self.opts.checkpoint_path is not None and os.path.exists(self.opts.checkpoint_path):
            print('Loading the trained models from {}...'.format(self.opts.checkpoint_path))
            self.network.load_state_dict(torch.load(self.opts.checkpoint_path, map_location=lambda storage, loc: storage))
        else:
            print('Loading the trained models from step {}...'.format(resume_iters))
            net_path = os.path.join(self.opts.checkpoint_dir, '{}-network.ckpt'.format(resume_iters))
            if os.path.exists(net_path):
                self.network.load_state_dict(torch.load(net_path, map_location=lambda storage, loc: storage))
            else:
                assert False, 'Checkpoint flie not exist!'


    def infer_from_image_paths(self, video_path, face_detector, save_path=None):

        if os.path.isdir(video_path):
            frame_paths = glob.glob(os.path.join(video_path, '*.jpg')) + glob.glob(os.path.join(video_path, '*.png'))
            frame_paths.sort()
        elif isinstance(video_path, list):
            frame_paths = video_path
        else:
            frame_paths = [video_path]


        if save_path is None:
            save_path = self.opts.save_path + '_model{}'.format(self.opts.test_iter)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        
        if len(frame_paths) == 1:
            mesh_save_path = overlay_origin_save_path = overlay_save_path = proj_save_path = coeff_save_path = save_path
        else:
            mesh_save_path = os.path.join(save_path, 'mesh')
            overlay_origin_save_path = os.path.join(save_path, 'overlay')
            overlay_save_path = os.path.join(save_path, 'overlay_crop')
            proj_save_path = os.path.join(save_path, 'proj')
            coeff_save_path = os.path.join(save_path, 'coeffs')

            if not os.path.exists(mesh_save_path):
                os.makedirs(mesh_save_path)
            if not os.path.exists(overlay_save_path):
                os.makedirs(overlay_save_path)
            if not os.path.exists(overlay_origin_save_path):
                os.makedirs(overlay_origin_save_path)
            if not os.path.exists(proj_save_path):
                os.makedirs(proj_save_path)
            if not os.path.exists(coeff_save_path):
                os.makedirs(coeff_save_path)

        coeff_bs_list = []
        coeff_angle_list = []

        self.network.eval()
        with torch.no_grad():
            length = len(frame_paths)
            for idx, frame_path in enumerate(frame_paths):
                print('process frames [{}/{}].'.format(idx+1, length))

                frame = cv2.imread(frame_path)
                # detect bbox
                if self.opts.detect_type == 'box':
                    boxes = face_detector(frame)
                    if len(boxes) == 0:
                        print('No face detected of {}'.format(frame_path))
                        continue
                    bbox = boxes[0]
                else:
                    bbox, _ = face_detector.detect(frame)
                    if bbox is None:
                        print('No face detected of {}'.format(frame_path))
                        continue
                # preprocess data
                # [bs, c, h, w]
                input_data, roi_box = Preprocess(frame, self.opts.input_size, bbox, self.opts.detect_type, return_box=True)
                input_data = input_data.to(self.device)
                # network inference
                pred_coeffs = self.network(input_data)  # [bs, 159]

                # rendered_img: [1, h, w, 4]
                rendered_img, pred_lm2d, coeffs, mesh = self.FaceDecoder.decode_face(pred_coeffs, return_coeffs=True)
                render_mask = rendered_img[:, :, :, 3].detach()
                rendered_img = rendered_img[:, :, :, :3]

                # save retargeting parameters
                coeff_bs = coeffs['coeff_bs']
                coeff_angle = coeffs['coeff_angle']
                coeff_bs_list.append(coeff_bs)
                coeff_angle_list.append(coeff_angle)

                # overlay processed image
                eval_intput_data = input_data * 255 # [0,1] -> [0,255]
                eval_intput_data = eval_intput_data.permute(0, 2, 3, 1)  # [bs, c, h, w] -> [bs, h, w, c]

                eval_mask = (render_mask > 0).type(torch.uint8)
                eval_mask = eval_mask.view(eval_mask.size(0), eval_mask.size(1), eval_mask.size(2), 1)
                eval_mask = eval_mask.repeat(1, 1, 1, 3)
                eval_overlay_images = rendered_img * eval_mask + eval_intput_data * (1 - eval_mask)
                eval_overlay_images = eval_overlay_images.cpu().numpy()
                eval_overlay_image = np.squeeze(eval_overlay_images)[:, :, ::-1] # [h,w,c] BGR

                # overlay original image
                rendered_img = rendered_img.squeeze().cpu().numpy()
                rendered_mask = eval_mask.squeeze().cpu().numpy()
                rendered_img = rendered_img[:, :, ::-1]  # [h,w,c] BGR
                raw_img = frame.copy()
                composed_img = overlying_image_resize(roi_box, raw_img, rendered_img, rendered_mask)


                eval_proj_image = frame.copy()
                # map landmarks to original image
                pred_lm2d_ori = similar_transform_crop_to_origin(pred_lm2d.squeeze(), roi_box, self.opts.input_size)
                pred_lm2d_ori = pred_lm2d_ori.cpu().numpy()

                # draw landmarks
                for k in range(68):
                    rr1, cc1 = draw.disk((pred_lm2d_ori[k][1], pred_lm2d_ori[k][0]), 2)
                    draw.set_color(eval_proj_image, [rr1, cc1], [0, 0, 255])

                base_input_name = os.path.basename(frame_path)[:-4]
                # save parameters
                coeff_name = base_input_name + '_coeff.mat'
                for key in coeffs.keys():
                    coeffs[key] = coeffs[key].cpu().numpy()
                savemat(os.path.join(coeff_save_path, coeff_name), coeffs)
                # save cropped overlay image
                overlay_name = base_input_name + '_overlay_crop.jpg'
                cv2.imwrite(os.path.join(overlay_save_path, overlay_name), eval_overlay_image)
                # save overlay image
                overlay_origin_name = base_input_name + '_overlay.jpg'
                cv2.imwrite(os.path.join(overlay_origin_save_path, overlay_origin_name), composed_img)
                # save image with projected landmarks
                proj_name = base_input_name + '_proj.jpg'
                cv2.imwrite(os.path.join(proj_save_path, proj_name), eval_proj_image)
                # save mesh
                face_shape, tri, face_color = mesh
                face_shape = face_shape[0].detach().cpu()
                tri = tri[0].detach().cpu()
                face_color = face_color[0].detach().cpu()
                obj_name = base_input_name + '_mesh.obj'
                write_obj(os.path.join(mesh_save_path, obj_name), face_shape, tri + 1, face_color)

            coeff_bs_array = torch.cat(coeff_bs_list)
            coeff_angle_array = torch.cat(coeff_angle_list)
            coeff_bs_array = coeff_bs_array.cpu().numpy()
            coeff_angle_array = coeff_angle_array.cpu().numpy()

            coeff_bs_name = 'coeff_bs.npy'
            coeff_angle_name = 'coeff_angle.npy'

            np.savetxt(os.path.join(save_path, coeff_bs_name), coeff_bs_array)
            np.savetxt(os.path.join(save_path, coeff_angle_name), coeff_angle_array)


    def render_shape(self, image_path, face_detector, save_path=None):
        if save_path is None:
            save_path = self.opts.save_path + '_model{}'.format(self.opts.test_iter)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.network.eval()
        with torch.no_grad():

            if os.path.isfile(image_path):
                image_paths = [image_path]
            else:
                image_paths = glob.glob(os.path.join(image_path, '*.jpg')) + glob.glob(os.path.join(image_path, '*.png'))

            for image_path in image_paths:

                frame = cv2.imread(image_path)
                # detect bbox
                if self.opts.detect_type == 'box':
                    boxes = face_detector(frame)
                    if len(boxes) == 0:
                        print('No face detected of {}'.format(image_path))
                        continue
                    bbox = boxes[0]
                else:
                    bbox, _ = face_detector.detect(frame)
                    if bbox is None:
                        print('No face detected of {}'.format(image_path))
                        continue
                # preprocess data
                # [bs, c, h, w]
                input_data, roi_box = Preprocess(frame, self.opts.input_size, bbox, self.opts.detect_type, return_box=True)
                input_data = input_data.to(self.device)
                # network inference
                pred_coeffs = self.network(input_data)  # [bs, 159]

                # get vertices on the original image plane
                face_shape_2d = self.FaceDecoder.get_face_on_2d_plane(pred_coeffs)
                face_shape_2d = face_shape_2d[0].detach().cpu()
                face_shape_2d_ori = similar_transform_3d(face_shape_2d, roi_box, self.opts.input_size)

                # vertices input for render function
                face_shape_2d_ori_array = face_shape_2d_ori.numpy()
                face_shape_2d_ori_array = face_shape_2d_ori_array
                face_shape_2d_ori_array[:, 2] = -1 * face_shape_2d_ori_array[:, 2]
                face_shape_2d_ori_array = face_shape_2d_ori_array.astype(np.float32).copy(order='C')

                # triangle input for render function
                tri = self.FaceDecoder.facemodel.tri.detach().cpu() - 1
                triangle_array = tri.numpy().astype(np.int32)
                triangle_array = triangle_array[:, ::-1]
                triangle_array = triangle_array.copy(order='C')

                from render_api import render
                eval_overlay_image = frame.copy()
                overlay_image = render(eval_overlay_image, face_shape_2d_ori_array, triangle_array, alpha=1.0)

                base_input_name = os.path.basename(image_path)[:-4]
                overlay_name = base_input_name + '_overlay.jpg'
                cv2.imwrite(os.path.join(save_path, overlay_name), overlay_image)


    def run_facial_motion_retargeting(self, src_coeff_path, target_img_path, face_detector, save_path=None):
        '''
        src_coeff_path: source 3DMM parameter path or dirs with *.mat format
        target_img_path: retarget object
        '''

        if save_path is None:
            save_path = self.opts.save_path + '_model{}'.format(self.opts.test_iter)

        if not os.path.exists(save_path):
            os.makedirs(save_path)


        self.network.eval()
        with torch.no_grad():
            tgt_img = cv2.imread(target_img_path)
            # detect bbox
            if self.opts.detect_type == 'box':
                tgt_boxes = face_detector(tgt_img)
                if len(tgt_boxes) == 0:
                    print('No face detected of {}'.format(target_img_path))
                    return
                tgt_bbox = tgt_boxes[0]

            else:
                tgt_bbox, _ = face_detector.detect(tgt_img)
                if tgt_bbox is None:
                    print('No face detected of {}'.format(target_img_path))
                    return

            # preprocess data
            # [bs, c, h, w]
            tgt_input_data, tgt_roi_box = Preprocess(tgt_img, self.opts.input_size, tgt_bbox, self.opts.detect_type, return_box=True)
            tgt_input_data = tgt_input_data.to(self.device)

            # network inference
            tgt_pred_coeffs = self.network(tgt_input_data)  # [bs, 159]

            # file dirs or file path
            if os.path.isfile(src_coeff_path):
                src_coeff_path_list = [src_coeff_path]
            else:
                src_coeff_path_list = glob.glob(os.path.join(src_coeff_path, '*.mat'))
                src_coeff_path_list.sort()

            length = len(src_coeff_path_list)
            for idx, coeff_path in enumerate(src_coeff_path_list):
                print('process frames [{}/{}].'.format(idx+1, length))
                # fetch retargeting parameters
                src_coeffs = loadmat(coeff_path)
                src_bs_coeff = src_coeffs['coeff_bs']
                src_angles = src_coeffs['coeff_angle']
                src_translation = src_coeffs['coeff_translation']
                src_bs_coeff = torch.from_numpy(src_bs_coeff).to(self.device)
                src_angles = torch.from_numpy(src_angles).to(self.device)
                src_translation = torch.from_numpy(src_translation).to(self.device)

                # fetch identity parameters from source image
                tgt_id_coeff, tgt_bs_coeff, tgt_tex_coeff, tgt_angles, tgt_gamma, tgt_translation = self.FaceDecoder.Split_coeff(tgt_pred_coeffs)

                # get retargeting result
                combined_coeffs = torch.cat([tgt_id_coeff, src_bs_coeff, tgt_tex_coeff, src_angles, tgt_gamma, src_translation], dim=1)
                rendered_img, _, _, retargeted_mesh = self.FaceDecoder.decode_face(combined_coeffs, return_coeffs=True)
                rendered_img = rendered_img[:, :, :, :3]

                rendered_img = rendered_img.squeeze().cpu().numpy()
                rendered_img = rendered_img[:, :, ::-1]  # [h,w,c] BGR

                base_input_name = os.path.basename(coeff_path)[:-4]
                # save retargeting result
                render_name = base_input_name + '_retarget.jpg'
                cv2.imwrite(os.path.join(save_path, render_name), rendered_img)

                face_shape, tri, face_color = retargeted_mesh
                face_shape = face_shape[0].detach().cpu()
                tri = tri[0].detach().cpu()
                face_color = face_color[0].detach().cpu()
                obj_name = base_input_name + '_mesh.obj'
                write_obj(os.path.join(save_path, obj_name), face_shape, tri + 1, face_color)

