import os
import time
import datetime
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from skimage import draw

from core.data_loader import get_voxceleb2_loader, get_trainset_loader
from core.losses import *
from model.resnet import ResNet50_3DMM
from model.mobilenetv2 import MobilenetV2_3DMM
from core.face_decoder import FaceDecoder



class TrainSolver(object):

    def __init__(self, opts):
        self.opts = opts
        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # face decoder
        self.FaceDecoder = FaceDecoder(opts.gpmm_model_path, opts.gpmm_delta_bs_path, batch_size=self.opts.batch_size, device=self.device, img_size=opts.input_size)


        # data loader
        print('load dataset...')
        if opts.dataset == 'voxceleb2':
            self.data_loader = get_voxceleb2_loader(opts.voxceleb2_root, opts.n_frames, opts.input_size, opts.batch_size,
                                            is_train=True, num_workers=opts.num_workers,
                                            use_data_files=opts.use_data_files, save_pickle=opts.save_pickle)
        elif opts.dataset == 'full':
            self.data_loader = get_trainset_loader(opts.voxceleb2_root, opts.feafa_root, opts.lp_300w_root,
                                                opts.n_frames, opts.input_size, opts.batch_size,
                                                is_train=True, num_workers=opts.num_workers,
                                                use_data_files=opts.use_data_files, save_pickle=opts.save_pickle)

            print('dataset length: ', len(self.data_loader.dataset))

            # build models and initialize optimizer
            self.build_model()

            # logger
            self.logger = SummaryWriter(opts.log_dir)

    def build_model(self):
        """
        Build models and initialize optimizers.
        """

        # initialize network
        self.init_model()

        # build optimizers
        self.optimizer = torch.optim.Adam(self.network.parameters(), self.opts.lr)


        # load pretrained face recognition network
        if not self.opts.IDENTITY_LOSS_W == 0:
            from facenet_pytorch import InceptionResnetV1
            self.FaceNet = InceptionResnetV1(pretrained='vggface2', device=self.device).eval()
            # fix parameters
            for param in self.FaceNet.parameters():
                param.requires_grad = False


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
        print('Loading the trained models from step {}...'.format(resume_iters))

        net_path = os.path.join(self.opts.checkpoint_dir, '{}-network.ckpt'.format(resume_iters))
        self.network.load_state_dict(torch.load(net_path, map_location=lambda storage, loc: storage))


    def reset_grad(self):
        """
        Reset the gradient buffers.
        """
        self.optimizer.zero_grad()


    def visualize_results(self, input_data_i, rendered_img_i, render_mask, landmark_i, pred_lm2d, face_mask_i):

        # get overlay images
        eval_input_images = input_data_i  # [bs, c, h, w]
        eval_rendered_images = rendered_img_i / 255
        eval_rendered_images = eval_rendered_images.permute(0, 3, 1, 2)  # to [bs, c, h, w]
        eval_mask = (render_mask > 0).type(torch.uint8)
        eval_mask = eval_mask.view(eval_mask.size(0), 1, eval_mask.size(1), eval_mask.size(2))
        eval_mask = eval_mask.repeat(1, 3, 1, 1)
        eval_overlay_images = eval_rendered_images * eval_mask + eval_input_images * (1 - eval_mask)

        # get face masks
        eval_face_masks = face_mask_i  # [bs, h, w]
        eval_face_masks = eval_face_masks.unsqueeze(dim=1)  # [bs, 1, h, w]
        eval_face_masks = eval_face_masks.repeat(1, 3, 1, 1)

        # project landmarks on input images
        eval_proj_images = input_data_i.detach().cpu().numpy()
        eval_proj_images = eval_proj_images * 255  # [0,1] -> [0,255]
        eval_proj_images = eval_proj_images.transpose(0, 2, 3, 1)  # [bs, c, h, w] -> [bs, h, w, c]

        eval_gt_lm2d = landmark_i.detach().cpu().numpy()
        eval_pred_lm2d = pred_lm2d.detach().cpu().numpy()
        for bs_idx in range(eval_proj_images.shape[0]):
            proj_img = eval_proj_images[bs_idx]
            lm_2d = eval_gt_lm2d[bs_idx]
            lm2d_pred = eval_pred_lm2d[bs_idx]

            for k in range(68):
                rr, cc = draw.disk((lm_2d[k][1], lm_2d[k][0]), 2)
                draw.set_color(proj_img, [rr, cc], [255, 255, 255])  # while

                rr1, cc1 = draw.disk((lm2d_pred[k][1], lm2d_pred[k][0]), 2)
                draw.set_color(proj_img, [rr1, cc1], [255, 0, 0])  # red

        eval_proj_images = eval_proj_images / 255
        eval_proj_images = eval_proj_images.transpose(0, 3, 1, 2)
        eval_proj_images = torch.from_numpy(eval_proj_images).to(self.device)

        visual_result = torch.cat((eval_input_images, eval_face_masks, eval_proj_images, eval_overlay_images, eval_rendered_images))

        return visual_result


    def train(self):
        data_iterator = iter(self.data_loader)

        # Start training from scratch or resume training.
        start_iters = 0
        if self.opts.resume_iters:
            start_iters = self.opts.resume_iters
            self.restore_model(self.opts.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()

        # set the weight of expression-exclusive loss to zero in the beginning.
        self.opts.CONFLICT_LOSS_W = 0

        for i in range(start_iters, self.opts.num_iters):
            # Fetch input data
            try:
                input_data, landmark, skin_mask, front_flag = next(data_iterator)
            except:
                data_iterator = iter(self.data_loader)
                input_data, landmark, skin_mask, front_flag = next(data_iterator)

            # input image
            input_data = input_data.to(self.device)  # [bs, n_frames, c, h, w]
            # input landmark
            landmark = landmark.to(self.device)  # [bs, n_frames, 68, 2]
            # face skin mask
            skin_mask = skin_mask.to(self.device)  # [bs, n_frames, h, w]
            # flag depicting whether this face is front or not 
            front_flag = front_flag.to(self.device)  # [bs, n_frames, 1]


            #### training
            # 3D face reconstruction losses
            total_photo_loss = 0
            total_identity_loss = 0
            total_lm_loss = 0
            total_reg_loss = 0
            total_sparse_reg_loss = 0
            # identity consistent loss
            total_id_consistent_loss = 0
            # expression exclusive loss
            total_exp_exclusive_loss = 0

            # storing identity coefficients
            coeff_id_list = []

            for n_idx in range(self.opts.n_frames):
                input_data_i = input_data[:, n_idx, :, :, :]
                landmark_i = landmark[:, n_idx, :, :]
                skin_mask_i = skin_mask[:, n_idx, :, :]
                front_flag_i = front_flag[:, n_idx, :]

                pred_coeffs_i = self.network(input_data_i)

                # decoding face with predicted 3DMM paramters
                rendered_img, pred_lm2d, coeff_bs, coeff_id, coeff_tex = self.FaceDecoder.decode_face(pred_coeffs_i)
                coeff_id_list.append(coeff_id)

                mask = rendered_img[:, :, :, 3].detach()
                rendered_img_i = rendered_img[:, :, :, :3]
                # [bs, c, h, w] ->[bs, h, w, c]
                input_image_i = input_data_i.permute(0, 2, 3, 1) * 255

                # losses
                if not self.opts.PHOTO_LOSS_W == 0:
                    photo_loss = photo_loss_skin_mask(rendered_img_i, input_image_i, mask > 0, skin_mask_i)
                    total_photo_loss += photo_loss / self.opts.n_frames

                if not self.opts.IDENTITY_LOSS_W == 0:
                    input_face_i = input_data_i
                    # [bs, h, w, c] -> [bs, c, h, w], [0,255] -> [0,1]
                    rendered_face_i = rendered_img_i.permute(0, 3, 1, 2) / 255.0

                    # get face embedding
                    face_embedding_gt = self.FaceNet(input_face_i)
                    face_embedding_pred = self.FaceNet(rendered_face_i)

                    identity_loss = face_identity_loss(face_embedding_pred, face_embedding_gt)
                    total_identity_loss += identity_loss / self.opts.n_frames

                if not self.opts.LM_LOSS_W == 0:
                    lm_loss = landmark_loss_combined(pred_lm2d, landmark_i, front_flag_i)
                    total_lm_loss += lm_loss / self.opts.n_frames


                if not self.opts.REG_W == 0:
                    reg_loss = coeff_reg_loss(coeff_id, coeff_tex)
                    total_reg_loss += reg_loss / self.opts.n_frames

                if not self.opts.SP_LOSS_W == 0:
                    sparse_reg_loss = bs_coeff_reg_loss(coeff_bs)
                    total_sparse_reg_loss += sparse_reg_loss / self.opts.n_frames

                if not self.opts.CONFLICT_LOSS_W == 0:
                    conflict_loss = expression_exclusive_loss(coeff_bs)
                    total_exp_exclusive_loss += conflict_loss


            if not self.opts.ID_CON_LOSS_W == 0:
                # calculate average identity coeff
                coeff_id_tensors = torch.cat(coeff_id_list, dim=0) # [n_frame x n_bs, n_id]
                coeff_id_tensors = coeff_id_tensors.view(self.opts.n_frames, self.opts.batch_size, -1)
                psedo_id_label = torch.mean(coeff_id_tensors, dim=0)
                # fixed value
                psedo_id_label = psedo_id_label.detach()
                for n_idx in range(self.opts.n_frames):
                    total_id_consistent_loss += id_consistent_loss(coeff_id_list[n_idx], psedo_id_label)


            total_loss = total_photo_loss * self.opts.PHOTO_LOSS_W + \
                         total_lm_loss * self.opts.LM_LOSS_W + \
                         total_reg_loss * self.opts.REG_W + \
                         total_sparse_reg_loss * self.opts.SP_LOSS_W + \
                         total_identity_loss * self.opts.IDENTITY_LOSS_W + \
                         total_id_consistent_loss * self.opts.ID_CON_LOSS_W + \
                         total_exp_exclusive_loss * self.opts.CONFLICT_LOSS_W

            self.reset_grad()
            total_loss.backward()
            self.optimizer.step()

            # Logging.
            loss = {}
            loss['train/total_loss'] = total_loss.item()
            if not self.opts.PHOTO_LOSS_W == 0:
                loss['train/photo_loss'] = total_photo_loss.item()
            if not self.opts.LM_LOSS_W == 0:
                loss['train/landmark_loss'] = total_lm_loss.item()
            if not self.opts.REG_W == 0:
                loss['train/reg_loss'] = total_reg_loss.item()
            if not self.opts.SP_LOSS_W == 0:
                loss['train/sparse_reg_loss'] = total_sparse_reg_loss.item()
            if not self.opts.IDENTITY_LOSS_W == 0:
                loss['train/identity_loss'] = total_identity_loss.item()
            if not self.opts.ID_CON_LOSS_W == 0:
                loss['train/id_consistent_loss'] = total_id_consistent_loss.item()
            if not self.opts.CONFLICT_LOSS_W == 0:
                loss['train/exp_exclusive_loss'] = total_exp_exclusive_loss.item()

            
            # add expression exclusive loss at this moment
            if  self.opts.CONFLICT_LOSS_W == 0 and (i+1) >= self.opts.num_iters * 2 / 3:
                self.opts.CONFLICT_LOSS_W = 10


            # print out training information.
            if (i + 1) % self.opts.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.opts.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                # log
                for tag, value in loss.items():
                    self.logger.add_scalar(tag, value, i + 1)

            # evaluation
            if (i + 1) % self.opts.eval_step == 0:
                # visualization
                visual_result = self.visualize_results(input_data_i, rendered_img_i, mask, landmark_i, pred_lm2d, skin_mask_i)

                # create grid of images
                img_grid = torchvision.utils.make_grid(visual_result, nrow=self.opts.batch_size)
                self.logger.add_image('visual_images', img_grid, i + 1)

            # save model checkpoints.
            if (i + 1) % self.opts.model_save_step == 0:
                net_path = os.path.join(self.opts.checkpoint_dir, '{}-network.ckpt'.format(i + 1))
                torch.save(self.network.state_dict(), net_path)

                print('Saved model checkpoints into {}...'.format(self.opts.checkpoint_dir))
