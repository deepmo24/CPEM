import numpy as np
import torch
import torch.nn.functional as F


def photo_loss_skin_mask(pred_img, gt_img, render_mask, skin_mask):
    '''
    l_2,1 norm loss for rendered face
    :param pred_img:  [bs, h, w, c]
    :param gt_img:  [bs, h, w, c]
    :param render_mask: image render mask [bs, h, w], assumes the value lies in [0,1]
    :param skin_mask: external skin mask, [bs, h, w], assumes the value lies in [0,1]
    :return: loss
    '''
    mask_overlap = render_mask * skin_mask 

    # l2 norm for RGB color space
    # mask other region except face skin
    loss = torch.sum(torch.square(pred_img - gt_img), 3)
    loss = torch.sqrt(loss + 1e-6) * mask_overlap

    # l1 norm for pixel position space
    loss = torch.sum(loss, dim=(1, 2)) / (torch.sum(mask_overlap, dim=(1, 2)) + 1e-6)
    loss = torch.mean(loss)

    return loss


def face_identity_loss(pred_img, gt_img):
    cos = torch.nn.CosineSimilarity(dim=1)

    loss = 1 - cos(pred_img, gt_img)
    loss = torch.mean(loss)

    return loss


def landmark_loss(pred_lms, gt_lms):
    '''
    mse loss for 2d landmarks
    :param pred_lms: [bs, 68, 2]
    :param gt_lms: [bs, 68, 2]
    :return:
    '''
    # Increase the weight of inner face region
    w = torch.ones((1, 68)).to(pred_lms.device)
    w[:, 17:68] = 10 

    loss = torch.sum(torch.square(pred_lms - gt_lms), dim=2) * w
    loss = torch.mean(loss)

    return loss


def landmark_loss_combined(pred_lms, gt_lms, front_flag):   
    '''
    mse loss for 2d landmarks
    :param pred_lms: [bs, 68, 2]
    :param gt_lms: [bs, 68, 2]
    :param front_flag: [bs,1]
    :return:
    '''
    bs = pred_lms.size(0)

    # Increase the weight of inner face region
    w = torch.ones((bs, 68)).to(pred_lms.device)
    w[:, 17:36] = 10 
    w[:, 48:68] = 10

    # for the eye landmarks:
    nonzero_idx = torch.nonzero(front_flag)[:, 0]
    w[nonzero_idx, 36:48] = w[nonzero_idx, 36:48] * 10 * front_flag[nonzero_idx] 

    loss = torch.sum(torch.square(pred_lms - gt_lms), dim=2) * w
    loss = torch.mean(loss)

    return loss


def coeff_reg_loss(id_coeff, tex_coeff):

    loss = torch.square(id_coeff).sum() + \
            torch.square(tex_coeff).sum() * 1.7e-3

    return loss


def bs_coeff_reg_loss(bs_coeff):
    '''
    l1 loss to enforce sparse coefficients
    '''
    loss = torch.mean(torch.abs(bs_coeff))

    return loss



def id_consistent_loss(pred_id_coeff, gt_id_coeff, loss_type='l2', id_flag=None):

    diff = pred_id_coeff - gt_id_coeff
    if id_flag is not None:
        diff = diff * id_flag

    if loss_type == 'l1':
        loss = torch.mean(torch.abs(diff))
    elif loss_type == 'l2':
        # mean sqared loss
        loss = torch.mean(torch.square(diff))
    elif loss_type == 'cosine':
        cos = torch.nn.CosineSimilarity(dim=1)
        loss = 1 - cos(pred_id_coeff, gt_id_coeff)
        loss = torch.mean(loss)

    return loss



contradictory_index_list = [
    [[0, 2], [8]],
    [[1, 3], [9]],
    [[4], [12]],
    [[5], [13]],
    [[6], [10]],
    [[7], [11]],
    [[14], [17]],
    [[15], [18]],
    [[20], [22]],
    [[23], [24]],
    [[25], [27, 29, 31]],
    [[26], [28, 30, 32]],
    [[33], [35, 41]],
    [[34, 40], [36, 37]]
]


def expression_exclusive_loss(bs_coeff):

    total_loss = 0
    for idx in contradictory_index_list:
        left = bs_coeff[:, idx[0]]
        right = bs_coeff[:, idx[1]]

        left_max_v, _ = left.max(dim=1)
        right_max_v, _ = right.max(dim=1)

        flag = (left_max_v > right_max_v).type(torch.uint8)
        loss_left = (1-flag) * torch.sum(torch.square(left), dim=1)
        loss_right = flag * torch.sum(torch.square(right), dim=1)
        loss = torch.mean(loss_left) + torch.mean(loss_right)
        total_loss += loss

    return total_loss