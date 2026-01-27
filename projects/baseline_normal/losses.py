import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

COS_EPS = 1e-7

def define_loss(loss_name):
    if loss_name == 'L1':
        return l1_loss
    elif loss_name == 'L2':
        return l2_loss
    elif loss_name == 'AL':
        return angular_loss
    elif loss_name == 'NLL_vonmf':
        return vonmf_loss
    elif loss_name == 'NLL_angmf':
        return angmf_loss
    else:
        raise Exception('invalid loss fn name: %s' % loss_name)

def l1_loss(norm_out, gt_norm, gt_norm_mask):
    """ norm_out:       (B, 3, ...)
        gt_norm:        (B, 3, ...)
        gt_norm_mask:   (B, 1, ...)
    """
    pred_norm = norm_out[:, 0:3, ...]

    l1 = torch.sum(torch.abs(gt_norm - pred_norm), dim=1, keepdim=True)
    l1 = l1[gt_norm_mask]
    return torch.mean(l1)

def l2_loss(norm_out, gt_norm, gt_norm_mask):
    """ norm_out:       (B, 3, ...)
        gt_norm:        (B, 3, ...)
        gt_norm_mask:   (B, 1, ...)
    """
    pred_norm = norm_out[:, 0:3, ...]

    l2 = torch.sum(torch.square(gt_norm - pred_norm), dim=1, keepdim=True)
    l2 = l2[gt_norm_mask]
    return torch.mean(l2)

def angular_loss(norm_out, gt_norm, gt_norm_mask):
    """ norm_out:       (B, 3, ...)
        gt_norm:        (B, 3, ...)
        gt_norm_mask:   (B, 1, ...)
    """
    pred_norm = norm_out[:, 0:3, ...]
    dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1).unsqueeze(1)
    valid_mask = torch.logical_and(gt_norm_mask, torch.abs(dot.detach()) < 1-COS_EPS)
    angle = torch.acos(dot[valid_mask])
    return torch.mean(angle)

def nll_vonmf(dot, pred_kappa):
    loss = - torch.log(pred_kappa) - (pred_kappa * (dot - 1)) + torch.log(1 - torch.exp(- 2 * pred_kappa))
    return loss

def vonmf_loss(norm_out, gt_norm, gt_norm_mask):
    """ norm_out:       (B, 4, ...)
        gt_norm:        (B, 3, ...)
        gt_norm_mask:   (B, 1, ...)
    """
    pred_norm, pred_keppa = norm_out[:, 0:3, ...], norm_out[:, 3:, ...]

    dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1).unsqueeze(1)
    valid_mask = torch.logical_and(gt_norm_mask, torch.abs(dot.detach()) < 1-COS_EPS)

    nll = nll_vonmf(dot[valid_mask], pred_keppa[valid_mask])
    return torch.mean(nll)

def nll_angmf(dot, pred_kappa):
    loss = - torch.log(torch.square(pred_kappa) + 1) + pred_kappa * torch.acos(dot) + torch.log(1 + torch.exp(-pred_kappa * np.pi))
    return loss

def angmf_loss(norm_out, gt_norm, gt_norm_mask):
    """ norm_out:       (B, 4, ...)
        gt_norm:        (B, 3, ...)
        gt_norm_mask:   (B, 1, ...)
    """
    pred_norm, pred_kappa = norm_out[:, 0:3, ...], norm_out[:, 3:, ...]

    dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1).unsqueeze(1)
    valid_mask = torch.logical_and(gt_norm_mask, torch.abs(dot.detach()) < 1-COS_EPS)

    nll = nll_angmf(dot[valid_mask], pred_kappa[valid_mask])
    return torch.mean(nll)