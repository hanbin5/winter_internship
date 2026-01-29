import torch
import torch.nn as nn
import torch.nn.functional as F

from projects.baseline_normal.losses import define_loss

class ComputeLoss(nn.Module):
    def __init__(self, args):
        """ args.loss_fn can be one of following:
            - L1        - L1 loss
            - L2        - L2 loss
            - AL        - Angular loss
            - NLL_vonmf - NLL of vonMF distribution
            - NLL_angmf - NLL of Angular vonMF distribution
        """
        super.__init__()

        self.loss_name = loss_name = args.loss_fn
        self.loss_fn = define_loss(loss_name)
    
    def forward(self, norm_out, gt_norm, gt_norm_mask):
        """ norm_out:       (B, 3, ...) or (B, 4, ...)
            gt_norm:        (B, 3, ...)
            gt_norm_mask:   (B, 1, ...)
        """
        loss = self.loss_fn(norm_out, gt_norm, gt_norm_mask)
        return loss