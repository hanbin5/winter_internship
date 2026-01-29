import numpy as np

import torch.nn as nn
from torchvision import transforms

from models.conv_encoder_decoder.dense_depth import DenseDepth

from models.uareme.mnmaoptimiser import MNMAoptimiser
from models.uareme.mfopt import gtsam_rot


class UAREME(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.normal_vector_estimator = DenseDepth(
                 num_classes=4, 
                 B=5, 
                 pretrained=True, 
                 NF=2048, 
                 BN=False,
                 down=2, 
                 learned_upsampling=False,
        )

        self.single_frame_optimiser = MNMAoptimiser(
                 use_kappa=True
        )
        self.multi_frame_optimiser = gtsam_rot(
                 window_length=10,
                 interframe_sigma=0.1,
                 robust=True
        )

    def forward(self, imgs, **kwargs):
        """
        Args:
            imgs: (B, T, C, H, W) sequence batch
        Returns:
            list of T tensors, each (B, num_classes, H, W)
        """
        T = imgs.shape[1]
        R_opts = []
        R_opt = np.eye(3)
        for t in range(T):
            model_out = self.normal_vector_estimator(imgs[:, t])[0]  # (B, C, H, W) -> (B, num_classes, H, W)
            pred_norm = model_out[:, :3, :, :]  # (1, 3, H, W)
            norm_out = pred_norm.detach().cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
            pred_kappa = model_out[:, 3:, :, :] # (1, 1, H, W) 
            kappa_out = pred_kappa.detach().cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
            pred_kappa[pred_kappa > self.kappa_thresh] = self.kappa_thresh
            pred_norm_vec = pred_norm[0, ...].view(3, -1)   # (3, H*W)
            pred_kappa_vec = pred_kappa[0, ...].view(1, -1) # (1, H*W)

            R_torch, cov_torch = self.single_frame_optimiser(R_opt, pred_norm_vec, pred_kappa_vec)
            R_opt = R_torch.detach().numpy()
            cov = cov_torch.detach().numpy()

            R_opts.append(R_opt)
            R_opt = self.multi_frame_optimiser.optimise(R_opt, cov)

        return R_opts