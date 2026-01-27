""" 
U-ARE-ME: Uncertainty-Aware Rotation Estimation in Manhattan Environments
Aalok Patwardhan, Callum Rhodes, Gwangbin Bae, Andrew J. Davison 2024.
https://callum-rhodes.github.io/U-ARE-ME/
Copyright (c) 2025 by the authors.
This code is licensed (see LICENSE for details)

The file contains a U-ARE-ME class wrapper

"""

"""
INIT:       UAREME(args) where args contains:
            - ckpt_path: str (checkpoint path)
            - b_kappa: bool (default: True)
            - kappa_threshold: float (default: 75.0)
            - b_multiframe: bool (default: True)
            - b_robust: bool (default: True)
            - window_length: int (default: 30)
            - interframe_sigma: float (default: 0.75)
            - use_trt: bool (default: False)

CALL:       R_opt, norm_out, kappa_out = model(img, format='RGB')
            - img: numpy array (H, W, 3), uint8
            - R_opt: (3, 3) rotation matrix
            - norm_out: (H, W, 3) surface normals
            - kappa_out: (H, W, 1) confidence

If using multiframe, input images must be sequential.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn

from torchvision import transforms
from utils.MNMAoptimiser import MNMAoptimiser
from utils.input import preprocess_img, define_model
import utils.visualisation as vis_utils
import cv2
import time


class UAREME(nn.Module):
    def __init__(self, args):
        super().__init__()

        # Kappa settings
        self.use_kappa = getattr(args, 'b_kappa', True)
        self.kappa_thresh = getattr(args, 'kappa_threshold', 75.0)

        # Multiframe settings
        self.use_multi = getattr(args, 'b_multiframe', True)
        if self.use_multi:
            from utils import MFOpt as mfOpt_utils
            window_length = getattr(args, 'window_length', 30)
            interframe_sigma = getattr(args, 'interframe_sigma', 0.75)
            b_robust = getattr(args, 'b_robust', True)

            self.multiframe_optimiser = mfOpt_utils.gtsam_rot(
                window_length=window_length,
                interframe_sigma=interframe_sigma,
                robust=b_robust
            )

        # Model settings
        self.device = torch.device("mps" if torch.cuda.is_available() else "cpu")
        checkpoints_dir = os.path.dirname(args.ckpt_path) if args.ckpt_path else None
        use_trt = getattr(args, 'use_trt', False)
        self.model = define_model(self.device, trt=use_trt, checkpoints_dir=checkpoints_dir)
        self.normalise_fn = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.R_opt = np.eye(3)

        self.singleframe_optimiser = MNMAoptimiser(use_kappa=self.use_kappa)
    
        self.MODES_DICT = {1: 'RGB', 2: 'Normals', 3: 'Confidence'}
        self.DISPLAY_MODE = self.MODES_DICT[1]
        self.TITLE_FONT = cv2.FONT_HERSHEY_SIMPLEX                                        # List of displays
        self.TITLE_WIDTHS = {self.MODES_DICT[i]: cv2.getTextSize(t, self.TITLE_FONT, 1, 2)[0][0] 
                        + 20 for i, t in self.MODES_DICT.items()}                                            # Display title width 
        self.DISPLAY_WIDTH = 56
        self.prev_frame_time = time.time()


    def forward(self, img : np.ndarray, **kwargs):
        # Preprocess image to torch format. The input image must be uint8 format
        img_torch = preprocess_img(img, format, self.device, self.normalise_fn)

        ####################################################################################
        # Run image through network
        if img_torch is not None:
            with torch.no_grad():
                model_out = self.model(img_torch)[0]
                pred_norm = model_out[:, :3, :, :]  # (1, 3, H, W)
                norm_out = pred_norm.detach().cpu().permute(0, 2, 3, 1).squeeze(0).numpy()  # (H, W, 3)
                pred_kappa = model_out[:, 3:, :, :]  # (1, 1, H, W)
                kappa_out = pred_kappa.detach().cpu().permute(0, 2, 3, 1).squeeze(0).numpy()  # (H, W, 1)
                pred_kappa[pred_kappa > self.kappa_thresh] = self.kappa_thresh
                pred_norm_vec = pred_norm[0,...].view(3, -1)  # (3, H*W)
                pred_kappa_vec = pred_kappa[0,...].view(1, -1)  # (1, H*W) 
        
        ####################################################################################
        # MNMA Rotation optimisation
        init_R = self.R_opt if self.use_multi else np.eye(3)
        R_torch, cov_torch = self.singleframe_optimiser.optimise(init_R, pred_norm_vec, pred_kappa_vec)
        self.R_opt = R_torch.detach().numpy()        # Optimised rotation estimate
        cov = cov_torch.detach().numpy()        # Covariance of rotation estimate

        ####################################################################################
        # Multiframe Optimisation
        if self.use_multi:
            self.R_opt = self.multiframe_optimiser.optimise(self.R_opt, cov)

        return self.R_opt.copy(), norm_out, kappa_out