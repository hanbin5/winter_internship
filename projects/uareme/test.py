import os
import sys
import numpy as np
from tqdm import tqdm
import glob

import torch
import torch.nn.functional as F

import time
from torchvision import transforms
import cv2
from PIL import Image

import sys
sys.path.append('../../')
import utils.utils as utils
import utils.visualize as vis_utils

#↓↓↓↓
#NOTE: project-specific imports (e.g. config)
import projects.uareme.config as config
from projects.baseline_normal.dataloader import *
#↑↑↑↑


def test(args, model, test_loader, device, results_dir=None):
    with torch.no_grad():
        total_normal_errors = None

        for data_dict in tqdm(test_loader):

            #↓↓↓↓
            #NOTE: forward pass
            img = data_dict['img'].to(device)
            scene_names = data_dict['scene_name']
            img_names = data_dict['img_name']
            intrins = data_dict['intrins'].to(device)

            # pad input
            _, _, orig_H, orig_W = img.shape
            lrtb = utils.get_padding(orig_H, orig_W)
            img, intrins = utils.pad_input(img, intrins, lrtb)

            # forward pass
            pred_list = model(img, intrins=intrins, mode='test')
            norm_out = pred_list[-1]

            # crop the padded part
            norm_out = norm_out[:, :, lrtb[2]:lrtb[2]+orig_H, lrtb[0]:lrtb[0]+orig_W]

            pred_norm, pred_kappa = norm_out[:, :3, :, :], norm_out[:, 3:, :, :]
            pred_kappa = None if pred_kappa.size(1) == 0 else pred_kappa
            #↑↑↑↑

            if 'normal' in data_dict.keys():
                gt_norm = data_dict['normal'].to(device)
                gt_norm_mask = data_dict['normal_mask'].to(device)

                pred_error = utils.compute_normal_error(pred_norm, gt_norm)
                if total_normal_errors is None:
                    total_normal_errors = pred_error[gt_norm_mask]
                else:
                    total_normal_errors = torch.cat((total_normal_errors, pred_error[gt_norm_mask]), dim=0)

            if results_dir is not None:
                prefixs = ['%s_%s' % (i,j) for (i,j) in zip(scene_names, img_names)]
                vis_utils.visualize_normal(results_dir, prefixs, img, pred_norm, pred_kappa,
                                           gt_norm, gt_norm_mask, pred_error)

        if total_normal_errors is not None:
            metrics = utils.compute_normal_metrics(total_normal_errors)
            print("mean median rmse 5 7.5 11.25 22.5 30")
            print("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f" % (
                metrics['mean'], metrics['median'], metrics['rmse'],
                metrics['a1'], metrics['a2'], metrics['a3'], metrics['a4'], metrics['a5']))


if __name__ == '__main__':
    device = torch.device('cuda')
    args = config.get_args(test=True)

    if args.ckpt_path is None:
        ckpt_paths = glob.glob(os.path.join(args.output_dir, 'models', '*.pt'))
        ckpt_paths.sort()
        args.ckpt_path = ckpt_paths[-1]
    assert os.path.exists(args.ckpt_path)

    #↓↓↓↓
    #NOTE: define and load model
    from models.uareme import UAREME
    model = UAREME(args).to(device)
    model.eval()
    #↑↑↑↑

    # test the model
    if args.mode == 'benchmark':
        # do not resize/crop the images when benchmarking
        args.input_height = args.input_width = 0
        args.data_augmentation_same_fov = 0

        for dataset_name, split in [('nyuv2', 'test'),
                                    ('scannet', 'test'),
                                    ('ibims', 'ibims'),
                                    ('sintel', 'sintel'),
                                    ('vkitti', 'vkitti'),
                                    ('oasis', 'val')
                                    ]:

            args.dataset_name_test = dataset_name
            args.test_split = split
            test_loader = TestLoader(args).data

            results_dir = None
            if args.visualize:
                results_dir = os.path.join(args.output_dir, 'test', dataset_name)
                os.makedirs(results_dir, exist_ok=True)

            print(f"\n=== Testing on {dataset_name} ({split}) ===")
            test(args, model, test_loader, device, results_dir)

    else:
        print(f"Unknown mode: {args.mode}")
        print("Available modes: benchmark")
