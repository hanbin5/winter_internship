import os
import numpy as np

import torch

import sys
sys.path.append('../../')
import projects.uareme.config as config

import utils.utils as utils
from utils.input import prepare_images
from utils.visualisation import visualize_MFinImage
import cv2

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = config.get_args(test=True)

    script_dir = os.path.dirname(os.path.realpath(__file__))

    if args.ckpt_path is None:
        # checkpoints/{exp_name}/{exp_id}.pt
        args.ckpt_path = os.path.join(script_dir, 'checkpoints', args.exp_name, f'{args.exp_id}.pt')
    assert os.path.exists(args.ckpt_path), f"Checkpoint not found: {args.ckpt_path}"

    from models.uareme import UAREME

    model = UAREME(args).to(device)
    model = utils.load_checkpoint(args.ckpt_path, model)
    model.eval()

    # prepare color images from video or image pattern
    sample_videos_dir = os.path.join(script_dir, 'sample', 'videos')
    files = sorted([f for f in os.listdir(sample_videos_dir) if os.path.isfile(os.path.join(sample_videos_dir, f))])
    test_path = os.path.join(sample_videos_dir, files[0])
    color_images = prepare_images(test_path)
    print(f"Loaded {len(color_images)} images from {test_path}")

    R_traj = []

    with torch.no_grad():
        for color_image in color_images:
            R_opt, norm_out, kappa_out = model(color_image, format='RGB')
            # Returned values: R_opt (3, 3), norm_out (H, W, 3), kappa_out (H, W, 1)

            # Post processing
            R_traj.append(R_opt)

    # Save video with rotation axes overlay
    output_dir = os.path.join(script_dir, 'sample', 'output')
    os.makedirs(output_dir, exist_ok=True)

    h, w = color_images[0].shape[:2]
    video_path = os.path.join(output_dir, 'output.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 30, (w, h))

    for i, (color_image, R_opt) in enumerate(zip(color_images, R_traj)):
        img_vis = color_image.copy()
        img_vis = visualize_MFinImage(img_vis, R_opt)
        video_writer.write(cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))

    video_writer.release()
    print(f"Saved video to {video_path}")
