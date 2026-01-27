import os
import sys
import glob
import numpy as np

import torch
import cv2

import sys
sys.path.append('../../')
import utils.utils as utils
import projects.uareme.config as config
from utils.input import prepare_images
from utils.visualisation import visualize_MFinImage

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = config.get_args(test=True)
    assert os.path.exists(args.ckpt_path)

    from models.uareme import UAREME

    model = UAREME(args).to(device)
    model = utils.load_checkpoint(args.ckpt_path, model)
    model.eval()

    file_name = 'Boston-Drone.mp4'
    test_path = os.path.join('./samples/videos', file_name)
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
    output_dir = os.path.join('./sample/output')
    os.makedirs(output_dir, exist_ok=True)

    h, w = color_images[0].shape[:2]
    video_path = os.path.join(output_dir, file_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 30, (w, h))

    for i, (color_image, R_opt) in enumerate(zip(color_images, R_traj)):
        img_vis = color_image.copy()
        img_vis = visualize_MFinImage(img_vis, R_opt)
        video_writer.write(cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))

    video_writer.release()
    print(f"Saved video to {video_path}")
