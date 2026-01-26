""" Get samples from UZH-FPV Drone Racing Dataset (https://fpv.ifi.uzh.ch/)
    NOTE: This dataset contains grayscale images from Qualcomm Snapdragon Flight board
    NOTE: Camera intrinsics are from Kalibr calibration (indoor forward facing configuration)
    NOTE: Pose groundtruth is in format: frame_id tx ty tz qx qy qz qw
"""
import os
import cv2
import numpy as np

from data import Sample

# Dataset path configuration
DATASET_DIR = os.path.expanduser('~/data')
DATASET_PATH = os.path.join(DATASET_DIR, 'UZH-FPV')

# Camera intrinsics for Snapdragon camera (cam0 - left stereo)
# From: https://github.com/rpng/open_vins/blob/master/config/uzhfpv_indoor/kalibr_imucam_chain.yaml
INTRINSICS = np.array([
    [278.667,   0.0,     319.752],
    [  0.0,   278.490,   241.969],
    [  0.0,     0.0,       1.0  ]
], dtype=np.float32)

# Distortion coefficients (equidistant model)
DISTORTION = np.array([-0.01372, 0.02073, -0.01279, 0.00252], dtype=np.float32)


def load_groundtruth(gt_path):
    """Load groundtruth poses from file.

    Returns:
        dict: {frame_id: {'tx': float, 'ty': float, 'tz': float,
                         'qx': float, 'qy': float, 'qz': float, 'qw': float}}
    """
    poses = {}
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 8:
                frame_id = int(parts[0])
                poses[frame_id] = {
                    'tx': float(parts[1]),
                    'ty': float(parts[2]),
                    'tz': float(parts[3]),
                    'qx': float(parts[4]),
                    'qy': float(parts[5]),
                    'qz': float(parts[6]),
                    'qw': float(parts[7])
                }
    return poses


def get_sample(args, sample_path, info):
    """Get a sample from UZH-FPV dataset.

    Args:
        args: Arguments object with flags like load_img, load_intrins, load_pose
        sample_path: Path like "indoor_forward_3/rgb/0.png"
        info: Additional info dict

    Returns:
        Sample: Sample object containing image, pose, intrinsics, etc.
    """
    # Parse sample path
    # e.g. sample_path = "indoor_forward_3/rgb/0.png"
    parts = sample_path.split('/')
    scene_name = parts[0]
    img_name = os.path.splitext(parts[-1])[0]  # "0" from "0.png"

    img_path = os.path.join(DATASET_PATH, sample_path)
    gt_path = os.path.join(DATASET_PATH, scene_name, 'groundtruth.txt')

    assert os.path.exists(img_path), f"Image not found: {img_path}"

    # Read image (H, W, 3) - convert grayscale to RGB if needed
    img = None
    if getattr(args, 'load_img', True):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            # Convert grayscale to RGB if needed
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0

    # Read intrinsics (3, 3)
    intrins = None
    if getattr(args, 'load_intrins', True):
        intrins = INTRINSICS.copy()

    # Read pose
    pose = None
    if getattr(args, 'load_pose', True):
        if os.path.exists(gt_path):
            poses = load_groundtruth(gt_path)
            frame_id = int(img_name)
            if frame_id in poses:
                p = poses[frame_id]
                pose = np.array([
                    p['tx'], p['ty'], p['tz'],
                    p['qx'], p['qy'], p['qz'], p['qw']
                ], dtype=np.float32)

    sample = Sample(
        img=img,
        pose=pose,
        intrins=intrins,

        dataset_name='uzh_fpv',
        scene_name=scene_name,
        img_name=img_name,
        info=info
    )

    return sample


def _get_image_dir(scene_name):
    """Get image directory for a scene (rgb or output folder)."""
    rgb_dir = os.path.join(DATASET_PATH, scene_name, 'rgb')
    if os.path.exists(rgb_dir):
        return rgb_dir, 'rgb'

    output_dir = os.path.join(DATASET_PATH, scene_name, 'output')
    if os.path.exists(output_dir):
        return output_dir, 'output'

    return None, None


def get_all_samples(scene_name):
    """Get all sample paths for a given scene.

    Args:
        scene_name: Scene name like "indoor_forward_3" or "race_1"

    Returns:
        list: List of sample paths like ["indoor_forward_3/rgb/0.png", ...]
    """
    img_dir, folder_name = _get_image_dir(scene_name)
    if img_dir is None:
        return []

    samples = []
    for fname in sorted(os.listdir(img_dir), key=lambda x: int(os.path.splitext(x)[0])):
        if fname.endswith('.png'):
            samples.append(f"{scene_name}/{folder_name}/{fname}")

    return samples


def get_available_scenes():
    """Get list of available scenes in the dataset.

    Returns:
        list: List of scene names
    """
    if not os.path.exists(DATASET_PATH):
        return []

    scenes = []
    for name in os.listdir(DATASET_PATH):
        scene_dir = os.path.join(DATASET_PATH, name)
        if os.path.isdir(scene_dir):
            img_dir, _ = _get_image_dir(name)
            if img_dir is not None:
                scenes.append(name)

    return sorted(scenes)
