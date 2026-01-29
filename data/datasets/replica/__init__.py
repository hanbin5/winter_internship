import os
import json
import cv2
import numpy as np

from data import Sample

from projects import DATASET_DIR
DATASET_PATH = os.path.join(DATASET_DIR, 'omnidata_starter_dataset')


def get_sample(args, sample_path, info):
    """Load a single sample from Replica dataset.

    Args:
        args: Arguments containing load flags (load_img, load_depth, load_intrins, etc.)
        sample_path: Path like "replica/apartment_0/point_0_view_0"
        info: Additional info dict (may contain crop_H, crop_W, etc.)

    Returns:
        Sample object with loaded data
    """
    # Parse path: "replica/apartment_0/point_0_view_0"
    parts = sample_path.split('/')
    scene_name = '/'.join(parts[:-1])  # "replica/apartment_0"
    img_name = parts[-1]  # "point_0_view_0"

    # Construct file paths
    rgb_path = os.path.join(DATASET_PATH, 'rgb', sample_path + '_domain_rgb.png')
    depth_path = os.path.join(DATASET_PATH, 'depth_euclidean', sample_path + '_domain_depth_euclidean.png')
    mask_path = os.path.join(DATASET_PATH, 'mask_valid', sample_path + '_domain_mask_valid.png')
    pose_path = os.path.join(DATASET_PATH, 'point_info', sample_path + '_domain_fixatedpose.json')

    # Read image (H, W, 3)
    img = None
    if args.load_img:
        assert os.path.exists(rgb_path), f"RGB not found: {rgb_path}"
        img = cv2.cvtColor(cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

    # Read depth (H, W, 1)
    depth = depth_mask = None
    if getattr(args, 'load_depth', False):
        assert os.path.exists(depth_path), f"Depth not found: {depth_path}"
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0  # Convert to meters
        else:
            depth = depth.astype(np.float32)
        if len(depth.shape) == 2:
            depth = depth[:, :, np.newaxis]

        # Load mask if available
        if os.path.exists(mask_path):
            depth_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            depth_mask = (depth_mask > 0).astype(np.float32)[:, :, np.newaxis]
        else:
            depth_mask = (depth > 0).astype(np.float32)

    # Read intrinsics from pose info
    intrins = None
    pose_data = None
    if args.load_intrins or getattr(args, 'load_pose', False):
        assert os.path.exists(pose_path), f"Pose not found: {pose_path}"
        with open(pose_path, 'r') as f:
            pose_data = json.load(f)

        if args.load_intrins:
            # Construct intrinsics from FOV and resolution
            fov = pose_data['field_of_view_rads']
            resolution = pose_data['resolution']
            focal = resolution / (2 * np.tan(fov / 2))
            cx, cy = resolution / 2, resolution / 2
            intrins = np.array([
                [focal, 0, cx],
                [0, focal, cy],
                [0, 0, 1]
            ], dtype=np.float32)

    # Store pose in info if requested
    if getattr(args, 'load_pose', False) and pose_data is not None:
        info = info.copy()
        info['pose'] = {
            'camera_location': np.array(pose_data['camera_location'], dtype=np.float32),
            'camera_rotation_quaternion': np.array(pose_data['camera_rotation_final_quaternion'], dtype=np.float32),
            'point_location': np.array(pose_data['point_location'], dtype=np.float32),
        }

    # Normal is not directly available in omnidata format for replica
    # It can be computed from depth if needed
    normal = normal_mask = None

    sample = Sample(
        img=img,
        depth=depth,
        depth_mask=depth_mask,
        normal=normal,
        normal_mask=normal_mask,
        intrins=intrins,

        dataset_name='replica',
        scene_name=scene_name,
        img_name=img_name,
        info=info
    )

    return sample


def get_sequences(scene_path):
    """Get all sequences (grouped by point) in a scene.

    Args:
        scene_path: Scene name like "replica/apartment_0"

    Returns:
        Dict mapping point_id to list of view sample paths
    """
    rgb_dir = os.path.join(DATASET_PATH, 'rgb', scene_path)

    if not os.path.exists(rgb_dir):
        return {}

    # Group files by point_id
    sequences = {}
    for filename in os.listdir(rgb_dir):
        if not filename.endswith('_domain_rgb.png'):
            continue

        # Parse: "point_0_view_5_domain_rgb.png"
        base = filename.replace('_domain_rgb.png', '')
        parts = base.split('_')
        point_id = int(parts[1])
        view_id = int(parts[3])

        if point_id not in sequences:
            sequences[point_id] = []

        sample_path = f"{scene_path}/{base}"
        sequences[point_id].append((view_id, sample_path))

    # Sort views within each sequence by view_id
    for point_id in sequences:
        sequences[point_id].sort(key=lambda x: x[0])
        sequences[point_id] = [path for _, path in sequences[point_id]]

    return sequences
