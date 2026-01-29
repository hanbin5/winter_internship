import os
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from data.augmentations import get_transform
from data.datasets.replica import get_sample, get_sequences

import logging
logger = logging.getLogger('root')


class ReplicaSequenceDataset(Dataset):
    """Dataset that returns sequences of frames grouped by point.

    Each sequence contains multiple views of the same point location.
    The dataset supports variable-length sequences.
    """

    def __init__(self, args, scene_list, mode='train', transform=None):
        """
        Args:
            args: Arguments containing load flags and other configs
            scene_list: List of scene paths like ["replica/apartment_0", "replica/apartment_1"]
            mode: 'train' or 'test'
            transform: Transform to apply to each sample
        """
        self.args = args
        self.mode = mode
        self.transform = transform or get_transform(args, dataset_name='replica', mode=mode)

        # Collect all sequences from all scenes
        self.sequences = []  # List of (sequence_id, [sample_paths])
        self.sequence_lengths = []  # Length of each sequence

        for scene_path in scene_list:
            scene_sequences = get_sequences(scene_path)
            for point_id, sample_paths in scene_sequences.items():
                seq_id = f"{scene_path}/point_{point_id}"
                self.sequences.append((seq_id, sample_paths))
                self.sequence_lengths.append(len(sample_paths))

        logger.info(f"Loaded {len(self.sequences)} sequences from {len(scene_list)} scenes")
        logger.info(f"Sequence length range: {min(self.sequence_lengths)}-{max(self.sequence_lengths)}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        """Return all samples in a sequence.

        Returns:
            dict with:
                - 'sequence_id': str, identifier for this sequence
                - 'samples': list of dicts, each containing transformed sample data
                - 'length': int, number of frames in sequence
        """
        seq_id, sample_paths = self.sequences[index]

        samples = []
        for sample_path in sample_paths:
            sample = get_sample(self.args, sample_path, info={})
            if self.transform:
                sample = self.transform(sample)
            samples.append(sample)

        return {
            'sequence_id': seq_id,
            'samples': samples,
            'length': len(samples)
        }

    def get_sequence_length(self, index):
        """Get length of sequence at index without loading data."""
        return self.sequence_lengths[index]


class SequenceBucketBatchSampler(Sampler):
    """Batch sampler that groups sequences with similar lengths into buckets.

    This minimizes padding waste by ensuring sequences in the same batch
    have similar lengths.
    """

    def __init__(self, dataset, batch_size, bucket_boundaries=None, shuffle=True, drop_last=True, seed=0):
        """
        Args:
            dataset: ReplicaSequenceDataset instance
            batch_size: Number of sequences per batch
            bucket_boundaries: List of length boundaries for buckets.
                              Default: [5, 10, 15, 20, 30, 40, 50, inf]
            shuffle: Whether to shuffle within buckets
            drop_last: Whether to drop incomplete batches
            seed: Random seed for shuffling
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

        # Default bucket boundaries
        if bucket_boundaries is None:
            bucket_boundaries = [5, 10, 15, 20, 30, 40, 50, float('inf')]
        self.bucket_boundaries = bucket_boundaries

        # Assign sequences to buckets based on length
        self.buckets = defaultdict(list)
        for idx in range(len(dataset)):
            length = dataset.get_sequence_length(idx)
            bucket_id = self._get_bucket_id(length)
            self.buckets[bucket_id].append(idx)

        # Log bucket distribution
        for bucket_id, indices in sorted(self.buckets.items()):
            if bucket_id < len(bucket_boundaries) - 1:
                lower = bucket_boundaries[bucket_id - 1] if bucket_id > 0 else 0
                upper = bucket_boundaries[bucket_id]
                logger.info(f"Bucket {bucket_id} (len {lower}-{upper}): {len(indices)} sequences")
            else:
                logger.info(f"Bucket {bucket_id} (len {bucket_boundaries[-2]}+): {len(indices)} sequences")

        # Pre-compute batches
        self._create_batches()

    def _get_bucket_id(self, length):
        """Get bucket ID for a sequence of given length."""
        for i, boundary in enumerate(self.bucket_boundaries):
            if length <= boundary:
                return i
        return len(self.bucket_boundaries) - 1

    def _create_batches(self):
        """Create batches from bucketed sequences."""
        random.seed(self.seed)

        self.batches = []

        for bucket_id, indices in self.buckets.items():
            # Shuffle within bucket
            if self.shuffle:
                indices = indices.copy()
                random.shuffle(indices)

            # Create batches from this bucket
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    self.batches.append(batch)

        # Shuffle batch order
        if self.shuffle:
            random.shuffle(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

    def set_epoch(self, epoch):
        """Update seed for a new epoch."""
        self.seed = epoch
        self._create_batches()


def sequence_collate_fn(batch):
    """Collate function for variable-length sequences.

    Pads sequences to the maximum length in the batch.

    Args:
        batch: List of dicts from ReplicaSequenceDataset.__getitem__

    Returns:
        dict with:
            - 'imgs': (B, T, C, H, W) tensor, padded images
            - 'depths': (B, T, 1, H, W) tensor if available, padded depths
            - 'intrins': (B, T, 3, 3) tensor if available
            - 'poses': dict of (B, T, ...) tensors if available
            - 'lengths': (B,) tensor of actual sequence lengths
            - 'sequence_ids': list of sequence IDs
            - 'mask': (B, T) bool tensor, True for valid (non-padded) frames
    """
    batch_size = len(batch)
    max_length = max(item['length'] for item in batch)

    # Get dimensions from first valid sample
    first_sample = batch[0]['samples'][0]

    # Initialize output tensors
    result = {
        'sequence_ids': [item['sequence_id'] for item in batch],
        'lengths': torch.tensor([item['length'] for item in batch], dtype=torch.long),
        'mask': torch.zeros(batch_size, max_length, dtype=torch.bool)
    }

    # Check what data is available
    has_img = 'img' in first_sample
    has_depth = 'depth' in first_sample and first_sample['depth'] is not None
    has_intrins = 'intrins' in first_sample and first_sample['intrins'] is not None
    has_pose = 'pose' in first_sample.get('info', {})

    if has_img:
        C, H, W = first_sample['img'].shape
        result['imgs'] = torch.zeros(batch_size, max_length, C, H, W)

    if has_depth:
        _, dH, dW = first_sample['depth'].shape
        result['depths'] = torch.zeros(batch_size, max_length, 1, dH, dW)
        result['depth_masks'] = torch.zeros(batch_size, max_length, 1, dH, dW)

    if has_intrins:
        result['intrins'] = torch.zeros(batch_size, max_length, 3, 3)

    if has_pose:
        result['camera_locations'] = torch.zeros(batch_size, max_length, 3)
        result['camera_rotations'] = torch.zeros(batch_size, max_length, 4)
        result['point_locations'] = torch.zeros(batch_size, max_length, 3)

    # Fill tensors
    for b, item in enumerate(batch):
        length = item['length']
        result['mask'][b, :length] = True

        for t, sample in enumerate(item['samples']):
            if has_img:
                result['imgs'][b, t] = sample['img']

            if has_depth and sample.get('depth') is not None:
                result['depths'][b, t] = sample['depth']
                result['depth_masks'][b, t] = sample['depth_mask']

            if has_intrins and sample.get('intrins') is not None:
                result['intrins'][b, t] = sample['intrins']

            if has_pose and 'pose' in sample.get('info', {}):
                pose = sample['info']['pose']
                result['camera_locations'][b, t] = torch.from_numpy(pose['camera_location'])
                result['camera_rotations'][b, t] = torch.from_numpy(pose['camera_rotation_quaternion'])
                result['point_locations'][b, t] = torch.from_numpy(pose['point_location'])

    return result


class SequenceTrainLoader:
    """Train loader for sequence-based training."""

    def __init__(self, args, scene_list, epoch=0):
        self.dataset = ReplicaSequenceDataset(
            args,
            scene_list=scene_list,
            mode='train'
        )

        self.batch_sampler = SequenceBucketBatchSampler(
            self.dataset,
            batch_size=args.batch_size,
            bucket_boundaries=getattr(args, 'bucket_boundaries', None),
            shuffle=True,
            drop_last=True,
            seed=epoch
        )

        self.data = DataLoader(
            self.dataset,
            batch_sampler=self.batch_sampler,
            num_workers=getattr(args, 'num_workers', 4),
            collate_fn=sequence_collate_fn,
            pin_memory=True
        )

    def set_epoch(self, epoch):
        """Update epoch for shuffling."""
        self.batch_sampler.set_epoch(epoch)


class SequenceTestLoader:
    """Test loader for sequence-based evaluation."""

    def __init__(self, args, scene_list):
        self.dataset = ReplicaSequenceDataset(
            args,
            scene_list=scene_list,
            mode='test'
        )

        # No bucketing for test, just sequential
        self.data = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=getattr(args, 'num_workers', 1),
            collate_fn=sequence_collate_fn,
            pin_memory=True
        )
