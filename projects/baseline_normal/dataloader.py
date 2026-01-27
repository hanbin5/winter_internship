import os
import random

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from projects import PROJECT_DIR

class NormalDataset(Dataset):
    def __init__(self, args, dataset_name='nyuv2', split='test', mode='test', epoch=0):
        self.args = args
        self.split = split
        self.mode = mode
        self.batch_size = args.batch_size
        assert mode in ['train', 'test']

        # data split
        split_path = os.path.join(PROJECT_DIR, 'data', 'datasets', dataset_name, 'split', split+'.txt')
        assert os.path.exists(split_path)
        with open(split_path, 'r') as f:
            self.filenames = [i.strip() for i in f.readlines()]

        if dataset_name == 'nyuv2':
            from data.datasets.nyuv2 import get_sample
        elif dataset_name == 'scannet':
            from data.datasets.scannet import get_sample
        elif dataset_name == 'ibims':
            from data.datasets.ibims import get_sample
        elif dataset_name == 'sintel':
            from data.datasets.sintel import get_sample
        elif dataset_name == 'vkitti':
            from data.datasets.vkitti import get_sample
        elif dataset_name == 'oasis':
            from data.datasets.oasis import get_sample
        self.get_sample = get_sample

        if self.mode == 'train':
            random.seed(epoch)
            random.shuffle(self.filenames)
            num_batches = len(self.filenames) // (args.batch_size * args.accumulate_grad_batches)
            num_imgs = num_batches * args.batch_size * args.accumulate_grad_batches
            self.filenames = self.filenames[:num_imgs]

    def __len__(self):
        return len(self.filenames)
    
class TrainLoader:
    def __init__(self, args, epoch=0):
        self.train_samples = NormalDataset(
            args, 
            dataset_name=args.dataset_name_train, 
            split=args.train_split, 
            mode='train', 
            epoch=epoch
        )

        self.data = DataLoader(
            self.train_samples, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers, 
            pin_memory=True, 
            drop_last=True
        )

class ValLoader:
    def __init__(self, args):
        self.val_samples = NormalDataset(
            args, 
            dataset_name=args.dataset_name_val, 
            split=args.val_split,
            mode='test',
            epoch=None
        )
        
        self.data = DataLoader(
            self.train_samples, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers, 
            pin_memory=True
        )

