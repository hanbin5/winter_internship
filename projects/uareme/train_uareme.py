"""
U-ARE-ME Training Script
Uncertainty-Aware Rotation Estimation in Manhattan Environments

This script trains the DSINE (Dense Surface Normal Estimation) backbone
which predicts per-pixel surface normals and uncertainty (kappa) from RGB images.

Author's Note: The key insight is that surface normals predicted by neural networks
are unreliable near object boundaries and on small objects. We learn to predict
this uncertainty (kappa) so that unreliable predictions can be down-weighted
during rotation optimization.
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.dsine.v00 import DSINE_v00


class SurfaceNormalDataset(Dataset):
    """
    Dataset for surface normal estimation training.

    Expected data structure:
    - data_root/
        - rgb/          # RGB images
        - normal/       # Ground truth surface normal maps (in camera coordinates)
        - mask/         # Valid pixel masks (optional)

    Surface normals are stored as (H, W, 3) images where each pixel
    contains a unit vector [nx, ny, nz] pointing outward from the surface.
    """
    def __init__(self, data_root, split='train', img_size=(480, 640)):
        self.data_root = data_root
        self.split = split
        self.img_size = img_size  # (H, W)

        # Load file lists
        rgb_dir = os.path.join(data_root, split, 'rgb')
        self.samples = []

        if os.path.exists(rgb_dir):
            for fname in sorted(os.listdir(rgb_dir)):
                if fname.endswith(('.png', '.jpg', '.jpeg')):
                    rgb_path = os.path.join(rgb_dir, fname)
                    normal_path = os.path.join(data_root, split, 'normal', fname.replace('.jpg', '.png').replace('.jpeg', '.png'))
                    mask_path = os.path.join(data_root, split, 'mask', fname.replace('.jpg', '.png').replace('.jpeg', '.png'))

                    if os.path.exists(normal_path):
                        self.samples.append({
                            'rgb': rgb_path,
                            'normal': normal_path,
                            'mask': mask_path if os.path.exists(mask_path) else None
                        })

        # ImageNet normalization (used by EfficientNet backbone)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load RGB image
        rgb = Image.open(sample['rgb']).convert('RGB')
        rgb = rgb.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        rgb = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float() / 255.0
        rgb = self.normalize(rgb)

        # Load ground truth surface normals
        normal = Image.open(sample['normal'])
        normal = normal.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        normal = torch.from_numpy(np.array(normal)).float()

        # Convert from [0, 255] to [-1, 1] range
        if normal.max() > 1:
            normal = normal / 127.5 - 1.0

        # Ensure unit vectors
        if len(normal.shape) == 3:
            normal = normal.permute(2, 0, 1)  # (3, H, W)
        normal = F.normalize(normal, p=2, dim=0)

        # Load mask if available
        if sample['mask'] is not None:
            mask = Image.open(sample['mask']).convert('L')
            mask = mask.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
            mask = torch.from_numpy(np.array(mask)).float() / 255.0
        else:
            mask = torch.ones(self.img_size[0], self.img_size[1])

        return {
            'rgb': rgb,           # (3, H, W)
            'normal': normal,     # (3, H, W)
            'mask': mask          # (H, W)
        }


class AngularLoss(nn.Module):
    """
    Angular loss for surface normal estimation.

    The core idea: we measure the angle between predicted and ground truth normals.
    Smaller angle = better prediction.

    L_angular = arccos(pred · gt) / π

    This is more geometrically meaningful than L1/L2 loss because normals
    are unit vectors on a sphere.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask=None):
        # pred, target: (B, 3, H, W)
        # mask: (B, H, W)

        # Normalize predictions (should already be normalized, but ensure it)
        pred = F.normalize(pred, p=2, dim=1)
        target = F.normalize(target, p=2, dim=1)

        # Compute dot product
        dot = (pred * target).sum(dim=1)  # (B, H, W)
        dot = torch.clamp(dot, -1.0, 1.0)  # Numerical stability

        # Angular error in radians
        angle_error = torch.acos(dot)  # (B, H, W)

        if mask is not None:
            angle_error = angle_error * mask
            loss = angle_error.sum() / (mask.sum() + 1e-8)
        else:
            loss = angle_error.mean()

        return loss


class UncertaintyLoss(nn.Module):
    """
    Uncertainty-aware loss with heteroscedastic aleatoric uncertainty.

    The key insight (from the paper):
    - Surface normals near object boundaries and on small objects are unreliable
    - We predict kappa (concentration parameter of von Mises-Fisher distribution)
    - Higher kappa = more confident prediction

    Loss = angular_error / kappa + log(kappa)

    This formulation encourages:
    - Low kappa (low confidence) for hard-to-predict regions
    - High kappa (high confidence) for easy regions
    - The log(kappa) term prevents kappa from going to infinity
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_normal, pred_kappa, target_normal, mask=None):
        # pred_normal: (B, 3, H, W)
        # pred_kappa: (B, 1, H, W)
        # target_normal: (B, 3, H, W)

        pred_normal = F.normalize(pred_normal, p=2, dim=1)
        target_normal = F.normalize(target_normal, p=2, dim=1)

        dot = (pred_normal * target_normal).sum(dim=1, keepdim=True)
        dot = torch.clamp(dot, -1.0, 1.0)
        angle_error = torch.acos(dot)  # (B, 1, H, W)

        # Ensure kappa > 0
        kappa = F.softplus(pred_kappa) + 1e-8

        # Uncertainty-weighted loss
        loss = angle_error / kappa + torch.log(kappa)

        if mask is not None:
            mask = mask.unsqueeze(1)
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()

        return loss


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        rgb = batch['rgb'].to(device)
        target_normal = batch['normal'].to(device)
        mask = batch['mask'].to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model(rgb)[0]  # (B, 4, H, W)
        pred_normal = output[:, :3, :, :]
        pred_kappa = output[:, 3:, :, :]

        # Compute loss
        loss = criterion(pred_normal, pred_kappa, target_normal, mask)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_angular_error = 0

    for batch in dataloader:
        rgb = batch['rgb'].to(device)
        target_normal = batch['normal'].to(device)
        mask = batch['mask'].to(device)

        output = model(rgb)[0]
        pred_normal = output[:, :3, :, :]
        pred_kappa = output[:, 3:, :, :]

        loss = criterion(pred_normal, pred_kappa, target_normal, mask)
        total_loss += loss.item()

        # Compute angular error in degrees
        pred_normal = F.normalize(pred_normal, p=2, dim=1)
        dot = (pred_normal * target_normal).sum(dim=1)
        dot = torch.clamp(dot, -1.0, 1.0)
        angle_error = torch.acos(dot) * 180 / np.pi  # Convert to degrees

        if mask is not None:
            mean_angle = (angle_error * mask).sum() / (mask.sum() + 1e-8)
        else:
            mean_angle = angle_error.mean()
        total_angular_error += mean_angle.item()

    n_batches = len(dataloader)
    return total_loss / n_batches, total_angular_error / n_batches


def main():
    parser = argparse.ArgumentParser(description='Train U-ARE-ME Surface Normal Estimation')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to dataset root directory')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--img_height', type=int, default=480)
    parser.add_argument('--img_width', type=int, default=640)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    os.makedirs(args.output_dir, exist_ok=True)

    # Model
    model = DSINE_v00().to(device)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        print(f'Resumed from {args.resume}')

    # Dataset
    train_dataset = SurfaceNormalDataset(
        args.data_root, split='train',
        img_size=(args.img_height, args.img_width)
    )
    val_dataset = SurfaceNormalDataset(
        args.data_root, split='val',
        img_size=(args.img_height, args.img_width)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')

    # Loss and optimizer
    criterion = UncertaintyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_angle = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f'Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_angle={val_angle:.2f}°')

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print(f'  -> Saved best model (val_loss={val_loss:.4f})')

        # Save periodic checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.output_dir, f'checkpoint_epoch{epoch}.pt'))


if __name__ == '__main__':
    main()
