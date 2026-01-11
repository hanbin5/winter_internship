import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from uareme_cls import UAREME

class ICLNUIMDataset(Dataset):
    def __init__(self, image_path, gt_path):
        self.image_path = image_path
        self.gt_path = gt_path
        
        self.gt_rotations = self._load_gt_sim(self.gt_path)
        self.images = sorted([f for f in os.listdir(self.image_path) if f.endswith('.png')],
                             key=lambda x: int(x.split('.')[0]))[1:]

    def _load_gt_sim(self, path):
        with open(path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        matrices = []
        for i in range(0, len(lines), 3):
            rows = [list(map(float, lines[i+j].split())) for j in range(3)]
            mat_3x4 = np.array(rows)
            matrices.append(mat_3x4[:, :3])
        
        return matrices
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_path, self.images[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gt_rot = self.gt_rotations[idx]

        return image, gt_rot.astype(np.float32)
    
def compute_are(R_gt, R_pred):
    if torch.is_tensor(R_gt):
        R_gt = R_gt.detach().cpu().numpy()
    if torch.is_tensor(R_pred):
        R_pred = R_pred.detach().cpu().numpy()
    
    R_diff = np.matmul(R_gt.T, R_pred)
    trace = np.trace(R_diff)

    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)

    return np.degrees(angle_rad)

def evaluate(R_gts, R_preds):
    R_gt0 = R_gts[0]
    R_pred0 = R_preds[0]
    R_align = np.matmul(R_gt0, R_pred0.T)

    ares = []
    for R_g, R_p in zip(R_gts, R_preds):
        R_p_aligned = np.matmul(R_align, R_p)

        are = compute_are(R_g, R_p_aligned)
        
        ares.append(are)
    
    return ares

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    dataset = ICLNUIMDataset(
        "/home/hanbin5/data/ICL-NUIM/living_room_traj0_frei_png/rgb",
        "/home/hanbin5/data/ICL-NUIM/livingRoom0n.gt.sim.txt"
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(len(dataset.images))
    print(len(dataset.gt_rotations))

    uareme = UAREME()

    preds = []
    gts = []
    for image, gt_rot in dataloader:
        img_np = image.squeeze(0).numpy()
        R_pred, _, _ = uareme.run(img_np)

        preds.append(R_pred)
        gts.append(gt_rot.squeeze(0).numpy())
    
    all_ares = evaluate(gts, preds)    

    print("Final Results")
    print(f"Average ARE: {np.mean(all_ares):.4f} degrees")


if __name__ == "__main__":
    main()