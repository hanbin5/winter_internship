import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.spatial.transform import Rotation as R

from models.uareme import UAREME

class ICLNUIMDataset(Dataset):
    def __init__(self, image_path, gt_path):
        self.image_path = image_path
        self.gt_path = gt_path
        
        self.gt_rotations = self._load_gt_freiburg(self.gt_path)
        self.images = sorted([f for f in os.listdir(self.image_path) if f.endswith('.png')],
                             key=lambda x: int(x.split('.')[0]))

    def _load_gt_freiburg(self, path):
        data = np.loadtxt(path)
        matrices = []
        for row in data:
            quat = row[4:8]
            rot_matrix = R.from_quat(quat).as_matrix()
            matrices.append(rot_matrix)
        return matrices
    
    def __len__(self):
        return min(len(self.images), len(self.gt_rotations))
    
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

def evaluate_with_symmetry(R_gts, R_preds):
    def get_24_symmetries():
        # Right-handed 24-fold rotation matrices
        perms = [[0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]]
        signs = [[1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1], [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1]]
        soms = []
        for p in perms:
            for s in signs:
                m = np.zeros((3, 3))
                for i in range(3):
                    m[i, p[i]] = s[i]
                if np.isclose(np.linalg.det(m), 1.0):
                    soms.append(m)
        return soms

    symmetries = get_24_symmetries()
    best_avg_are = float('inf')
    
    for R_sym in symmetries:
        # 핵심: R_preds에 오른쪽에 대칭 행렬을 곱함
        curr_preds = [np.matmul(Rp, R_sym) for Rp in R_preds]
        
        # 0번 프레임 정렬
        R_align = np.matmul(R_gts[0].T, curr_preds[0])
        
        ares = []
        for Rg, Rp in zip(R_gts, curr_preds):
            Rp_aligned = np.matmul(R_align, Rp)
            ares.append(compute_are(Rg, Rp_aligned))
        
        avg_are = np.mean(ares)
        if avg_are < best_avg_are:
            best_avg_are = avg_are
            
    return best_avg_are

def evaluate(R_gts, R_preds):
    R_gt0 = R_gts[0]
    R_pred0 = R_preds[0]
    R_align = np.matmul(R_gt0, R_pred0.T)

    ares = []
    for R_g, R_p in zip(R_gts, R_preds):
        R_p_aligned = np.matmul(R_align, R_p)

        are = compute_are(R_g, R_p)
        
        ares.append(are)
    
    return ares

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    dataset = ICLNUIMDataset(
        "/home/hanbin5/data/UZH-FPV/indoor_forward_3/rgb",
        "/home/hanbin5/data/UZH-FPV/indoor_forward_3/groundtruth.txt"
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(len(dataset.images))
    print(len(dataset.gt_rotations))

    uareme = UAREME()

    preds = []
    gts = []
    for image, gt_rot in dataloader:
        img_np = image.squeeze(0).numpy()
        R_pred, _, _ = uareme(img_np)

        preds.append(R_pred)
        gts.append(gt_rot.squeeze(0).numpy())
    
    all_ares = evaluate_with_symmetry(gts, preds)    

    print("Final Results")
    print(f"Average ARE: {np.mean(all_ares):.4f} degrees")


if __name__ == "__main__":
    main()