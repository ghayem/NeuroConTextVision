import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
from glob import glob
import os
import torch.nn as nn
from monai.networks.nets import ViT
import pandas as pd

# -----------------------
# ViT Backbone
# -----------------------
class ViTBackboneNet(nn.Module):
    def __init__(self, simclr_ckpt_path):
        super().__init__()
        self.backbone = ViT(
            in_channels=1,
            img_size=(96,96,96),
            patch_size=(16,16,16),
            hidden_size=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            save_attn=True
        )
        
        # Load pretrained weights
        ckpt = torch.load(simclr_ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        backbone_state_dict = {k[9:]: v for k,v in state_dict.items() if k.startswith("backbone.")}
        self.backbone.load_state_dict(backbone_state_dict, strict=True)
        print("Backbone weights loaded!!")

    def forward(self, x):
        features = self.backbone(x)
        cls_token = features[0][:,0]
        return cls_token

# -----------------------
# Classifier (for regression)
# -----------------------
class Classifier(nn.Module):
    def __init__(self, d_model=768, num_classes=1):
        super().__init__()
        self.fc = nn.Linear(d_model, num_classes)
    def forward(self, x):
        return self.fc(x)

# -----------------------
# Preprocessing
# -----------------------
def preprocess_nifti(path):
    img = nib.load(path)
    data = img.get_fdata()
    data = (data - data.mean()) / (data.std() + 1e-8)
    tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    tensor = F.interpolate(tensor.unsqueeze(0), size=(96,96,96), mode="trilinear", align_corners=False).squeeze(0)
    return tensor  # shape (1,96,96,96)

# -----------------------
# Load backbone
# -----------------------
backbone = ViTBackboneNet("./checkpoints/BrainIAC.ckpt")
backbone.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone.to(device)

# -----------------------
# Extraction
# -----------------------
KDE_DIR = "../data/kde_samples"
nii_paths = sorted(glob(os.path.join(KDE_DIR, "pmid_*.nii.gz")))

print("="*100)
print("BRAINIAC EMBEDDING EXTRACTION")
print(f"Processing directory: {KDE_DIR}")
print(f"Found {len(nii_paths)} NIfTI files")
print("="*100)

# -----------------------
# Directories for saving
# -----------------------
EMBED_DIR = "embeddings"
NPY_DIR = os.path.join(EMBED_DIR, "npy")
CSV_DIR = os.path.join(EMBED_DIR, "csv")
os.makedirs(NPY_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# -----------------------
# Extract and save progressively
# -----------------------
for idx, path in enumerate(nii_paths, 1):
    filename = os.path.basename(path)
    pmid = filename.split('_')[1].split('.')[0]
    
    tensor = preprocess_nifti(path).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = backbone(tensor).cpu()
    
    emb_np = embedding.numpy().flatten()
    
    # Save .npy
    npy_path = os.path.join(NPY_DIR, f"pmid_{pmid}_embedding.npy")
    np.save(npy_path, emb_np)
    
    # Save .csv
    csv_path = os.path.join(CSV_DIR, f"pmid_{pmid}_embedding.csv")
    pd.DataFrame([emb_np], columns=[f'feature_{i}' for i in range(len(emb_np))]).to_csv(csv_path, index=False)
    
    print(f"[{idx}/{len(nii_paths)}] Processed: {filename} → PMIDs saved: {npy_path} & {csv_path}")

print("="*100)
print("All embeddings saved:")
print(f"  - NPY files: {NPY_DIR}/")
print(f"  - CSV files: {CSV_DIR}/")
print("="*100)