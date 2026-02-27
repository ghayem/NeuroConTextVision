import os
from glob import glob

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer

# -----------------------
# PATHS
# -----------------------
KDE_DIR = "../data/kde_samples"
EMBED_DIR = "embeddings"

NPY_DIR = os.path.join(EMBED_DIR, "npy")
CSV_DIR = os.path.join(EMBED_DIR, "csv")
os.makedirs(NPY_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# LOAD SEGVol MODEL
# -----------------------
MODEL_NAME = "BAAI/SegVol"

print("Loading SegVol model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    test_mode=True
)

model.model.text_encoder.tokenizer = tokenizer
model.eval().to(DEVICE)
print("SegVol loaded successfully.")

# -----------------------
# EXTRACTION FUNCTIONS
# -----------------------
def load_nifti(path):
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    data = (data - data.mean()) / (data.std() + 1e-8)  # Z-score
    return data

def resize_volume(volume):
    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(
        tensor,
        size=(128, 128, 128),
        mode="trilinear",
        align_corners=False
    )
    return tensor.to(DEVICE)

def extract_embedding(tensor):
    with torch.no_grad():
        features = model.model.image_encoder(tensor)
        if isinstance(features, tuple):
            features = features[0]
        embedding = features.mean(dim=1)
    return embedding.squeeze().cpu().numpy()

# -----------------------
# LOAD KDE FILES
# -----------------------
paths = sorted(glob(os.path.join(KDE_DIR, "pmid_*.nii.gz")))
if len(paths) == 0:
    raise RuntimeError("No KDE volumes found.")

print("Found", len(paths), "KDE volumes")

# -----------------------
# EXTRACTION LOOP (PROGRESSIVE)
# -----------------------
for idx, p in enumerate(tqdm(paths), 1):
    fname = os.path.basename(p)
    pmid = fname.replace("pmid_", "").replace(".nii.gz", "")
    
    # Skip if already saved (allows progressive extraction)
    npy_path = os.path.join(NPY_DIR, f"pmid_{pmid}_embedding.npy")
    csv_path = os.path.join(CSV_DIR, f"pmid_{pmid}_embedding.csv")
    if os.path.exists(npy_path) and os.path.exists(csv_path):
        continue
    
    volume = load_nifti(p)
    tensor = resize_volume(volume)
    vec = extract_embedding(tensor)
    
    # Save .npy
    np.save(npy_path, vec)
    
    # Save .csv
    df = pd.DataFrame([vec], columns=[f"feat_{i}" for i in range(len(vec))])
    df.insert(0, "pmid", [pmid])
    df.to_csv(csv_path, index=False)
    
    print(f"[{idx}/{len(paths)}] Saved: {npy_path} & {csv_path}")

print("="*100)
print("All SegVol embeddings saved:")
print(f"  - NPY files: {NPY_DIR}/")
print(f"  - CSV files: {CSV_DIR}/")
print("="*100)