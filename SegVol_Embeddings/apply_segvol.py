import os
from glob import glob

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer


 
# PATHS 

KDE_DIR = "../data/KDE_samples/KDE_samples"
OUT_DIR = "embeddings"

MODEL_NAME = "BAAI/SegVol"

os.makedirs(OUT_DIR, exist_ok=True)

NPY_OUTPUT = os.path.join(OUT_DIR, "segvol_embeddings_kde.npy")
CSV_OUTPUT = os.path.join(OUT_DIR, "segvol_embeddings_kde.csv")
PMID_CSV_OUTPUT = os.path.join(OUT_DIR, "segvol_embeddings_pmid.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

 
# LOAD SEGVol MODEL

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

 
# EXTRACT EMBEDDINGS FUNCTIONS

def load_nifti(path):
    """
    Load KDE NIfTI and normalize
    """
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)

    # Z-score normalization
    data = (data - data.mean()) / (data.std() + 1e-8)

    return data


def resize_volume(volume):
    """
    Resize to SegVol size
    """
    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)

    tensor = F.interpolate(
        tensor,
        size=(128, 128, 128),
        mode="trilinear",
        align_corners=False
    )

    return tensor.to(DEVICE)


def extract_embedding(tensor):
    """
    Forward pass through SegVol image encoder
    """
    with torch.no_grad():

        features = model.model.image_encoder(tensor)

        # SegVol sometimes returns tuple
        if isinstance(features, tuple):
            features = features[0]

        embedding = features.mean(dim=1)

    return embedding.squeeze().cpu().numpy()

 
# LOAD KDE FILES

paths = sorted(glob(os.path.join(KDE_DIR, "pmid_*.nii.gz")))

if len(paths) == 0:
    raise RuntimeError("No KDE volumes found.")

print("Found", len(paths), "KDE volumes")


 
# EXTRACTION LOOP
embeddings = []
image_names = []
pmid_list = []

for p in tqdm(paths):

    volume = load_nifti(p)
    tensor = resize_volume(volume)

    vec = extract_embedding(tensor)

    embeddings.append(vec)
    image_names.append(os.path.basename(p))
    fname = os.path.basename(p)
    pmid_list.append(fname.replace("pmid_", "").replace(".nii.gz", ""))

embeddings = np.vstack(embeddings)

print("Embedding shape:", embeddings.shape)


 
# SAVE NUMPY 
np.save(NPY_OUTPUT, embeddings)
print("Saved numpy embeddings.")

# SAVE CSV 
df = pd.DataFrame(embeddings)
df.columns = [f"feat_{i}" for i in range(embeddings.shape[1])]

df.insert(0, "image_path", image_names)

df.to_csv(CSV_OUTPUT, index=False)

print("Saved CSV embeddings.")

# SAVE CSV WITH PMIDs
df_pmid = pd.DataFrame(embeddings)
df_pmid.columns = [f"feat_{i}" for i in range(embeddings.shape[1])]
df_pmid.insert(0, "pmid", pmid_list)
df_pmid.to_csv(PMID_CSV_OUTPUT, index=False)