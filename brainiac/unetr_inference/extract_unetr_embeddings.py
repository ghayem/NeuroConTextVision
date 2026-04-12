#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import torch
import numpy as np
from monai.networks.nets import UNETR
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    NormalizeIntensityd, EnsureTyped, Resized
)

# ==========================================
#  ROBUST PATH RESOLUTION (Works from ANY folder)
# ==========================================
CHECKPOINT_PATH = "checkpoints/segmentation.ckpt"
INPUT_DIR = "kde_data/kde_samples/kde_processed/"  #  Folder containing input .nii.gz files
OUTPUT_DIR = "unetr_embeddings/"  #  Folder where results will be saved
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================
#  LOAD ENCODER
# ==========================================
def load_encoder():
    print(f" Loading checkpoint from: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    
    # Clean prefixes
    state_dict = {k.replace("model.", "").replace("unetr.", ""): v 
                  for k, v in ckpt["state_dict"].items()}
    
    # Initialize UNETR (out_channels=1 matches your binary checkpoint)
    model = UNETR(
        in_channels=1, out_channels=1, img_size=(96, 96, 96),
        feature_size=16, hidden_size=768, spatial_dims=3
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE).eval()
    
    return model.vit  # Return only the ViT encoder

# ==========================================
#  SAFE EMBEDDING EXTRACTION
# ==========================================
def extract_embedding(encoder, x):
    with torch.no_grad():
        out = encoder(x)
        
        #  FIX: MONAI ViT often returns a list of tensors (hidden states per block)
        if isinstance(out, (list, tuple)):
            feat = out[-1]  # Grab the deepest layer
        else:
            feat = out
            
        # Double-check for nested lists (edge case in some MONAI versions)
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
            
        if not isinstance(feat, torch.Tensor):
            raise ValueError(f"Expected tensor, got {type(feat)}")
            
        #  Remove CLS token if present (seq_len = 217: 216 patches + 1 CLS)
        if feat.dim() == 3 and feat.shape[1] == 217:
            feat = feat[:, 1:, :]
            
        #  Global Average Pool over patches -> [B, 768]
        emb = feat.mean(dim=1)
        
        return emb.squeeze(0).cpu().numpy()

# ==========================================
# MAIN LOOP
# ==========================================
def run_extraction():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not os.path.exists(INPUT_DIR):
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")
        
    encoder = load_encoder()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        NormalizeIntensityd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(96, 96, 96)),
        EnsureTyped(keys=["image"]),
    ])

    input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "pmid_*.nii.gz")))
    if not input_files:
        print(f"No pmid_*.nii.gz files found in {INPUT_DIR}")
        return

    print(f"Found {len(input_files)} files. Extracting encoder embeddings...\n")

    for i, path in enumerate(input_files, 1):
        fname = os.path.basename(path)
        pmid = fname.replace("pmid_", "").replace(".nii.gz", "")
        out_path = os.path.join(OUTPUT_DIR, f"pmid_{pmid}.npy")

        if os.path.exists(out_path):
            print(f"[{i}/{len(input_files)}] ⏭️  Skipping {pmid} (exists)")
            continue

        print(f"[{i}/{len(input_files)}] Processing {pmid}...", end=" ")
        try:
            data = transforms({"image": path})
            x = data["image"].unsqueeze(0).to(DEVICE)
            
            embedding = extract_embedding(encoder, x)
            np.save(out_path, embedding)
            print(f"Saved ({embedding.shape})")
        except Exception as e:
            print(f"Failed: {e}")

    print(f"\nExtraction complete! Embeddings saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_extraction()