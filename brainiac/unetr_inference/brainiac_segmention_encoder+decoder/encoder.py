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

CHECKPOINT_PATH = "checkpoints/segmentation.ckpt"
INPUT_DIR = "kde_data/kde_samples/kde_processed/"
OUTPUT_DIR = "unetr_embeddings/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    print(f" Loading checkpoint from: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state_dict = {k.replace("model.", "").replace("unetr.", ""): v 
                  for k, v in ckpt["state_dict"].items()}
    
    model = UNETR(
        in_channels=1, out_channels=1, img_size=(96, 96, 96),
        feature_size=16, hidden_size=768, spatial_dims=3
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE).eval()
    return model

def run_extraction():
    if not os.path.exists(CHECKPOINT_PATH) or not os.path.exists(INPUT_DIR):
        raise FileNotFoundError("Checkpoint or Input directory not found.")
        
    model = load_model()
    vit = model.vit
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #  Register hooks for layers 3, 6, 9, 12 (indices 2, 5, 8, 11)
    captured = {}
    def make_hook(idx):
        def hook(module, input, output):
            captured[idx] = output.detach()
        return hook

    TARGET_IDX = [2, 5, 8, 11]
    hooks = [vit.blocks[i].register_forward_hook(make_hook(i)) for i in TARGET_IDX]

    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        NormalizeIntensityd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(96, 96, 96)),
        EnsureTyped(keys=["image"]),
    ])

    input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "pmid_*.nii.gz")))
    print(f" Found {len(input_files)} files. Extracting features via hooks...\n")

    for i, path in enumerate(input_files, 1):
        fname = os.path.basename(path)
        pmid = fname.replace("pmid_", "").replace(".nii.gz", "")
        pmid_dir = os.path.join(OUTPUT_DIR, f"pmid_{pmid}")
        os.makedirs(pmid_dir, exist_ok=True)

        if all(os.path.exists(os.path.join(pmid_dir, f"vit_block_{s}.npy")) for s in range(4)) and \
           os.path.exists(os.path.join(pmid_dir, "input.npy")):
            print(f"[{i}/{len(input_files)}] ⏭️  Skipping {pmid} (exists)")
            continue

        print(f"[{i}/{len(input_files)}] Processing {pmid}...", end=" ")
        captured.clear()  # Reset for new image
        try:
            data = transforms({"image": path})
            x = data["image"].unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                _ = vit(x)  # Triggers hooks automatically
                
            for s, idx in enumerate(TARGET_IDX):
                feat = captured.get(idx)
                if feat is None:
                    raise RuntimeError(f"Hook {idx} failed to capture.")
                
                # Safely unwrap if MONAI returns lists/tuples instead of tensors
                while isinstance(feat, (list, tuple)):
                    feat = feat[-1]  # Grab the actual tensor
                    
                if not isinstance(feat, torch.Tensor):
                    raise ValueError(f"Expected tensor, got {type(feat)}")
                    
                np.save(os.path.join(pmid_dir, f"vit_block_{s}.npy"), feat.squeeze(0).cpu().numpy())
                
            np.save(os.path.join(pmid_dir, "input.npy"), data["image"].numpy())
            print(" Saved 4 blocks + input")
        except Exception as e:
            print(f" Failed: {e}")

    # Cleanup hooks
    for h in hooks: h.remove()
    print(f"\n Extraction complete! Features saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_extraction()
