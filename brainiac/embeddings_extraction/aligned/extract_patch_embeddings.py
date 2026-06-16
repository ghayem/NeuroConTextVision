import argparse
import pickle
import sys
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    NormalizeIntensityd, EnsureTyped, Resized
)

# Import the specific segmentation model class
from segmentation_model import ViTUNETRSegmentationModel

# ---------------------------------------------------------------------
# Helper: Dual Checkpoint Model Loading
# ---------------------------------------------------------------------

def load_extraction_model(segmentation_ckpt, brainiac_ckpt, device):
    """
    Loads the ViT backbone from a UNETR segmentation structure.
    """
    ckpt = torch.load(segmentation_ckpt, map_location="cpu")
    config = ckpt["hyper_parameters"]
    state_dict = ckpt["state_dict"]

    model = ViTUNETRSegmentationModel(
        simclr_ckpt_path=brainiac_ckpt,
        img_size=tuple(config["model"]["img_size"]),
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"]
    )

    new_state_dict = { (k[6:] if k.startswith("model.") else k): v
                      for k, v in state_dict.items() }
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device).eval()
    return model.unetr.vit

# ---------------------------------------------------------------------
# Extraction with Disk-Saving Logic
# ---------------------------------------------------------------------

def extract_and_save_split(pmids, text_embeddings, nii_dir, vit, transform, device, output_subdir):
    """
    Processes scans one-by-one and saves individual pkl files to free RAM.
    """
    output_subdir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i, pmid in enumerate(tqdm(pmids, desc=f"Saving to {output_subdir.name}")):
            nii_path = Path(nii_dir) / f"pmid_{pmid}.nii.gz"

            if not nii_path.exists():
                continue

            try:
                # 1. Preprocess
                data = transform({"image": str(nii_path)})
                x = data["image"].unsqueeze(0).to(device)

                # 2. Extract [216, 768]
                tokens = vit.patch_embedding(x)
                if hasattr(vit, "pos_embed"):
                    tokens = tokens + vit.pos_embed
                for blk in vit.blocks:
                    tokens = blk(tokens)
                if hasattr(vit, "norm"):
                    tokens = vit.norm(tokens)

                patch_matrix = tokens.squeeze(0).cpu().numpy() # [216, 768]
                text_vec = text_embeddings[i] # Aligned text vector

                # 3. Save individual pair to disk
                save_path = output_subdir / f"pmid_{pmid}.pkl"
                with open(save_path, "wb") as f:
                    pickle.dump({
                        "pmid": pmid,
                        "brain_patches": patch_matrix,
                        "text_embedding": text_vec
                    }, f)

            except Exception as e:
                print(f"  [FAILED] pmid_{pmid} -> {e}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nii_dir", required=True)
    parser.add_argument("--seg_checkpoint", required=True)
    parser.add_argument("--brainiac_checkpoint", required=True)
    parser.add_argument("--train_pmids", required=True)
    parser.add_argument("--test_pmids", required=True)
    parser.add_argument("--train_text_pkl", required=True)
    parser.add_argument("--test_text_pkl", required=True)
    parser.add_argument("--output_dir", default="./aligned_dataset_disk")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    vit_backbone = load_extraction_model(args.seg_checkpoint, args.brainiac_checkpoint, device)

    # Transforms
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        Resized(keys=["image"], spatial_size=(96, 96, 96)),
        EnsureTyped(keys=["image"]),
    ])

    # Data Loading
    with open(args.train_text_pkl, "rb") as f: train_text = pickle.load(f)
    with open(args.test_text_pkl, "rb") as f: test_text = pickle.load(f)
    train_pmids = [l.strip() for l in open(args.train_pmids) if l.strip()]
    test_pmids = [l.strip() for l in open(args.test_pmids) if l.strip()]

    root_out = Path(args.output_dir)

    print("\n=== Processing TRAIN (Individual Files) ===")
    extract_and_save_split(train_pmids, train_text, args.nii_dir, vit_backbone, transforms, device, root_out / "train")

    print("\n=== Processing TEST (Individual Files) ===")
    extract_and_save_split(test_pmids, test_text, args.nii_dir, vit_backbone, transforms, device, root_out / "test")

    print(f"\n✅ Extraction complete. Data saved to {root_out}")

if __name__ == "__main__":
    main()
