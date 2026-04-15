#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import torch
import numpy as np
import nibabel as nib
from monai.networks.nets import UNETR
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped

# ==========================================
#  CONFIGURATION
# ==========================================
CHECKPOINT_PATH = "checkpoints/segmentation.ckpt"
FEATURES_DIR = "unetr_embeddings/"
OUTPUT_DIR = "decoder_predictions/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_GRID = 6  # 96 / 16 = 6 patches per dimension

# ==========================================
#  LOAD MODEL
# ==========================================
def load_model():
    print(f" Loading checkpoint from: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state_dict = {k.replace("model.", "").replace("unetr.", ""): v 
                  for k, v in ckpt["state_dict"].items()}
    out_channels = state_dict["out.conv.conv.weight"].shape[0]
    
    model = UNETR(
        in_channels=1, out_channels=out_channels, img_size=(96, 96, 96),
        feature_size=16, hidden_size=768, spatial_dims=3
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE).eval()
    return model, out_channels

# ==========================================
#  HELPER: RESHAPE [B, N, C] -> [B, C, D, H, W]
# ==========================================
def reshape_to_3d(feat):
    B, N, C = feat.shape
    D = PATCH_GRID  # 6
    return feat.view(B, D, D, D, C).permute(0, 4, 1, 2, 3).contiguous()

# ==========================================
#  DECODER INFERENCE LOOP
# ==========================================
def run_decoder():
    if not os.path.exists(FEATURES_DIR):
        raise FileNotFoundError(f"Features directory not found: {FEATURES_DIR}")
        
    model, out_channels = load_model()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    load_transform = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        EnsureTyped(keys=["image"]),
    ])

    pmid_dirs = sorted(glob.glob(os.path.join(FEATURES_DIR, "pmid_*")))
    if not pmid_dirs:
        print(" No feature folders found. Run extract_unetr_embeddings.py first.")
        return

    print(f" Found {len(pmid_dirs)} feature sets. Running decoder...\n")

    for i, feat_dir in enumerate(pmid_dirs, 1):
        pmid = os.path.basename(feat_dir).replace("pmid_", "")
        out_path = os.path.join(OUTPUT_DIR, f"pmid_{pmid}.nii.gz")
        
        if os.path.exists(out_path):
            print(f"[{i}/{len(pmid_dirs)}] ⏭️  Skipping {pmid} (exists)")
            continue

        print(f"[{i}/{len(pmid_dirs)}] Decoding {pmid}...", end=" ")
        try:
            #  Load 4 raw ViT blocks [1, 216, 768]
            features = []
            for s in range(4):
                feat_np = np.load(os.path.join(feat_dir, f"vit_block_{s}.npy"))
                features.append(torch.from_numpy(feat_np).unsqueeze(0).to(DEVICE))
                
            #  Load preprocessed input image
            input_np = np.load(os.path.join(feat_dir, "input.npy"))
            input_tensor = torch.from_numpy(input_np).unsqueeze(0).to(DEVICE)
            
            #  CORRECT MONAI UNETR MAPPING (Matches your checkpoint exactly):
            # encoder1 -> input image (1 ch -> 16 ch)
            # encoder2 -> ViT feature 0 (layer 3)  (768 ch -> 32 ch)
            # encoder3 -> ViT feature 1 (layer 6)  (768 ch -> 64 ch)
            # encoder4 -> ViT feature 2 (layer 9)  (768 ch -> 128 ch)
            # BOTTLENECK -> ViT feature 3 (layer 12) reshaped to [1, 768, 6, 6, 6]
            
            enc1_out = model.encoder1(input_tensor)
            enc2_out = model.encoder2(reshape_to_3d(features[0]))
            enc3_out = model.encoder3(reshape_to_3d(features[1]))
            enc4_out = model.encoder4(reshape_to_3d(features[2]))
            bottleneck = reshape_to_3d(features[3])  # No encoder5 in this version
            
            #  CHAIN DECODERS (Skip connections match U-Net topology)
            dec5_out = model.decoder5(bottleneck, enc4_out)  # 6³ -> 12³
            dec4_out = model.decoder4(dec5_out, enc3_out)    # 12³ -> 24³
            dec3_out = model.decoder3(dec4_out, enc2_out)    # 24³ -> 48³
            dec2_out = model.decoder2(dec3_out, enc1_out)    # 48³ -> 96³
            
            logits = model.out(dec2_out)
            
            #  Threshold to binary mask
            if out_channels == 1:
                pred = (torch.sigmoid(logits) > 0.5).float()
            else:
                pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                
            pred_np = pred.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
            
            #  Preserve original spatial metadata
            orig_path = f"kde_data/kde_samples/kde_processed/pmid_{pmid}.nii.gz"
            orig_nib = nib.load(orig_path)
            nib.save(nib.Nifti1Image(pred_np, orig_nib.affine, orig_nib.header), out_path)
            
            print(" Saved")
        except Exception as e:
            print(f" Failed: {e}")
            import traceback; traceback.print_exc()

    print(f"\n Decoder predictions saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_decoder()