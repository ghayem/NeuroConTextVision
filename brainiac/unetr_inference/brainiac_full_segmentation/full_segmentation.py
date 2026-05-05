import torch
import nibabel as nib
import os
import glob
import numpy as np
from monai.networks.nets import UNETR
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    NormalizeIntensityd, EnsureTyped
)
from monai.inferers import SlidingWindowInferer

# ==========================================
# CONFIGURATION
# ==========================================
#
# Update relative paths in your scripts !!!
#
CHECKPOINT_PATH = "checkpoints/segmentation.ckpt"
INPUT_DIR = "kde_data/kde_samples/kde_processed/"  #  Folder containing input .nii.gz files
OUTPUT_DIR = "predictions/"                        #  Folder where results will be saved
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# LOAD MODEL 
# ==========================================
def load_model():
    print(" Loading checkpoint...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state_dict = {k.replace("model.", "").replace("unetr.", ""): v for k, v in ckpt["state_dict"].items()}
    out_channels = state_dict["out.conv.conv.weight"].shape[0]
    
    model = UNETR(in_channels=1, out_channels=out_channels, img_size=(96,96,96), 
                  feature_size=16, hidden_size=768, spatial_dims=3)
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE).eval()
    print(f"✅ Model loaded | Device: {DEVICE} | Output channels: {out_channels}")
    return model, out_channels

# ==========================================
#  PROCESS SINGLE FILE
# ==========================================
def process_file(model, input_path, output_path, out_channels):
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        NormalizeIntensityd(keys=["image"]),
        EnsureTyped(keys=["image"]),
    ])
    data = transforms({"image": input_path})
    input_tensor = data["image"].unsqueeze(0).to(DEVICE)
    
    inferer = SlidingWindowInferer(roi_size=(96,96,96), sw_batch_size=2, overlap=0.25, mode="gaussian")
    
    with torch.no_grad():
        logits = inferer(input_tensor, model)
        
    if out_channels == 1:
        pred = (torch.sigmoid(logits) > 0.5).float()
    else:
        pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        
    pred_np = pred.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
    orig_img = nib.load(input_path)
    nib.save(nib.Nifti1Image(pred_np, orig_img.affine, orig_img.header), output_path)

# ==========================================
#  MAIN BATCH LOOP
# ==========================================
def run_batch_inference():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model, out_channels = load_model()
    
    # Find all .nii.gz files in INPUT_DIR
    input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.nii.gz")))
    if not input_files:
        print(f" No .nii.gz files found in {INPUT_DIR}")
        return
        
    print(f" Found {len(input_files)} files. Starting inference...\n")
    
    for i, input_path in enumerate(input_files, 1):
        filename = os.path.basename(input_path)          # e.g., "pmid_26629933.nii.gz"
        name_no_ext = os.path.splitext(filename)[0]      # e.g., "pmid_26629933"
        

        
        output_filename = f"{name_no_ext}.gz"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        print(f"[{i}/{len(input_files)}] Processing {filename}...", end=" ")
        try:
            process_file(model, input_path, output_path, out_channels)
            print(" Done")
        except Exception as e:
            print(f" Failed: {e}")
            
    print("\n All files processed! Results saved in:", OUTPUT_DIR)

if __name__ == "__main__":
    run_batch_inference()
