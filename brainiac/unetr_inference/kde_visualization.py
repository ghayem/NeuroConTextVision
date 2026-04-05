#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from glob import glob
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 🔹 Prevents GUI windows from popping up on Windows
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import plotting

# -------------------------------------------------------------------
# 1) Directories
# -------------------------------------------------------------------
KDE_DIR = "./predictions"          # Input: segmentation outputs
OUTPUT_DIR = "./visualizations"    # New folder for saved plots
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------------
# 2) Z-score normalize helper (keeps affine)
# -------------------------------------------------------------------
def normalize_img(img):
    """Z-score normalizes a Nifti image. Accepts path or Nifti object."""
    if isinstance(img, (str, Path)):
        img = nib.load(img)
    data = img.get_fdata()
    # Avoid division by zero for binary masks
    std = data.std() if data.std() > 1e-8 else 1.0
    data = (data - data.mean()) / std
    return nib.Nifti1Image(data, img.affine)

# -------------------------------------------------------------------
# 3) Collect sample files
# -------------------------------------------------------------------
nii_paths = sorted(glob(os.path.join(KDE_DIR, "pmid_*.nii.gz")))
print(f" Found {len(nii_paths)} NIfTI files in {KDE_DIR}")
if not nii_paths:
    raise RuntimeError(f"No pmid_*.nii.gz files found in {KDE_DIR}")

# -------------------------------------------------------------------
# 4) Plot & Save each map
# -------------------------------------------------------------------
plot_parameters = {
    "cmap": "bwr",
    "vmax": 6,
    "views": ["lateral"],
    "colorbar": True,
    "alpha": 1.0,
    "bg_on_data": True,
}

plt.ioff()  #  Turn off interactive mode for batch processing

for path in nii_paths:
    fname = os.path.basename(path)
    pmid_str = fname.replace("pmid_", "").replace(".nii.gz", "")
    img_norm = normalize_img(path)

    print(f" Plotting & saving {fname}...")
    
    #  Version-proof: nilearn returns either a Figure or a (fig, axes) tuple
    result = plotting.plot_img_on_surf(img_norm, title=f"PMID: {pmid_str}", **plot_parameters)
    fig = result[0] if isinstance(result, tuple) else result

    save_path = os.path.join(OUTPUT_DIR, f"{fname.replace('.nii.gz', '')}.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Free memory

print(f"\n Successfully saved {len(nii_paths)} plots to: {OUTPUT_DIR}/")