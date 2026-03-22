#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from glob import glob
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1) Folder with the 5 exported NIfTIs
# -------------------------------------------------------------------
KDE_DIR = "../BrainIAC/src/output/"
# KDE_DIR = "../KDE_images"

# -------------------------------------------------------------------
# 2) Z-score normalize helper (keeps affine)
# -------------------------------------------------------------------
def normalize_img(img):
    """
    Z-score normalizes a Nifti image.
    Accepts either a Nifti object or a path.
    """
    if isinstance(img, (str, Path)):
        img = nib.load(img)
    data = img.get_fdata()
    data = (data - data.mean()) / (data.std() + 1e-8)
    return nib.Nifti1Image(data, img.affine)

# -------------------------------------------------------------------
# 3) Collect the 5 sample files
# -------------------------------------------------------------------
nii_paths = sorted(glob(os.path.join(KDE_DIR, "pmid_*.nii.gz")))
print("Found NIfTI files:", *nii_paths, sep="\n  ")

if len(nii_paths) == 0:
    raise RuntimeError(f"No pmid_*.nii.gz files found in {KDE_DIR}")
elif len(nii_paths) != 5:
    print(f"Warning: expected 5 files, found {len(nii_paths)}. Plotting all of them anyway.")

# -------------------------------------------------------------------
# 4) Plot each KDE map on cortical surface
# -------------------------------------------------------------------
vmax = 6
plot_parameters = {
    "cmap": "bwr",
    "vmax": vmax,
    "views": ["lateral"],
    "colorbar": True,
    "alpha": 1.0,
    "bg_on_data": True,   # set to False if you want only colored activations
    # Optionally hide low |z|:
    # "threshold": 1.5,
}

for path in nii_paths:
    # Extract PMID from filename: pmid_<PMID>.nii.gz
    fname = os.path.basename(path)
    pmid_str = fname.replace("pmid_", "").replace(".nii.gz", "")

    img_norm = normalize_img(path)

    print(f"Plotting {fname} (PMID {pmid_str})")
    plotting.plot_img_on_surf(
        img_norm,
        title=f"PMID: {pmid_str} - KDE Groundtruth",
        **plot_parameters,
    )
    plt.suptitle(f"PMID: {pmid_str} - KDE Groundtruth", fontsize=20)
    plotting.show()
