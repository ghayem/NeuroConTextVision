import os
import nibabel as nib
import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
KDE_FILE = "../data/KDE_samples/pmid_21273134.nii.gz"
OUT_FILE = "./segmentation_results/output_seg.nii.gz"

def normalize_img(img_path):
    img = nib.load(img_path)
    data = img.get_fdata()
    std = data.std()
    if std == 0: return img
    data = (data - data.mean()) / (std + 1e-8)
    return nib.Nifti1Image(data, img.affine)

def compare_visual(kde_path, out_path):
    kde_norm = normalize_img(kde_path)
    out_img = nib.load(out_path)

    params = {
        "surf_mesh": "fsaverage",
        "views": ["lateral"],
        "bg_on_data": True,
        "darkness": 0.8,  
        "colorbar": True
    }

    print("Affichage de la Vérité Terrain (KDE)...")
    plotting.plot_img_on_surf(
        kde_norm, 
        title="GROUNDTRUTH (KDE) pmid_21273134",
        cmap="bwr", 
        vmax=6, 
        threshold=1.5, 
        **params
    )

    print("Affichage de la Prédiction (Output)...")
    plotting.plot_img_on_surf(
        out_img, 
        title="Segmentation (BrainIAC) pmid_21273134",
        cmap="hot", 
        vmax=1, 
        threshold=0.1, 
        **params
    )

    plotting.show()

if __name__ == "__main__":
    compare_visual(KDE_FILE, OUT_FILE)