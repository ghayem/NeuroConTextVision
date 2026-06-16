#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import warnings
from glob import glob
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Prevents GUI windows from popping up
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import image, datasets, surface, plotting
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore", category=FutureWarning, module="nilearn")

# -------------------------------------------------------------------
# 1) Normalization & Metric Helpers
# -------------------------------------------------------------------
def normalize_img_and_binarize(img, threshold=0.95):
    """Normalizes a Nifti image and binarizes at a specified z-score threshold."""
    if isinstance(img, (str, Path)):
        img = nib.load(img)
    data = img.get_fdata()

    std = data.std() if data.std() > 1e-8 else 1.0
    data = (data - data.mean()) / std

    binary_data = (data >= threshold).astype(np.float32)
    return nib.Nifti1Image(binary_data, img.affine), binary_data

def normalize_img(img):
    """Z-score normalizes a Nifti image. Accepts path or Nifti object."""
    if isinstance(img, (str, Path)):
        img = nib.load(img)
    data = img.get_fdata()
    std = data.std() if data.std() > 1e-8 else 1.0
    data = (data - data.mean()) / std
    return nib.Nifti1Image(data, img.affine)

def compute_dice(mask1, mask2):
    """Computes the Dice Similarity Coefficient between two binary masks."""
    intersection = np.sum(mask1 * mask2)
    total_elements = np.sum(mask1) + np.sum(mask2)
    if total_elements == 0:
        return 1.0
    return (2.0 * intersection) / total_elements

# -------------------------------------------------------------------
# 2) Main Pipeline
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate side-by-side surface binarized segmentations with high-visibility metrics."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing both original and paired '_seg.nii.gz' files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where visualization evaluation plots will be saved",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect sample files
    all_files = sorted(glob(os.path.join(args.input_dir, "*.nii.gz")))
    ground_truth_paths = [p for p in all_files if not p.endswith("_seg.nii.gz")]

    print(f" Found {len(ground_truth_paths)} ground truth NIfTI files in {args.input_dir}")
    if not ground_truth_paths:
        raise RuntimeError(f"No valid input base files found in {args.input_dir}")

    plot_parameters = {
        "cmap": "bwr",
        "vmax": 6,
        "views": ["lateral"],
        "colorbar": True,
        "alpha": 1.0,
        "bg_on_data": True,
    }

    plt.ioff()

    for gt_path in ground_truth_paths:
        fname = os.path.basename(gt_path)
        seg_name = fname.replace(".nii.gz", "_seg.nii.gz")
        seg_path = os.path.join(args.input_dir, seg_name)

        if not os.path.exists(seg_path):
            print(f" Warning: No matching segmentation found for {fname}. Skipping...")
            continue

        print(f" Plotting & evaluating paired maps for {fname}...")

        # Load images
        seg_img = nib.load(seg_path)
        gt_img_raw = nib.load(gt_path)

        # Resample spatial dimensions to match target template shape
        gt_img_resampled = image.resample_to_img(
            gt_img_raw, seg_img, interpolation="continuous", force_resample=True, copy_header=True
        )

        # Normalize and Binarize both for metric computation
        _, gt_bin_data = normalize_img_and_binarize(gt_img_resampled, threshold=0.95)
        _, seg_bin_data = normalize_img_and_binarize(seg_img, threshold=0.95)

        # Compute Metrics
        dice_score = compute_dice(gt_bin_data, seg_bin_data)
        r2 = r2_score(gt_bin_data.ravel(), seg_bin_data.ravel())

        # Z-score normalize (without binarizing) for surface plotting
        gt_norm_img = normalize_img(gt_img_resampled)
        seg_norm_img = normalize_img(seg_img)

        # --- Plot GT surface ---
        result_gt = plotting.plot_img_on_surf(
            gt_norm_img,
            title="Ground Truth",
            **plot_parameters
        )
        fig_gt = result_gt[0] if isinstance(result_gt, tuple) else result_gt

        # --- Plot Seg surface ---
        result_seg = plotting.plot_img_on_surf(
            seg_norm_img,
            title="Segmented",
            **plot_parameters
        )
        fig_seg = result_seg[0] if isinstance(result_seg, tuple) else result_seg

        # --- Grab rendered canvases from both figures ---
        fig_gt.canvas.draw()
        fig_seg.canvas.draw()

        buf_gt  = np.frombuffer(fig_gt.canvas.tostring_rgb(),  dtype=np.uint8)
        buf_seg = np.frombuffer(fig_seg.canvas.tostring_rgb(), dtype=np.uint8)

        w_gt,  h_gt  = fig_gt.canvas.get_width_height()
        w_seg, h_seg = fig_seg.canvas.get_width_height()

        img_gt  = buf_gt.reshape(h_gt,  w_gt,  3)
        img_seg = buf_seg.reshape(h_seg, w_seg, 3)

        plt.close(fig_gt)
        plt.close(fig_seg)

        # --- Combine side-by-side in a new figure ---
        # Pad heights if they differ (shouldn't, but defensive)
        max_h = max(h_gt, h_seg)
        if h_gt < max_h:
            pad = np.ones((max_h - h_gt, w_gt, 3), dtype=np.uint8) * 255
            img_gt = np.vstack([img_gt, pad])
        if h_seg < max_h:
            pad = np.ones((max_h - h_seg, w_seg, 3), dtype=np.uint8) * 255
            img_seg = np.vstack([img_seg, pad])

        combined = np.hstack([img_gt, img_seg])

        dpi = 150
        fig_out, ax_out = plt.subplots(
            1, 1,
            figsize=(combined.shape[1] / dpi, combined.shape[0] / dpi),
            dpi=dpi
        )
        ax_out.imshow(combined)
        ax_out.axis("off")

        # Metric banner
        text_metrics_display = f"Dice Score: {dice_score:.4f}   |   R² Score: {r2:.4f}"
        fig_out.suptitle(
            text_metrics_display,
            fontsize=14, fontweight="bold", y=1.01, color="black"
        )

        save_filename = fname.replace(".nii.gz", "_evaluation.png")
        save_path = os.path.join(args.output_dir, save_filename)
        fig_out.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig_out)

        print(f"   Saved → {save_path}")

    print(f"\n Successfully saved evaluation plots to: {args.output_dir}/")
