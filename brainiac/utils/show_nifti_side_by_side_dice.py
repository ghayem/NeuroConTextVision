#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import matplotlib

matplotlib.use('Agg')
import warnings

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from monai.data import MetaTensor
from monai.transforms import (Compose, EnsureChannelFirstd, EnsureTyped,
                              LoadImaged, Resized)
from nilearn import plotting

warnings.filterwarnings('ignore')


def load_and_resize_nifti(file_path: str, target_size: tuple) -> np.ndarray:
    """
    Loads a NIfTI file and resizes it to target size using MONAI transforms.
    This matches the preprocessing used during segmentation.
    """
    data = {"image": file_path}

    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),  # Adds channel dimension
        Resized(keys=["image"], spatial_size=target_size, mode="trilinear"),
        EnsureTyped(keys=["image"]),
    ])

    transformed = transforms(data)
    image_tensor = transformed["image"]

    # Remove channel dimension (1, H, W, D) -> (H, W, D)
    if isinstance(image_tensor, MetaTensor):
        image_numpy = image_tensor.squeeze(0).cpu().numpy()
    else:
        image_numpy = image_tensor.squeeze(0).numpy()

    return image_numpy


def normalize_z_score(data: np.ndarray) -> np.ndarray:
    """
    Applies z-score normalization to the data array.
    """
    mean = np.mean(data)
    std = np.std(data)

    if std == 0:
        return np.zeros_like(data)

    normalized = (data - mean) / std
    return normalized


def apply_threshold(data: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """
    Applies threshold to binarize the data.
    Values below threshold become 0, above become 1.
    """
    binary = (data > threshold).astype(np.float32)
    return binary


def compute_dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Computes Dice similarity coefficient between two binary arrays.
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    intersection = np.sum(pred_flat * target_flat)
    sum_pred = np.sum(pred_flat)
    sum_target = np.sum(target_flat)

    if sum_pred + sum_target == 0:
        return 1.0
    elif sum_pred == 0 or sum_target == 0:
        return 0.0

    dice = (2.0 * intersection) / (sum_pred + sum_target)
    return dice


def normalize_img_for_viz(img):
    """Z-score normalizes a Nifti image for visualization."""
    if isinstance(img, str):
        img = nib.load(img)
    data = img.get_fdata()
    std = data.std() if data.std() > 1e-8 else 1.0
    data = (data - data.mean()) / std
    return nib.Nifti1Image(data, img.affine)


def visualize_pair(ground_truth_path, segmentation_path, output_path=None, target_size=(96, 96, 96), threshold=0.95):
    """
    Creates a side-by-side visualization of ground truth and segmentation.
    Uses the exact same Dice computation as the original script.
    """
    # Load and resize both images (same as process_pair)
    original_resized = load_and_resize_nifti(ground_truth_path, target_size)
    seg_resized = load_and_resize_nifti(segmentation_path, target_size)

    # Normalize with z-score
    original_normalized = normalize_z_score(original_resized)
    seg_normalized = normalize_z_score(seg_resized)

    # Apply threshold for binarization
    original_binary = apply_threshold(original_normalized, threshold)
    seg_binary = apply_threshold(seg_normalized, threshold)

    # Compute Dice score (same as compute_dice_score)
    dice_score = compute_dice_score(original_binary, seg_binary)

    # Create Nifti images for visualization (using resized normalized data)
    gt_orig = nib.load(ground_truth_path)
    seg_orig = nib.load(segmentation_path)

    gt_img_viz = nib.Nifti1Image(original_normalized, gt_orig.affine)
    seg_img_viz = nib.Nifti1Image(seg_normalized, seg_orig.affine)

    # Plot ground truth
    gt_display = plotting.plot_img_on_surf(
        gt_img_viz,
        title="Ground Truth",
        cmap="bwr",
        vmax=6,
        views=["lateral"],
        colorbar=False,
        alpha=1.0,
        bg_on_data=True,
    )
    
    # Get figure and increase title font size for ground truth
    gt_fig = gt_display[0] if isinstance(gt_display, tuple) else gt_display
    gt_fig.suptitle("Ground Truth", fontsize=24)

    # Plot segmentation
    seg_display = plotting.plot_img_on_surf(
        seg_img_viz,
        title="Segmented",
        cmap="bwr",
        vmax=6,
        views=["lateral"],
        colorbar=False,
        alpha=1.0,
        bg_on_data=True,
    )
    
    # Get figure and increase title font size for segmentation
    seg_fig = seg_display[0] if isinstance(seg_display, tuple) else seg_display
    seg_fig.suptitle("Segmented", fontsize=24)

    # Convert figures to arrays
    def fig_to_array(fig):
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        w, h = fig.canvas.get_width_height()
        return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)

    gt_arr = fig_to_array(gt_fig)
    seg_arr = fig_to_array(seg_fig)

    # Close individual figures
    plt.close(gt_fig)
    plt.close(seg_fig)

    # Pad heights if they differ
    h1, h2 = gt_arr.shape[0], seg_arr.shape[0]
    if h1 != h2:
        max_h = max(h1, h2)
        def pad_height(arr, target_h):
            pad = target_h - arr.shape[0]
            return np.pad(arr, ((0, pad), (0, 0), (0, 0)), constant_values=255)
        gt_arr = pad_height(gt_arr, max_h)
        seg_arr = pad_height(seg_arr, max_h)

    # Combine side by side
    combined = np.concatenate([gt_arr, seg_arr], axis=1)

    # Create final figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.imshow(combined)
    ax.axis("off")
    
    # Add Dice score as subtitle
    fig.suptitle(f'Dice Score: {dice_score:.2f}', fontsize=20, fontweight='normal', color='orange', y=0.98)

    plt.tight_layout()

    # Save or show
    if output_path:
        if output_path.endswith('/'):
            basename = os.path.basename(ground_truth_path).replace('.nii.gz', '')
            output_path = os.path.join(output_path, f'{basename}_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to: {output_path}")
    else:
        plt.show()

    plt.close(fig)
    return dice_score


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ground truth and segmented NIfTI files side by side"
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        required=True,
        help="Path to ground truth .nii.gz file"
    )
    parser.add_argument(
        "--segmentation",
        type=str,
        required=True,
        help="Path to segmentation .nii.gz file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the visualization image (file or directory). If directory, auto-generates filename."
    )
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=3,
        default=[96, 96, 96],
        help="Target size for resizing (height width depth). Default: 96 96 96"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Threshold for binarization (default: 0.95)"
    )

    args = parser.parse_args()

    print(f"Ground truth: {args.ground_truth}")
    print(f"Segmentation: {args.segmentation}")

    target_size = tuple(args.target_size)
    dice_score = visualize_pair(args.ground_truth, args.segmentation, args.output, target_size, args.threshold)
    print(f"Dice Score: {dice_score:.4f}")


if __name__ == "__main__":
    main()
