import os
import argparse
import nibabel as nib
import numpy as np


def calculate_dice_with_normalization(input_path, seg_path, threshold=0.5):
    if not os.path.exists(input_path) or not os.path.exists(seg_path):
        print("Error: One or both files do not exist.")
        return

    # Load NIfTI volumes
    input_img = nib.load(input_path)
    seg_img = nib.load(seg_path)

    input_data = input_img.get_fdata()
    seg_data = seg_img.get_fdata()

    if input_data.shape != seg_data.shape:
        raise ValueError(
            f"Shape mismatch! Input is {input_data.shape}, but segmentation is {seg_data.shape}"
        )

    # 1. Clean up interpolation artifacts (clamp negative densities to 0)
    input_data = np.clip(input_data, a_min=0.0, a_max=None)

    # 2. Min-Max scale the volume dynamically to a clean [0, 1] range
    input_max = input_data.max()
    if input_max > 0:
        input_data = input_data / input_max
    else:
        print("Warning: Input image contains only zeros. Cannot normalize.")

    # 3. Compute boolean masks based on your target threshold
    input_binary = (input_data >= threshold).astype(bool)
    seg_binary = (seg_data > 0.5).astype(bool)

    # 4. Calculate Dice
    intersection = np.logical_and(input_binary, seg_binary).sum()
    total_voxels = input_binary.sum() + seg_binary.sum()

    if total_voxels == 0:
        dice_score = 1.0
    else:
        dice_score = (2.0 * intersection) / total_voxels

    print(f"Comparison: {os.path.basename(input_path)} vs {os.path.basename(seg_path)}")
    print(f"  Input Scaled Max Range:     [0.0, {input_data.max()}]")
    print(f"  Applied Threshold:          {threshold}")
    print(f"  Binarized Input Volume:     {input_binary.sum():,} voxels")
    print(f"  Segmentation Mask Volume:   {seg_binary.sum():,} voxels")
    print(f"  Voxel Intersection:         {intersection:,} voxels")
    print(f"  Dice Coefficient:           {dice_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate Dice score with robust min-max input normalization."
    )
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--seg_image", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()
    calculate_dice_with_normalization(args.input_image, args.seg_image, args.threshold)
