import os
import argparse
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Resized,
    NormalizeIntensityd,
    EnsureTyped,
    SaveImage,
)
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch

from segmentation_model import ViTUNETRSegmentationModel


def load_model_for_inference(config, state_dict):
    """
    Loads a ViTUNETRSegmentationModel for CPU inference.
    """
    model = ViTUNETRSegmentationModel(
        simclr_ckpt_path=config["pretrain"]["simclr_checkpoint_path"],
        img_size=tuple(config["model"]["img_size"]),
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
    )

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            k = k[len("model.") :]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    return model.eval().cpu()


def preprocess_image(image_path, config):
    """
    Loads and preprocesses a single image for CPU inference.
    """
    img_size = tuple(config["model"]["img_size"])

    transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Resized(keys=["image"], spatial_size=img_size, mode="trilinear"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image"]),
        ]
    )

    data = transforms({"image": image_path})
    return data["image"].unsqueeze(0).cpu()


def generate_segmentation(model, image_tensor, config):
    """
    Runs CPU inference and isolates the target batch element cleanly using MONAI slicing.
    """
    with torch.no_grad():
        pred = sliding_window_inference(
            inputs=image_tensor,
            roi_size=tuple(config["model"]["img_size"]),
            sw_batch_size=config["training"]["sw_batch_size"],
            predictor=model,
            overlap=0.5,
            device=torch.device("cpu"),
        )

    pred = torch.sigmoid(pred)
    # pred = (pred > 0.5).float()

    # Slicing allows MONAI's MetaTensor to handle its internal dict safely
    pred_sliced = pred[0]

    return pred_sliced


def save_segmentation(segmentation_tensor, output_dir, output_filename):
    """
    Saves the segmentation mask cleanly using modern MONAI 1.3+ utilities.
    """
    if len(segmentation_tensor.shape) == 3:
        segmentation_tensor = segmentation_tensor.unsqueeze(0)

    # Set output_postfix to "" to prevent MONAI from force-appending an extra suffix
    saver = SaveImage(
        output_dir=output_dir,
        output_postfix="",
        output_ext=".nii.gz",
        resample=False,
        separate_folder=False,
        squeeze_end_dims=True,
    )

    segmentation_tensor.meta["filename_or_obj"] = output_filename
    saver(segmentation_tensor)
    print(f"Segmentation saved to: {os.path.join(output_dir, output_filename)}")


def load_pmids_from_file(pmid_file_path):
    """
    Loads PMIDs from a text file (one PMID per line).
    """
    with open(pmid_file_path, 'r') as f:
        pmids = [line.strip() for line in f if line.strip()]
    return pmids


def find_nifti_files_for_pmid(input_folder, pmid):
    """
    Finds .nii.gz files for a given PMID.
    Assumes filenames contain the PMID or are named as {pmid}.nii.gz or {pmid}_*.nii.gz.
    """
    nifti_files = []
    for filename in os.listdir(input_folder):
        if filename.endswith('.nii.gz') and pmid in filename:
            nifti_files.append(os.path.join(input_folder, filename))
    return nifti_files


def process_single_image(model, image_path, output_dir, config):
    """
    Processes a single image file and saves the segmentation.
    """
    print(f"\nProcessing: {image_path}")

    # Load and preprocess image
    image_tensor = preprocess_image(image_path, config)

    # Generate segmentation
    segmentation_tensor = generate_segmentation(model, image_tensor, config)

    # Create output filename
    image_filename = os.path.basename(image_path)
    if image_filename.endswith(".nii.gz"):
        base_name = image_filename[:-7]
    else:
        base_name = os.path.splitext(image_filename)[0]

    output_filename = f"{base_name}_seg.nii.gz"

    # Save segmentation
    save_segmentation(segmentation_tensor.cpu(), output_dir, output_filename)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate segmentations for multiple images from a folder based on PMID list"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to segmentation checkpoint file",
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to folder containing input .nii.gz files",
    )
    parser.add_argument(
        "--pmid_file",
        type=str,
        required=True,
        help="Path to text file containing PMIDs (one per line)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save segmentation outputs",
    )
    parser.add_argument(
        "--simclr_checkpoint_path",
        type=str,
        required=False,
        default=None,
        help="Override SimCLR checkpoint path from saved config",
    )

    args = parser.parse_args()

    # Validate input paths
    if not os.path.exists(args.input_folder):
        raise FileNotFoundError(f"Input folder not found: {args.input_folder}")

    if not os.path.exists(args.pmid_file):
        raise FileNotFoundError(f"PMID file not found: {args.pmid_file}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load PMIDs
    print(f"Loading PMIDs from: {args.pmid_file}")
    pmids = load_pmids_from_file(args.pmid_file)
    print(f"Found {len(pmids)} PMIDs to process")

    # Load model once (reused for all images)
    print(f"\nLoading checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device("cpu"))
    config = checkpoint["hyper_parameters"]
    state_dict = checkpoint["state_dict"]

    if args.simclr_checkpoint_path:
        print(f"Overriding SimCLR checkpoint path with: {args.simclr_checkpoint_path}")
        config["pretrain"]["simclr_checkpoint_path"] = args.simclr_checkpoint_path

    print("Loading model...")
    model = load_model_for_inference(config, state_dict)
    print("Model loaded successfully")

    # Process each PMID
    successful_count = 0
    failed_count = 0
    failed_pmids = []

    for pmid in pmids:
        print(f"\n{'='*60}")
        print(f"Processing PMID: {pmid}")

        # Find all .nii.gz files for this PMID
        nifti_files = find_nifti_files_for_pmid(args.input_folder, pmid)

        if not nifti_files:
            print(f"Warning: No .nii.gz files found for PMID {pmid}")
            failed_count += 1
            failed_pmids.append(pmid)
            continue

        # Process each file for this PMID
        for nifti_file in nifti_files:
            try:
                process_single_image(model, nifti_file, args.output_dir, config)
                successful_count += 1
            except Exception as e:
                print(f"Error processing {nifti_file}: {str(e)}")
                failed_count += 1
                failed_pmids.append(f"{pmid} - {os.path.basename(nifti_file)}")

    # Print summary
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"Successfully processed: {successful_count} files")
    print(f"Failed: {failed_count} files")

    if failed_pmids:
        print("\nFailed items:")
        for item in failed_pmids:
            print(f"  - {item}")

    print("\nDone.")


if __name__ == "__main__":
    main()
