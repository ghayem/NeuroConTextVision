import os
import argparse
import glob
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
            NormalizeIntensityd(
                keys="image",
                nonzero=True,
                channel_wise=True,
            ),
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

    # Keep MetaTensor metadata intact
    pred_sliced = pred[0]

    return pred_sliced


def save_segmentation(segmentation_tensor, output_dir, output_filename):
    """
    Saves the segmentation mask using MONAI utilities.
    """
    if len(segmentation_tensor.shape) == 3:
        segmentation_tensor = segmentation_tensor.unsqueeze(0)

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

    print(
        f"Segmentation saved to: "
        f"{os.path.join(output_dir, output_filename)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate segmentations for all .nii.gz images "
            "in a folder on CPU"
        )
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to segmentation checkpoint file",
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input .nii.gz images",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory where segmentations will be saved. "
            "Defaults to input_dir if not specified."
        ),
    )

    parser.add_argument(
        "--simclr_checkpoint_path",
        type=str,
        default=None,
        help="Override SimCLR checkpoint path stored in checkpoint config",
    )

    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Gather input files
    # ------------------------------------------------------------------
    search_path = os.path.join(args.input_dir, "*.nii.gz")
    image_paths = sorted(glob.glob(search_path))

    # Avoid re-processing previously generated segmentations
    image_paths = [
        p for p in image_paths
        if not p.endswith("_seg.nii.gz")
    ]

    if not image_paths:
        print(
            f"No valid input .nii.gz files found in: "
            f"{args.input_dir}"
        )
        raise SystemExit(0)

    print(f"Found {len(image_paths)} images to process.")
    print(f"Output directory: {output_dir}")

    # ------------------------------------------------------------------
    # Load checkpoint and model
    # ------------------------------------------------------------------
    print(f"Loading checkpoint: {args.checkpoint_path}")

    checkpoint = torch.load(
        args.checkpoint_path,
        map_location=torch.device("cpu"),
    )

    config = checkpoint["hyper_parameters"]
    state_dict = checkpoint["state_dict"]

    if args.simclr_checkpoint_path:
        print(
            "Overriding SimCLR checkpoint path with: "
            f"{args.simclr_checkpoint_path}"
        )
        config["pretrain"][
            "simclr_checkpoint_path"
        ] = args.simclr_checkpoint_path

    print("Loading model...")
    model = load_model_for_inference(config, state_dict)

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    for idx, image_path in enumerate(image_paths, start=1):
        print(
            f"\n--- Processing [{idx}/{len(image_paths)}]: "
            f"{os.path.basename(image_path)} ---"
        )

        print("Preprocessing image...")
        image_tensor = preprocess_image(image_path, config)

        print("Generating segmentation...")
        segmentation_tensor = generate_segmentation(
            model,
            image_tensor,
            config,
        )

        image_filename = os.path.basename(image_path)
        base_name = image_filename[:-7]  # remove ".nii.gz"

        output_filename = f"{base_name}_seg.nii.gz"

        print("Saving segmentation...")
        save_segmentation(
            segmentation_tensor.cpu(),
            output_dir,
            output_filename,
        )

    print("\nAll segmentations finished successfully.")
