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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate segmentation for a single image on CPU"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to segmentation checkpoint file",
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to input image (.nii.gz)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save segmentation output",
    )
    parser.add_argument(
        "--simclr_checkpoint_path",
        type=str,
        required=False,
        default=None,
        help="Override SimCLR checkpoint path from saved config",
    )

    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device("cpu"))
    config = checkpoint["hyper_parameters"]
    state_dict = checkpoint["state_dict"]

    if args.simclr_checkpoint_path:
        print(f"Overriding SimCLR checkpoint path with: {args.simclr_checkpoint_path}")
        config["pretrain"]["simclr_checkpoint_path"] = args.simclr_checkpoint_path

    os.makedirs(args.output_dir, exist_ok=True)

    print("1. Loading model...")
    model = load_model_for_inference(config, state_dict)

    print(f"2. Loading and preprocessing image: {args.image_path}...")
    image_tensor = preprocess_image(args.image_path, config)

    print("3. Generating segmentation (this may take a moment on CPU)...")
    segmentation_tensor = generate_segmentation(model, image_tensor, config)

    # Safely strip complex double extension (.nii.gz) structures
    image_filename = os.path.basename(args.image_path)
    if image_filename.endswith(".nii.gz"):
        base_name = image_filename[:-7]
    else:
        base_name = os.path.splitext(image_filename)[0]

    # Append the required '_seg' suffix explicitly here
    output_filename = f"{base_name}_seg.nii.gz"

    print(f"4. Saving segmentation to {os.path.join(args.output_dir, output_filename)}...")
    save_segmentation(segmentation_tensor.cpu(), args.output_dir, output_filename)

    print("Done.")
