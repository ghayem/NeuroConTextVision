import os
import glob
import torch
import numpy as np

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    NormalizeIntensityd, EnsureTyped, Resized
)

from segmentation_model import ViTUNETRSegmentationModel

# =========================
# CONFIG
# =========================
CHECKPOINT_PATH = "./checkpoints/segmentation.ckpt"
SIMCLR_PATH = "./checkpoints/BrainIAC.ckpt"
INPUT_DIR = "./KDE_samples"
OUTPUT_DIR = "./brainiac_embeddings"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = (96, 96, 96)

# =========================
# LOAD MODEL
# =========================
def load_model():
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")

    config = ckpt["hyper_parameters"]
    state_dict = ckpt["state_dict"]

    model = ViTUNETRSegmentationModel(
        simclr_ckpt_path=SIMCLR_PATH,
        img_size=tuple(config["model"]["img_size"]),
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"]
    )

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            k = k[len("model."):]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.to(DEVICE).eval()

    return model


# =========================
# PREPROCESS
# =========================
def get_transforms():
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        Resized(keys=["image"], spatial_size=IMG_SIZE),
        EnsureTyped(keys=["image"]),
    ])


# =========================
# EXTRACTION
# =========================
def extract_patch_embeddings():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = load_model()
    vit = model.unetr.vit

    transforms = get_transforms()
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.nii.gz")))

    print(f"Found {len(files)} scans\n")

    for i, path in enumerate(files, 1):
        fname = os.path.basename(path).replace(".nii.gz", "")
        save_dir = os.path.join(OUTPUT_DIR, fname)
        os.makedirs(save_dir, exist_ok=True)

        print(f"[{i}/{len(files)}] {fname}")

        try:
            data = transforms({"image": path})
            x = data["image"].unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                
                tokens = vit.patch_embedding(x)

                if hasattr(vit, "pos_embed"):
                    tokens = tokens + vit.pos_embed

                for blk in vit.blocks:
                    tokens = blk(tokens)

                if hasattr(vit, "norm"):
                    tokens = vit.norm(tokens)

            tokens = tokens.squeeze(0)  # [216, 768]

            # sanity check
            assert tokens.shape == (216, 768), f"Wrong shape: {tokens.shape}"

            np.save(
                os.path.join(save_dir, "patch_embeddings.npy"),
                tokens.cpu().numpy()
            )

            print("saved\n")

        except Exception as e:
            print(f"FAILED: {e}\n")


if __name__ == "__main__":
    extract_patch_embeddings()