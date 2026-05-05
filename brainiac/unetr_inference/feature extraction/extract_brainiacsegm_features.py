import os
import glob
import torch
import numpy as np
import nibabel as nib

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
OUTPUT_DIR = "brainiac_embeddings/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = (96, 96, 96)
PATCH_SIZE = (16, 16, 16)
NUM_PATCHES = 216   # 6x6x6

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

    # clean keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            k = k[len("model."):]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.to(DEVICE).eval()

    return model


# =========================
# HOOKS FOR FEATURES
# =========================
def register_hooks(vit):
    features = {}

    def get_hook(name):
        def hook(module, input, output):
            features[name] = output.detach()
        return hook

    hooks = []

    # capture transformer blocks (skip features)
    for i in [2, 5, 8, 11]:
        hooks.append(vit.blocks[i].register_forward_hook(get_hook(f"block_{i}")))

    return features, hooks


# =========================
# PREPROCESS
# =========================
def get_transforms():
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        Resized(keys=["image"], spatial_size=IMG_SIZE),  # required by model
        EnsureTyped(keys=["image"]),
    ])


# =========================
# MAIN EXTRACTION
# =========================
def extract_embeddings():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = load_model()
    vit = model.unetr.vit  # encoder

    transforms = get_transforms()

    input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.nii.gz")))

    print(f"Found {len(input_files)} scans\n")

    for i, path in enumerate(input_files, 1):
        fname = os.path.basename(path).replace(".nii.gz", "")
        save_dir = os.path.join(OUTPUT_DIR, fname)
        os.makedirs(save_dir, exist_ok=True)

        print(f"[{i}/{len(input_files)}] {fname}")

        # -------------------------
        # LOAD IMAGE
        # -------------------------
        data = transforms({"image": path})
        x = data["image"].unsqueeze(0).to(DEVICE)

        # -------------------------
        # REGISTER HOOKS
        # -------------------------
        captured, hooks = register_hooks(vit)

        with torch.no_grad():
            out = vit(x)

        # unwrap tuple
        if isinstance(out, (list, tuple)):
            tokens = out[0]
        else:
            tokens = out

        tokens = tokens.squeeze(0)  # [216, 768]

        # remove hooks
        for h in hooks:
            h.remove()

        # -------------------------
        # PATCH EMBEDDINGS
        # -------------------------
        tokens = tokens.squeeze(0)  # [216, 768]

        # -------------------------
        # MEAN POOL (for retrieval)
        # -------------------------
        pooled = tokens.mean(dim=0)  # [768]

        # -------------------------
        # SAVE EVERYTHING
        # -------------------------

        # 1. patch embeddings (CRITICAL)
        np.save(os.path.join(save_dir, "patch_embeddings.npy"),
                tokens.cpu().numpy())

        # 2. pooled embedding (for NeuroConText)
        np.save(os.path.join(save_dir, "pooled_embedding.npy"),
                pooled.cpu().numpy())

        # 3. decoder input (same as tokens)
        np.save(os.path.join(save_dir, "decoder_tokens.npy"),
                tokens.cpu().numpy())

        # 4. skip features (multi-scale)
        for k, v in captured.items():
            feat = v
            while isinstance(feat, (list, tuple)):
                feat = feat[-1]
            np.save(os.path.join(save_dir, f"{k}.npy"),
                    feat.squeeze(0).cpu().numpy())

        # 5. original input (for reconstruction alignment)
        np.save(os.path.join(save_dir, "input.npy"),
                data["image"].numpy())

        # 6. metadata
        meta = {
            "original_shape": nib.load(path).shape,
            "resized_shape": IMG_SIZE,
            "patch_size": PATCH_SIZE,
            "num_patches": NUM_PATCHES,
        }
        np.save(os.path.join(save_dir, "meta.npy"), meta)

        print("   Saved embeddings + metadata\n")


if __name__ == "__main__":
    extract_embeddings()