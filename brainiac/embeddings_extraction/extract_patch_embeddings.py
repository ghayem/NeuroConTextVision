import os
import glob
import torch
import numpy as np
from multiprocessing import Pool
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    NormalizeIntensityd, EnsureTyped, Resized
)
from segmentation_model import ViTUNETRSegmentationModel

CHECKPOINT_PATH = "../checkpoints/segmentation.ckpt"
SIMCLR_PATH     = "../checkpoints/BrainIAC.ckpt"
INPUT_DIR       = "../../../data/brain_images/"
OUTPUT_DIR      = "./brainiac_embeddings"
IMG_SIZE        = (96, 96, 96)
NUM_WORKERS     = 8

_vit        = None
_transforms = None

def worker_init(_):
    global _vit, _transforms

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    config = ckpt["hyper_parameters"]
    state_dict = ckpt["state_dict"]

    model = ViTUNETRSegmentationModel(
        simclr_ckpt_path=SIMCLR_PATH,
        img_size=tuple(config["model"]["img_size"]),
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"]
    )
    new_state_dict = {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    _vit = model.unetr.vit

    _transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        Resized(keys=["image"], spatial_size=IMG_SIZE),
        EnsureTyped(keys=["image"]),
    ])

def process_file(args):
    i, total, path = args
    fname    = os.path.basename(path).replace(".nii.gz", "")
    out_path = os.path.join(OUTPUT_DIR, f"{fname}.npy")  # ← flat

    if os.path.exists(out_path):
        print(f"[{i}/{total}] {fname} — skip")
        return fname, "skipped"

    # removed os.makedirs for subfolder
    print(f"[{i}/{total}] {fname} — processing (pid {os.getpid()})")

    try:
        data = _transforms({"image": path})
        x    = data["image"].unsqueeze(0)

        with torch.no_grad():
            tokens = _vit.patch_embedding(x)
            if hasattr(_vit, "pos_embed"):
                tokens = tokens + _vit.pos_embed
            for blk in _vit.blocks:
                tokens = blk(tokens)
            if hasattr(_vit, "norm"):
                tokens = _vit.norm(tokens)

        tokens = tokens.squeeze(0)
        assert tokens.shape == (216, 768), f"Wrong shape: {tokens.shape}"
        np.save(out_path, tokens.numpy())
        print(f"[{i}/{total}] {fname} — saved")
        return fname, "ok"

    except Exception as e:
        print(f"[{i}/{total}] {fname} — FAILED: {e}")
        return fname, f"error: {e}"

def extract_patch_embeddings():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.nii.gz")))
    total = len(files)
    print(f"Found {total} scans — launching {NUM_WORKERS} workers\n")

    args = [(i, total, p) for i, p in enumerate(files, 1)]

    with Pool(processes=NUM_WORKERS, initializer=worker_init, initargs=(None,)) as pool:
        results = pool.map(process_file, args)

    ok      = sum(1 for _, s in results if s == "ok")
    skipped = sum(1 for _, s in results if s == "skipped")
    errors  = [(n, s) for n, s in results if s not in ("ok", "skipped")]

    print(f"\nDone — {ok} saved, {skipped} skipped, {len(errors)} failed")
    for name, err in errors:
        print(f"  ✗ {name}: {err}")

if __name__ == "__main__":
    extract_patch_embeddings()