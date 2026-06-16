"""
extract_brainiac_embeddings.py

Robust BrainIAC embedding extraction.

Features
--------
- Silently skips:
    - missing files
    - corrupt .nii.gz files
    - transform failures
    - NaN/Inf outputs
    - model inference failures
- Preserves perfect alignment between:
    - BrainIAC embeddings
    - text embeddings
- Never crashes because of a bad sample
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import get_validation_transform
from load_brainiac import load_brainiac

# ---------------------------------------------------------------------
# Make BrainIAC src importable
# ---------------------------------------------------------------------

BRAINIAC_SRC = Path(__file__).resolve().parent.parent / "BrainIAC" / "src"

if str(BRAINIAC_SRC) not in sys.path:
    sys.path.insert(0, str(BRAINIAC_SRC))


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class NiiDataset(Dataset):

    IMAGE_KEY = "image"

    def __init__(self, nii_paths, valid_indices, transform):
        self.nii_paths = nii_paths
        self.valid_indices = valid_indices
        self.transform = transform

    def __len__(self):
        return len(self.nii_paths)

    def __getitem__(self, idx):

        path = self.nii_paths[idx]
        original_index = self.valid_indices[idx]

        try:
            result = self.transform(
                {self.IMAGE_KEY: str(path)}
            )
            image = result[self.IMAGE_KEY]

            if torch.isnan(image).any() or torch.isinf(image).any():
                print(f"  [SKIP] {path.name} -> NaN/Inf in image tensor")
                return None

            return {
                "image": image,
                "orig_idx": original_index,
            }

        except Exception as e:
            print(f"  [SKIP] {path.name} -> {type(e).__name__}: {e}")
            return None


# ---------------------------------------------------------------------
# Custom collate_fn
# ---------------------------------------------------------------------

def collate_skip_none(batch):

    batch = [x for x in batch if x is not None]

    if len(batch) == 0:
        return None

    images = torch.stack([x["image"] for x in batch])
    orig_indices = [x["orig_idx"] for x in batch]

    return {
        "image": images,
        "orig_idx": orig_indices,
    }


# ---------------------------------------------------------------------
# PMID loader
# ---------------------------------------------------------------------

def load_pmids(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------

def extract_split(
    pmids,
    text_embeddings,
    nii_dir,
    model,
    transform,
    device,
    batch_size,
    num_workers,
):

    assert len(pmids) == len(text_embeddings), (
        f"Mismatch: {len(pmids)} PMIDs vs "
        f"{len(text_embeddings)} text embeddings"
    )

    # -------------------------------------------------------------
    # Resolve existing files
    # -------------------------------------------------------------

    valid_paths = []
    valid_indices = []

    for i, pmid in enumerate(pmids):

        nii_path = nii_dir / f"pmid_{pmid}.nii.gz"

        if not nii_path.exists():
            print(f"  [SKIP] {nii_path.name} -> missing")
            continue

        valid_paths.append(nii_path)
        valid_indices.append(i)

    print(f"{len(valid_paths)} / {len(pmids)} files found on disk")

    if len(valid_paths) == 0:
        return np.empty((0,)), np.empty((0,))

    # -------------------------------------------------------------
    # Dataset / loader
    # -------------------------------------------------------------

    dataset = NiiDataset(
        nii_paths=valid_paths,
        valid_indices=valid_indices,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_skip_none,
    )

    kept_brain = []
    kept_text = []

    model.eval()

    with torch.no_grad():

        for batch in tqdm(
            dataloader,
            desc="  BrainIAC inference",
            unit="batch",
        ):

            if batch is None:
                continue

            try:
                inputs = batch["image"].to(device)
                features = model(inputs)
                features_np = features.cpu().numpy()

            except Exception as e:
                print(f"  [SKIP] model batch failure -> {e}")
                continue

            orig_indices = batch["orig_idx"]

            for feat_vec, orig_idx in zip(features_np, orig_indices):
                if (
                    np.isnan(feat_vec).any()
                    or np.isinf(feat_vec).any()
                ):
                    print(
                        f"  [SKIP] pmid={pmids[orig_idx]} "
                        "-> NaN/Inf in embedding"
                    )
                    continue

                kept_brain.append(feat_vec)
                kept_text.append(text_embeddings[orig_idx])

    print(f"  {len(kept_brain)} samples kept")

    if len(kept_brain) == 0:
        return np.empty((0,)), np.empty((0,))

    brain_out = np.stack(kept_brain, axis=0)
    text_out = np.stack(kept_text, axis=0)

    return brain_out, text_out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--nii_dir", required=True)

    parser.add_argument("--train_pmids", required=True)
    parser.add_argument("--test_pmids", required=True)

    parser.add_argument("--train_text_pkl", required=True)
    parser.add_argument("--test_text_pkl", required=True)

    parser.add_argument("--checkpoint", default=None)

    parser.add_argument("--output_dir", default=".")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)

    args = parser.parse_args()

    # -------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------

    nii_dir = Path(args.nii_dir).resolve()

    out_dir = (
        Path(args.output_dir).resolve()
        / "new_and_aligned_embeddings"
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Device
    # -------------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # -------------------------------------------------------------
    # Text embeddings
    # -------------------------------------------------------------

    print("Loading text embeddings...")

    with open(args.train_text_pkl, "rb") as f:
        train_text_emb = pickle.load(f)

    with open(args.test_text_pkl, "rb") as f:
        test_text_emb = pickle.load(f)

    print(f"  train text: {train_text_emb.shape}")
    print(f"  test  text: {test_text_emb.shape}")

    # -------------------------------------------------------------
    # PMIDs
    # -------------------------------------------------------------

    train_pmids = load_pmids(args.train_pmids)
    test_pmids = load_pmids(args.test_pmids)

    print(f"  train PMIDs: {len(train_pmids)}")
    print(f"  test  PMIDs: {len(test_pmids)}")

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------

    print("Loading BrainIAC model...")

    model = load_brainiac(args.checkpoint, device)

    transform = get_validation_transform()

    # -------------------------------------------------------------
    # TRAIN
    # -------------------------------------------------------------

    print("\n=== TRAIN split ===")

    brain_train, text_train = extract_split(
        pmids=train_pmids,
        text_embeddings=train_text_emb,
        nii_dir=nii_dir,
        model=model,
        transform=transform,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # -------------------------------------------------------------
    # TEST
    # -------------------------------------------------------------

    print("\n=== TEST split ===")

    brain_test, text_test = extract_split(
        pmids=test_pmids,
        text_embeddings=test_text_emb,
        nii_dir=nii_dir,
        model=model,
        transform=transform,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # -------------------------------------------------------------
    # Final checks
    # -------------------------------------------------------------

    assert len(brain_train) == len(text_train)
    assert len(brain_test) == len(text_test)

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    outputs = {
        "brainiac_embeddings_train.pkl": brain_train,
        "text_embeddings_train.pkl": text_train,
        "brainiac_embeddings_test.pkl": brain_test,
        "text_embeddings_test.pkl": text_test,
    }

    for filename, arr in outputs.items():

        out_path = out_dir / filename

        with open(out_path, "wb") as f:
            pickle.dump(arr, f)

        print(f"Saved: {out_path}")
        print(f"  shape = {arr.shape}")

    print("\nDone.")


if __name__ == "__main__":
    main()
