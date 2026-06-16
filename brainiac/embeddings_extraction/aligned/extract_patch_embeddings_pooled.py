"""
Extracts mean-pooled ViT brain embeddings and saves 4 consolidated pkl files:
  brainiac_embeddings_train.pkl  brainiac_embeddings_test.pkl
  text_embeddings_train.pkl      text_embeddings_test.pkl

NaN handling:
  - Checks for NaN/Inf values in extracted embeddings
  - Skips both brain embedding and corresponding text embedding if NaN detected
  - Preserves alignment by only including valid pairs
  - Logs detailed information about skipped samples
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    NormalizeIntensityd, EnsureTyped, Resized,
)
from segmentation_model import ViTUNETRSegmentationModel


# ── NaN checking utilities ────────────────────────────────────────────────────────────

def has_nan_or_inf(array: np.ndarray, name: str = "") -> bool:
    """Check if array contains NaN or Inf values."""
    if array is None:
        print(f"  ⚠️ {name}: array is None")
        return True

    has_nan = np.isnan(array).any()
    has_inf = np.isinf(array).any()

    if has_nan or has_inf:
        if has_nan:
            nan_count = np.isnan(array).sum()
            print(f"  ❌ {name}: contains {nan_count} NaN values")
        if has_inf:
            inf_count = np.isinf(array).sum()
            print(f"  ❌ {name}: contains {inf_count} Inf values")
        return True

    return False

def validate_embedding(embedding: np.ndarray, pmid: str, split: str) -> bool:
    """
    Validate a single embedding for NaN/Inf.
    Returns True if valid, False if invalid.
    """
    if embedding is None:
        print(f"  [SKIP] {split} - PMID {pmid}: embedding is None")
        return False

    if has_nan_or_inf(embedding, f"{split} PMID {pmid}"):
        return False

    # Check for all zeros (failed embedding)
    if np.all(embedding == 0):
        print(f"  [SKIP] {split} - PMID {pmid}: embedding is all zeros")
        return False

    # Check for reasonable values (optional: check for extreme outliers)
    if np.abs(embedding).max() > 1e6:
        print(f"  [SKIP] {split} - PMID {pmid}: embedding has extreme values (max={np.abs(embedding).max():.2e})")
        return False

    return True

# ── Model loading ────────────────────────────────────────────────────────────

def load_vit_backbone(seg_ckpt: str, brainiac_ckpt: str, device: torch.device):
    ckpt = torch.load(seg_ckpt, map_location="cpu")
    cfg  = ckpt["hyper_parameters"]
    model = ViTUNETRSegmentationModel(
        simclr_ckpt_path = brainiac_ckpt,
        img_size         = tuple(cfg["model"]["img_size"]),
        in_channels      = cfg["model"]["in_channels"],
        out_channels     = cfg["model"]["out_channels"],
    )
    sd = {(k[6:] if k.startswith("model.") else k): v
          for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model.unetr.vit


# ── Single-scan forward pass → mean-pooled vector with NaN validation ───────────────────────────

@torch.no_grad()
def embed_scan(nii_path: Path, vit, transform, device: torch.device, pmid: str, split: str) -> np.ndarray:
    """
    Returns a (768,) float32 vector.
    Returns None if any NaN/Inf detected during processing.
    """
    try:
        data   = transform({"image": str(nii_path)})
        x      = data["image"].unsqueeze(0).to(device)          # [1,C,96,96,96]

        # Check input for NaN
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"  [FAIL] {split} - pmid_{pmid}: NaN/Inf in input after transform")
            return None

        tokens = vit.patch_embedding(x)                         # [1, N, 768]

        # Check after patch embedding
        if torch.isnan(tokens).any() or torch.isinf(tokens).any():
            print(f"  [FAIL] {split} - pmid_{pmid}: NaN/Inf after patch embedding")
            return None

        if hasattr(vit, "pos_embed"):
            tokens = tokens + vit.pos_embed

            # Check after positional embedding
            if torch.isnan(tokens).any() or torch.isinf(tokens).any():
                print(f"  [FAIL] {split} - pmid_{pmid}: NaN/Inf after positional embedding")
                return None

        for i, blk in enumerate(vit.blocks):
            tokens = blk(tokens)
            # Check after each block (optional: can be commented out for performance)
            if torch.isnan(tokens).any() or torch.isinf(tokens).any():
                print(f"  [FAIL] {split} - pmid_{pmid}: NaN/Inf after block {i}")
                return None

        if hasattr(vit, "norm"):
            tokens = vit.norm(tokens)

            # Check after normalization
            if torch.isnan(tokens).any() or torch.isinf(tokens).any():
                print(f"  [FAIL] {split} - pmid_{pmid}: NaN/Inf after normalization")
                return None

        # Mean pooling
        embedding = tokens.squeeze(0).mean(dim=0).cpu().numpy()      # [768]

        # Final validation
        if has_nan_or_inf(embedding, f"{split} PMID {pmid}"):
            return None

        return embedding

    except Exception as e:
        print(f"  [FAIL] {split} - pmid_{pmid}: Exception during embedding: {e}")
        return None


# ── Process one split with alignment preservation ────────────────────────────────────────

def process_split(
    pmids: list[str],
    text_embeddings,          # array-like, shape [N, D]
    nii_dir: str,
    vit,
    transform,
    device: torch.device,
    split_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        brain_embs : np.ndarray  shape [M, 768]   (M ≤ N, only valid pairs)
        text_embs  : np.ndarray  shape [M, D]     (aligned with brain_embs)

    Skips:
        - Missing NIfTI files
        - Failed embedding extraction
        - NaN/Inf values in either brain or text embeddings
        - Mismatched shapes
    """
    brain_rows = []
    text_rows = []
    skipped_count = 0
    skipped_pmids = []

    # Convert text_embeddings to list for easier indexing if it's a numpy array
    if isinstance(text_embeddings, np.ndarray):
        text_embeddings_list = [text_embeddings[i] for i in range(len(text_embeddings))]
    else:
        text_embeddings_list = text_embeddings

    for i, pmid in enumerate(tqdm(pmids, desc=f"{split_label}")):
        # Check if text embedding is valid first
        text_emb = text_embeddings_list[i]

        # Validate text embedding
        if not validate_embedding(text_emb, f"{split_label}_text_{pmid}", split_label):
            print(f"  [SKIP] {split_label} - PMID {pmid}: invalid text embedding")
            skipped_count += 1
            skipped_pmids.append((pmid, "text_invalid"))
            continue

        # Check if NIfTI file exists
        nii_path = Path(nii_dir) / f"pmid_{pmid}.nii.gz"
        if not nii_path.exists():
            print(f"  [SKIP] {split_label} - PMID {pmid}: NIfTI file not found: {nii_path.name}")
            skipped_count += 1
            skipped_pmids.append((pmid, "nifti_missing"))
            continue

        # Extract brain embedding
        brain_emb = embed_scan(nii_path, vit, transform, device, pmid, split_label)

        # Validate brain embedding
        if brain_emb is None or not validate_embedding(brain_emb, f"{split_label}_brain_{pmid}", split_label):
            print(f"  [SKIP] {split_label} - PMID {pmid}: invalid brain embedding")
            skipped_count += 1
            skipped_pmids.append((pmid, "brain_invalid"))
            continue

        # Both embeddings are valid, keep them
        brain_rows.append(brain_emb)
        text_rows.append(text_emb)

    # Convert to numpy arrays
    if len(brain_rows) == 0:
        print(f"  ⚠️ WARNING: No valid samples found for {split_label} split!")
        return np.array([]).reshape(0, 0), np.array([]).reshape(0, 0)

    brain_array = np.stack(brain_rows)
    text_array = np.stack(text_rows)

    # Final validation of stacked arrays
    print(f"\n  Final validation for {split_label}:")
    brain_valid = not has_nan_or_inf(brain_array, f"{split_label}_brain_stacked")
    text_valid = not has_nan_or_inf(text_array, f"{split_label}_text_stacked")

    if not brain_valid or not text_valid:
        print(f"  ⚠️ WARNING: NaN/Inf found in stacked arrays for {split_label}!")
        # Clean if necessary
        if not brain_valid:
            brain_array = np.nan_to_num(brain_array, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"  🧹 Cleaned NaN/Inf in {split_label}_brain_stacked")
        if not text_valid:
            text_array = np.nan_to_num(text_array, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"  🧹 Cleaned NaN/Inf in {split_label}_text_stacked")

    # Print statistics
    print(f"\n  {split_label.upper()} Statistics:")
    print(f"    Total samples: {len(pmids)}")
    print(f"    Valid pairs: {len(brain_rows)}")
    print(f"    Skipped: {skipped_count} ({skipped_count/len(pmids)*100:.1f}%)")

    if skipped_pmids:
        print(f"    Skipped samples breakdown:")
        skip_types = {}
        for _, reason in skipped_pmids:
            skip_types[reason] = skip_types.get(reason, 0) + 1
        for reason, count in skip_types.items():
            print(f"      - {reason}: {count}")

    print(f"    Brain embeddings shape: {brain_array.shape}")
    print(f"    Text embeddings shape: {text_array.shape}")

    # Verify alignment
    assert brain_array.shape[0] == text_array.shape[0], \
        f"Alignment mismatch: brain {brain_array.shape[0]} vs text {text_array.shape[0]}"

    return brain_array, text_array


# ── Additional validation after saving ────────────────────────────────────────────

def validate_saved_file(filepath: Path, name: str) -> bool:
    """Validate a saved pickle file contains valid data."""
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        if data is None:
            print(f"  ❌ {name}: file contains None")
            return False

        if len(data) == 0:
            print(f"  ⚠️ {name}: file contains empty array")
            return False

        if has_nan_or_inf(data, name):
            return False

        print(f"  ✅ {name}: valid (shape={data.shape})")
        return True

    except Exception as e:
        print(f"  ❌ {name}: failed to load - {e}")
        return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract mean-pooled ViT embeddings with NaN handling.")
    parser.add_argument("--nii_dir",             required=True,  help="Folder containing pmid_*.nii.gz files")
    parser.add_argument("--seg_checkpoint",      required=True,  help="UNETR segmentation .ckpt")
    parser.add_argument("--brainiac_checkpoint", required=True,  help="BrainIAC SimCLR .ckpt")
    parser.add_argument("--train_pmids",         required=True,  help="Text file with one train PMID per line")
    parser.add_argument("--test_pmids",          required=True,  help="Text file with one test PMID per line")
    parser.add_argument("--train_text_pkl",      required=True,  help="Pickle with train text embeddings [N, D]")
    parser.add_argument("--test_text_pkl",       required=True,  help="Pickle with test text embeddings [N, D]")
    parser.add_argument("--output_dir",          default=".",    help="Where to write the 4 pkl files (default: cwd)")
    parser.add_argument("--skip_nan_validation", action="store_true", help="Skip additional NaN validation (not recommended)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"NaN checking: {'DISABLED' if args.skip_nan_validation else 'ENABLED'}")

    # Load model
    print("\nLoading model...")
    vit = load_vit_backbone(args.seg_checkpoint, args.brainiac_checkpoint, device)
    print("Model loaded successfully")

    # Transforms (identical to original)
    transform = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        Resized(keys=["image"], spatial_size=(96, 96, 96)),
        EnsureTyped(keys=["image"]),
    ])

    # Load PMIDs & text embeddings
    print("\nLoading data...")
    train_pmids = [l.strip() for l in open(args.train_pmids) if l.strip()]
    test_pmids  = [l.strip() for l in open(args.test_pmids)  if l.strip()]

    print(f"  Train PMIDs: {len(train_pmids)}")
    print(f"  Test PMIDs: {len(test_pmids)}")

    with open(args.train_text_pkl, "rb") as f:
        train_text = pickle.load(f)
    with open(args.test_text_pkl,  "rb") as f:
        test_text  = pickle.load(f)

    print(f"  Train text embeddings shape: {train_text.shape if hasattr(train_text, 'shape') else len(train_text)}")
    print(f"  Test text embeddings shape: {test_text.shape if hasattr(test_text, 'shape') else len(test_text)}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Train ──
    print("\n" + "="*60)
    print("TRAIN SET PROCESSING")
    print("="*60)
    brain_train, text_train = process_split(
        train_pmids, train_text, args.nii_dir, vit, transform, device, "train"
    )

    # Save train files
    train_brain_path = out / "brainiac_embeddings_train.pkl"
    train_text_path = out / "text_embeddings_train.pkl"

    with open(train_brain_path, "wb") as f:
        pickle.dump(brain_train, f)
    with open(train_text_path, "wb") as f:
        pickle.dump(text_train, f)

    # ── Test ──
    print("\n" + "="*60)
    print("TEST SET PROCESSING")
    print("="*60)
    brain_test, text_test = process_split(
        test_pmids, test_text, args.nii_dir, vit, transform, device, "test"
    )

    # Save test files
    test_brain_path = out / "brainiac_embeddings_test.pkl"
    test_text_path = out / "text_embeddings_test.pkl"

    with open(test_brain_path, "wb") as f:
        pickle.dump(brain_test, f)
    with open(test_text_path, "wb") as f:
        pickle.dump(text_test, f)

    # ── Final validation ──
    if not args.skip_nan_validation:
        print("\n" + "="*60)
        print("FINAL VALIDATION")
        print("="*60)

        all_valid = True
        for path, name in [
            (train_brain_path, "brain_train"),
            (train_text_path, "text_train"),
            (test_brain_path, "brain_test"),
            (test_text_path, "text_test"),
        ]:
            if not validate_saved_file(path, name):
                all_valid = False

        if all_valid:
            print("\n✅ All files validated successfully - no NaN/Inf detected!")
        else:
            print("\n⚠️ Some files contain NaN/Inf - review skipped samples above")

    # Print final summary
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print(f"Train brain: {brain_train.shape}")
    print(f"Train text:  {text_train.shape}")
    print(f"Test brain:  {brain_test.shape}")
    print(f"Test text:   {text_test.shape}")
    print(f"\n✅ 4 files written to {out.resolve()}")

    # Verify alignment within each split
    assert brain_train.shape[0] == text_train.shape[0], "Train alignment mismatch after extraction!"
    assert brain_test.shape[0] == text_test.shape[0], "Test alignment mismatch after extraction!"
    print("\n✅ Alignment preserved across all splits")


if __name__ == "__main__":
    main()
