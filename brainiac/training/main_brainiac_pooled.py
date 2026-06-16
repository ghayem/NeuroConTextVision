"""
neurocontext_brainiac.py

Identical to the original NeuroConText training script except that:
  - DiFuMo / Gaussian embeddings are replaced by BrainIAC embeddings.
  - Text embeddings come from the aligned files produced by
    extract_brainiac_embeddings.py.
  - The four source files are loaded from a hard-coded path:
        <this_file's_parent> / new_and_aligned_embeddings /
  - Includes NaN/Inf detection and removal to preserve alignment.

No command-line arguments are needed for data paths.
All other hyper-parameters (lr, batch_size, etc.) remain identical to the
original script and can be edited in the "Training configuration" section.
"""

import gc
import os
import sys
import pickle
from collections import defaultdict
from functools import partial
from pathlib import Path

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

# Local module imports
from layers import ClipModel, ProjectionHead, ResidualHead
from losses import ClipLoss
from plotting import plot_matrix
from training import predict, train, count_parameters
from metrics import mix_match
from src.utils import plot_training, recall_n

# ---------------------------------------------------------------------------
# Setup environment
# ---------------------------------------------------------------------------
current_folder_path = os.getcwd()
parent_folder_path = os.path.dirname(current_folder_path)

if current_folder_path not in sys.path:
    sys.path.append(current_folder_path)
if parent_folder_path not in sys.path:
    sys.path.append(parent_folder_path)

os.chdir(current_folder_path)
print("Current Working Directory:", os.getcwd())

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
BASE_OUTPUT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_OUTPUT_DIR / "plots"
SIM_MATRICES_DIR = PLOTS_DIR / "similarity_matrices"
LOSS_CURVES_DIR = PLOTS_DIR / "loss_curves"

PLOTS_DIR.mkdir(exist_ok=True)
SIM_MATRICES_DIR.mkdir(exist_ok=True)
LOSS_CURVES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# NaN detection and removal functions
# ---------------------------------------------------------------------------
def remove_nan_pairs(brain_data, text_data, name_prefix):
    """Remove rows where EITHER brain or text embedding has NaN/Inf.
    Returns aligned pair with invalid rows removed."""

    print(f"\nChecking {name_prefix} for NaN/Inf values...")

    # Check each array individually
    brain_has_nan = np.isnan(brain_data).any(axis=1)
    brain_has_inf = np.isinf(brain_data).any(axis=1)
    text_has_nan = np.isnan(text_data).any(axis=1)
    text_has_inf = np.isinf(text_data).any(axis=1)

    brain_invalid = brain_has_nan | brain_has_inf
    text_invalid = text_has_nan | text_has_inf

    # Combine: remove if EITHER is invalid
    either_invalid = brain_invalid | text_invalid
    num_invalid = either_invalid.sum()

    if num_invalid > 0:
        print(f"⚠️ Found {num_invalid} samples with NaN/Inf in {name_prefix}:")
        print(f"   - Brain only invalid: {(brain_invalid & ~text_invalid).sum()}")
        print(f"   - Text only invalid: {(~brain_invalid & text_invalid).sum()}")
        print(f"   - Both invalid: {(brain_invalid & text_invalid).sum()}")

        # Remove invalid pairs
        valid_mask = ~either_invalid
        cleaned_brain = brain_data[valid_mask]
        cleaned_text = text_data[valid_mask]

        print(f"✅ Removed {num_invalid} samples. Remaining: {cleaned_brain.shape[0]}")
        return cleaned_brain, cleaned_text
    else:
        print(f"✅ {name_prefix}: No NaN/Inf pairs found")
        return brain_data, text_data

# ---------------------------------------------------------------------------
# Load aligned BrainIAC + text embeddings
# ---------------------------------------------------------------------------
ALIGNED_DIR = Path(__file__).resolve().parent / "new_and_aligned_embeddings"

if not ALIGNED_DIR.exists():
    raise FileNotFoundError(
        f"Aligned embeddings directory not found: {ALIGNED_DIR}\n"
        "Run extract_brainiac_embeddings.py first."
    )

_files = {
    # variable name in this script   →   filename inside ALIGNED_DIR
    "brainiac_train": "brainiac_embeddings_train.pkl",
    "text_train": "text_embeddings_train.pkl",
    "brainiac_test": "brainiac_embeddings_test.pkl",
    "text_test": "text_embeddings_test.pkl",
}

print(f"Loading aligned embeddings from {ALIGNED_DIR} …")
_loaded = {}
for var, fname in _files.items():
    fpath = ALIGNED_DIR / fname
    if not fpath.exists():
        raise FileNotFoundError(f"Missing file: {fpath}")
    print(f"  📦 {fname}")
    with open(fpath, "rb") as _f:
        _loaded[var] = pickle.load(_f)
    gc.collect()

# ---------------------------------------------------------------------------
# Remove NaN/Inf rows from embeddings (preserve alignment)
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("REMOVING NaN/Inf VALUES FROM EMBEDDINGS")
print("="*60)

# Clean train pairs
brain_train, text_train = remove_nan_pairs(
    _loaded["brainiac_train"],
    _loaded["text_train"],
    "TRAIN"
)

# Clean test pairs
brain_test, text_test = remove_nan_pairs(
    _loaded["brainiac_test"],
    _loaded["text_test"],
    "TEST"
)

del _loaded
gc.collect()

# ---------------------------------------------------------------------------
# Assign to friendly variable names
# ---------------------------------------------------------------------------
preprocessed_train_gaussian_embeddings = brain_train
preprocessed_train_text_embeddings = text_train
preprocessed_test_gaussian_embeddings = brain_test
preprocessed_test_text_embeddings = text_test

print("\n" + "="*60)
print("FINAL DATA SUMMARY")
print("="*60)
print(f"Train brain shape: {preprocessed_train_gaussian_embeddings.shape}")
print(f"Train text shape:  {preprocessed_train_text_embeddings.shape}")
print(f"Test brain shape:  {preprocessed_test_gaussian_embeddings.shape}")
print(f"Test text shape:   {preprocessed_test_text_embeddings.shape}")

# Verify alignment
assert preprocessed_train_gaussian_embeddings.shape[0] == preprocessed_train_text_embeddings.shape[0], \
    f"Train alignment mismatch: {preprocessed_train_gaussian_embeddings.shape[0]} vs {preprocessed_train_text_embeddings.shape[0]}"

assert preprocessed_test_gaussian_embeddings.shape[0] == preprocessed_test_text_embeddings.shape[0], \
    f"Test alignment mismatch: {preprocessed_test_gaussian_embeddings.shape[0]} vs {preprocessed_test_text_embeddings.shape[0]}"

# Final NaN check
print("\nFinal NaN validation:")
assert not np.isnan(preprocessed_train_gaussian_embeddings).any(), "Train brain still has NaN!"
assert not np.isnan(preprocessed_train_text_embeddings).any(), "Train text still has NaN!"
assert not np.isnan(preprocessed_test_gaussian_embeddings).any(), "Test brain still has NaN!"
assert not np.isnan(preprocessed_test_text_embeddings).any(), "Test text still has NaN!"
print("✅ No NaN values remain in any dataset")

print("✅ Embeddings loaded, cleaned, and aligned.")

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
plot_verbose = True
batch_size = 128
lr = 1e-4
weight_decay = 0.1
dropout = 0.6
num_epochs = 50

output_size = 512
# output_size = preprocessed_test_gaussian_embeddings.shape[1]

device = "cuda" if torch.cuda.is_available() else "cpu"

criterion = ClipLoss()
is_clip_loss = isinstance(criterion, ClipLoss)

loss_specific_kwargs = {
    "logit_scale": 10 if is_clip_loss else np.log(10),
    "logit_bias": None if is_clip_loss else -10,
}

# ---------------------------------------------------------------------------
# Test DataLoader
# ---------------------------------------------------------------------------
test_dataset = TensorDataset(
    torch.from_numpy(preprocessed_test_gaussian_embeddings).float(),
    torch.from_numpy(preprocessed_test_text_embeddings).float(),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
)

# ---------------------------------------------------------------------------
# K-fold cross-validation + training loop
# ---------------------------------------------------------------------------
recall_fn = partial(recall_n, thresh=0.95, reduce_mean=True)
validation_size = 1000

# Ensure we have enough samples for KFold
n_samples = len(preprocessed_train_text_embeddings)
n_splits = max(1, n_samples // validation_size)
k_fold = KFold(n_splits=n_splits)

metrics = {
    "train": defaultdict(list),
    "validation": defaultdict(list),
    "test": defaultdict(list),
}

number_of_folds_to_run = 1

for fold, (train_index, val_index) in enumerate(
    k_fold.split(preprocessed_train_text_embeddings)
):
    val_index = val_index[:validation_size]

    if fold >= number_of_folds_to_run:
        break

    print(f"\n{'=' * 60}")
    print(f"FOLD {fold}")
    print(f"{'=' * 60}")

    train_dataset = TensorDataset(
        torch.from_numpy(preprocessed_train_gaussian_embeddings[train_index]).float(),
        torch.from_numpy(preprocessed_train_text_embeddings[train_index]).float(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_dataset = TensorDataset(
        torch.from_numpy(preprocessed_train_gaussian_embeddings[val_index]).float(),
        torch.from_numpy(preprocessed_train_text_embeddings[val_index]).float(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # ----------------------------------------------------------------
    # Model definition
    # ----------------------------------------------------------------
    model = ClipModel(
        image_model=nn.Sequential(
            ProjectionHead(
                preprocessed_train_gaussian_embeddings.shape[1],
                output_size,
                dropout=dropout,
            ),
            ResidualHead(output_size, dropout=dropout),
            ResidualHead(output_size, dropout=dropout),
        ),
        text_model=nn.Sequential(
            ProjectionHead(
                preprocessed_train_text_embeddings.shape[1],
                output_size,
                dropout=dropout,
            ),
            ResidualHead(output_size, dropout=dropout),
            ResidualHead(output_size, dropout=dropout),
        ),
        **loss_specific_kwargs,
    )

    print(f"Parameters: {count_parameters(model)}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = None
    output_dir = BASE_OUTPUT_DIR

    clip_model, clip_train_loss, clip_val_loss, callback_outputs = train(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        num_epochs=num_epochs,
        device=device,
        verbose=True,
        output_dir=output_dir,
        callbacks=[],
    )

    # ----------------------------------------------------------------
    # Save explicit loss curves
    # ----------------------------------------------------------------
    loss_curve_path = LOSS_CURVES_DIR / f"fold_{fold}_loss_curves.png"

    plt.figure(figsize=(10, 6))

    plt.plot(
        clip_train_loss,
        label="Train Loss",
        linewidth=2,
    )

    plt.plot(
        clip_val_loss,
        label="Validation Loss",
        linewidth=2,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training vs Validation Loss — Fold {fold}")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(loss_curve_path, dpi=300)
    plt.close()

    print(f"📉 Saved loss curves to: {loss_curve_path}")

    # ----------------------------------------------------------------
    # Optional training plots from original utilities
    # ----------------------------------------------------------------
    if plot_verbose:
        callback_plot_kwargs = [
            {
                "ylabel": "Validation\nRecall@10",
                "color": "b",
                "ylim": [0, 1],
            },
            {
                "ylabel": "Diagonal Mean",
                "color": "b",
                "ylim": [1e-7, 1],
                "yscale": "log",
            },
            {
                "ylabel": "Non-diagonal Mean",
                "color": "b",
                "ylim": [1e-7, 1],
                "yscale": "log",
            },
            {
                "ylabel": "Logit scale",
                "color": "black",
            },
            {
                "ylabel": "Logit bias",
                "color": "black",
            },
        ]

        plot_training(
            clip_train_loss,
            clip_val_loss,
            callback_outputs,
            callback_kwargs=callback_plot_kwargs,
        )

    # ----------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------
    small_train_dataset = TensorDataset(
        torch.from_numpy(
            preprocessed_train_gaussian_embeddings[train_index][:1000]
        ).float(),
        torch.from_numpy(
            preprocessed_train_text_embeddings[train_index][:1000]
        ).float(),
    )

    small_train_loader = DataLoader(
        small_train_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    for loader_name, loader, weights_path in [
        ("train", small_train_loader, output_dir / "last.pt"),
        ("validation", val_loader, output_dir / "best_val.pt"),
        ("test", test_loader, output_dir / "best_val.pt"),
    ]:
        print(f"\nEvaluating on {loader_name} set...")

        clip_model.load_state_dict(torch.load(weights_path))

        image_embeddings, text_embeddings = predict(
            clip_model,
            loader,
            device=device,
        )

        similarity = (image_embeddings @ text_embeddings.T).softmax(dim=1).numpy()

        # ------------------------------------------------------------
        # Save similarity matrices
        # ------------------------------------------------------------
        brain_to_brain = (image_embeddings @ image_embeddings.T).numpy()[:100, :100]

        text_to_text = (text_embeddings @ text_embeddings.T).numpy()[:100, :100]

        brain_to_text = similarity[:100, :100]

        fig, axes = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=(18, 6),
        )

        plot_matrix(
            brain_to_brain,
            ax=axes[0],
            title="Brain-to-Brain Similarity",
        )

        plot_matrix(
            text_to_text,
            ax=axes[1],
            title="Text-to-Text Similarity",
        )

        plot_matrix(
            brain_to_text,
            ax=axes[2],
            title="Brain-to-Text Similarity",
        )

        fig.suptitle(
            f"Similarity Matrices — Fold {fold} — {loader_name.upper()}",
            fontsize=16,
            fontweight="bold",
        )

        plt.tight_layout()

        sim_plot_path = (
            SIM_MATRICES_DIR / f"fold_{fold}_{loader_name}_similarity_matrices.png"
        )

        plt.savefig(sim_plot_path, dpi=300)
        plt.close()

        print(f"🧠 Saved similarity matrices to: {sim_plot_path}")

        # ------------------------------------------------------------
        # Metrics
        # ------------------------------------------------------------
        metrics[loader_name]["recall@10"].append(
            recall_fn(
                similarity,
                np.eye(len(similarity)),
                n_first=10,
            )
        )

        metrics[loader_name]["recall@100"].append(
            recall_fn(
                similarity,
                np.eye(len(similarity)),
                n_first=100,
            )
        )

        metrics[loader_name]["mix_match"].append(100 * mix_match(similarity))

# ---------------------------------------------------------------------------
# Final metrics summary
# ---------------------------------------------------------------------------
print(f"\nMetrics after {fold} fold(s)")

for loader_name in ["train", "validation", "test"]:
    print("=" * 10, loader_name, "=" * 10)

    for metric_name in [
        "recall@10",
        "recall@100",
        "mix_match",
    ]:
        val_mean = np.mean(metrics[loader_name][metric_name])
        val_std = np.std(metrics[loader_name][metric_name])

        print(f"{metric_name}: {val_mean:.3f} +- {val_std:.3f}")

print("\n✅ All plots saved successfully.")
print(f"📁 Plots directory: {PLOTS_DIR}")
