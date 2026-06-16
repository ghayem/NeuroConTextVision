"""
main_2d_brainiac_v3.py

Definitive robust version — NaN-free on CPU, text→brain retrieval only.

Loss change:
  Replaced symmetric CLIP loss with a unidirectional TextToBrainLoss.
  At retrieval time the query is always a text and the gallery is brain
  embeddings, so training the brain→text direction wastes gradient signal
  and can hurt t2b recall by splitting model capacity.

Other NaN fixes (unchanged from previous version):
  1. NaNGuard backward hook — zeroes NaN/Inf gradients before optimizer step.
  2. Transformer output clamped post-encoder (not pre), at the CLS token.
  3. logit_scale clamped in log-space (max ln(100) ≈ 4.6052) inside the loss.
  4. WarmupOptimizer — first 2 steps at lr * 0.01 to seed momentum buffers.
  5. AdamW eps=1e-6 for CPU float32 stability.
  6. NaN-safe collate_fn at the DataLoader level.

Visualisation additions:
  - plot_loss_curves()  : train + val loss vs epoch, saved to loss_curves.png
  - plot_similarity_matrices() : text-text, brain-brain, text-brain cosine
    similarity matrices for a random 256-sample subset of each split,
    saved to similarity_matrices_{split}.png
"""

import gc
import os
import sys
import pickle
import glob
import argparse
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

from layers import ClipModel, ProjectionHead
from plotting import plot_matrix
from training import predict, train, count_parameters
from metrics import mix_match
from src.utils import plot_training, recall_n


# ---------------------------------------------------------------------------
# 1. Asymmetric Text→Brain Loss
# ---------------------------------------------------------------------------
class TextToBrainLoss(nn.Module):
    """
    Unidirectional CLIP loss — text→brain direction only.

    training.py calls:  criterion(image_embed, text_embed, logit_scale, logit_bias)
    where image_embed = brain embeddings, text_embed = text embeddings.

    logits   shape: (B_brain, B_text)
    logits.T shape: (B_text,  B_brain)  ← row i = text_i scores over all brains

    We want text_i to rank brain_i highest, so we apply cross-entropy over
    logits.T only.  Dropping the brain→text direction means every gradient
    step is fully dedicated to the retrieval direction used at inference.
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def _stable_log_softmax(self, mat):
        """Row-wise numerically stable log-softmax. mat: (B, B)."""
        m = mat.max(dim=-1, keepdim=True)[0]
        s = mat - m
        return s - torch.log(torch.exp(s).sum(dim=-1, keepdim=True) + self.eps)

    def forward(self, image_embed, text_embed, logit_scale, logit_bias=None):
        device = image_embed.device
        B = image_embed.shape[0]
        labels = torch.arange(B, device=device)

        image_embed = F.normalize(image_embed, dim=-1, eps=1e-8)  # brains
        text_embed  = F.normalize(text_embed,  dim=-1, eps=1e-8)  # texts

        # Cap logit_scale (log-domain) at ln(100) ≈ 4.6052
        scale  = torch.exp(torch.clamp(logit_scale, max=4.6052))
        logits = scale * image_embed @ text_embed.T  # (B_brain, B_text)

        if logit_bias is not None:
            logits = logits + logit_bias

        # logits.T[i] = scores of text_i over all brains → label = i (its paired brain)
        return F.nll_loss(self._stable_log_softmax(logits.T), labels)


# ---------------------------------------------------------------------------
# 2. NaN Guard — zeroes bad gradients before any optimizer step
# ---------------------------------------------------------------------------
def register_nan_guard(model: nn.Module) -> None:
    def _guard(grad):
        if grad is None:
            return grad
        bad = torch.isnan(grad) | torch.isinf(grad)
        if bad.any():
            grad = grad.clone()
            grad[bad] = 0.0
        return grad

    for param in model.parameters():
        if param.requires_grad:
            param.register_hook(_guard)


# ---------------------------------------------------------------------------
# 3. Weight Initialization
# ---------------------------------------------------------------------------
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.MultiheadAttention):
        if hasattr(m, "in_proj_weight") and m.in_proj_weight is not None:
            nn.init.xavier_uniform_(m.in_proj_weight, gain=1.0)
        if hasattr(m, "out_proj") and hasattr(m.out_proj, "weight"):
            nn.init.xavier_uniform_(m.out_proj.weight, gain=1.0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# 4. Dataset with NaN-safe collation
# ---------------------------------------------------------------------------
class BrainPatchDataset(Dataset):
    def __init__(self, folder_path):
        self.file_paths = sorted(glob.glob(os.path.join(folder_path, "*.pkl")))
        if len(self.file_paths) == 0:
            raise FileNotFoundError(f"No .pkl files found in {folder_path}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with open(self.file_paths[idx], "rb") as f:
            data = pickle.load(f)
        brain = torch.from_numpy(data["brain_patches"]).float()
        text  = torch.from_numpy(data["text_embedding"]).float()
        if torch.isnan(brain).any() or torch.isnan(text).any():
            brain = torch.nan_to_num(brain, nan=0.0, posinf=1.0, neginf=-1.0)
            text  = torch.nan_to_num(text,  nan=0.0, posinf=1.0, neginf=-1.0)
        return brain, text


def safe_collate(batch):
    """Drop any sample that still has NaN after per-item cleaning."""
    clean = [
        (b, t) for b, t in batch
        if not (torch.isnan(b).any() or torch.isnan(t).any())
    ]
    if len(clean) == 0:
        clean = batch
    brains = torch.stack([b for b, _ in clean])
    texts  = torch.stack([t for _, t in clean])
    return brains, texts


# ---------------------------------------------------------------------------
# 5. PatchEncoder — output clamped AFTER transformer at the CLS token
# ---------------------------------------------------------------------------
class PatchEncoder(nn.Module):
    def __init__(
        self,
        input_dim=768,
        num_patches=216,
        hidden_dim=512,
        output_dim=512,
        dropout=0.5,
    ):
        super().__init__()
        self.patch_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, hidden_dim) * 0.01
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.01)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            dropout=dropout,
            norm_first=True,  # Pre-norm: more stable than post-norm
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        B, N, _ = x.shape
        x = self.patch_proj(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, : (N + 1)]
        x = self.transformer(x)
        # Clamp the CLS token output before the projection head
        cls_out = torch.clamp(x[:, 0], min=-10.0, max=10.0)
        return self.head(cls_out)


# ---------------------------------------------------------------------------
# 6. Diagnostic
# ---------------------------------------------------------------------------
def run_diagnostic(model, train_loader, device):
    print("\n--- Diagnostic forward pass ---")
    model.eval()
    with torch.no_grad():
        sample_brain, sample_text = next(iter(train_loader))
        sample_brain = sample_brain.to(device)
        sample_text  = sample_text.to(device)

        def stats(name, t):
            has_nan = torch.isnan(t).any().item()
            has_inf = torch.isinf(t).any().item()
            print(
                f"  {name:25s} shape={tuple(t.shape)}  "
                f"min={t.min():.4f}  max={t.max():.4f}  "
                f"nan={has_nan}  inf={has_inf}"
            )
            return has_nan or has_inf

        bad = False
        bad |= stats("input brain_patches", sample_brain)
        bad |= stats("input text_embedding", sample_text)
        if bad:
            raise RuntimeError("NaN/Inf in raw input — fix your .pkl files.")

        img_emb_raw = model.encode_image(sample_brain)
        stats("image_emb (pre-norm)",  img_emb_raw)
        txt_emb_raw = model.encode_text(sample_text)
        stats("text_emb  (pre-norm)",  txt_emb_raw)

        img_emb = F.normalize(img_emb_raw, dim=-1, eps=1e-8)
        txt_emb = F.normalize(txt_emb_raw, dim=-1, eps=1e-8)
        stats("image_emb (post-norm)", img_emb)
        stats("text_emb  (post-norm)", txt_emb)

        scale  = torch.exp(torch.clamp(model.logit_scale, max=4.6052))
        logits = scale * img_emb @ txt_emb.T
        stats("logits", logits)
        print(f"  logit_scale              value={model.logit_scale.item():.4f}  (exp={scale.item():.2f})")

    print("--- Diagnostic complete ---\n")
    model.train()


# ---------------------------------------------------------------------------
# 7. Warm-up optimizer wrapper
# ---------------------------------------------------------------------------
class WarmupOptimizer:
    """Scales LR by `factor` for the first `warmup_steps` steps, then reverts."""
    def __init__(self, optimizer, warmup_steps=2, factor=0.01):
        self.optimizer    = optimizer
        self.warmup_steps = warmup_steps
        self.factor       = factor
        self._step        = 0
        self._base_lrs    = [g["lr"] for g in optimizer.param_groups]

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

    def zero_grad(self, *a, **kw):
        self.optimizer.zero_grad(*a, **kw)

    def step(self, *a, **kw):
        self._step += 1
        if self._step <= self.warmup_steps:
            scale = self.factor + (1.0 - self.factor) * (self._step / self.warmup_steps)
            for g, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
                g["lr"] = base_lr * scale
        else:
            for g, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
                g["lr"] = base_lr
        self.optimizer.step(*a, **kw)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, sd):
        self.optimizer.load_state_dict(sd)


# ---------------------------------------------------------------------------
# 8. Visualisation helpers
# ---------------------------------------------------------------------------
def plot_loss_curves(train_losses, val_losses, output_dir: Path) -> None:
    """
    Plot train and validation loss curves on the same axes and save to disk.

    Parameters
    ----------
    train_losses : list[float]   one value per epoch
    val_losses   : list[float]   one value per epoch (same length)
    output_dir   : Path          directory to write loss_curves.png into
    """
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="Train loss",      color="#2196F3", linewidth=2)
    ax.plot(epochs, val_losses,   label="Validation loss", color="#F44336", linewidth=2,
            linestyle="--")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("TextToBrain Loss", fontsize=12)
    ax.set_title("Training and Validation Loss Curves", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Mark the epoch with the best (lowest) validation loss
    best_epoch = int(np.argmin(val_losses)) + 1
    best_val   = val_losses[best_epoch - 1]
    ax.axvline(best_epoch, color="#4CAF50", linestyle=":", linewidth=1.5,
               label=f"Best val epoch {best_epoch} ({best_val:.4f})")
    ax.legend(fontsize=11)

    out_path = output_dir / "loss_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Loss curves saved to {out_path}")


def plot_similarity_matrices(
    embeddings_by_split: dict,
    output_dir: Path,
    max_samples: int = 256,
) -> None:
    """
    Single figure: one row per split (train / validation / test),
    three columns (text×text, brain×brain, text×brain).
    Called exactly once after all splits are evaluated.

    Parameters
    ----------
    embeddings_by_split : dict  {split_name: (brain_emb, text_emb)}
                          brain_emb / text_emb are np.ndarray (N, D), un-normalised.
    output_dir          : Path  directory to write similarity_matrices.png into
    max_samples         : int   samples per split kept for readability
    """
    splits = [s for s in ["train", "validation", "test"] if s in embeddings_by_split]
    n_rows = len(splits)

    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6 * n_rows), squeeze=False)
    fig.suptitle(
        "Cosine Similarity Matrices  (text×text · brain×brain · text×brain)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    rng = np.random.default_rng(seed=0)

    for row, split_name in enumerate(splits):
        brain_embeddings, text_embeddings = embeddings_by_split[split_name]
        N = min(len(brain_embeddings), len(text_embeddings), max_samples)
        idx = np.sort(rng.choice(len(brain_embeddings), size=N, replace=False))

        B = brain_embeddings[idx]
        T = text_embeddings[idx]

        # L2-normalise
        B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
        T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-8)

        # Each panel: (matrix, title, x-axis label, y-axis label)
        # Matrix convention: rows = y-axis, cols = x-axis.
        # T @ T.T → rows=text,  cols=text
        # B @ B.T → rows=brain, cols=brain
        # T @ B.T → rows=text,  cols=brain  (text→brain retrieval; diagonal = paired match)
        panels = [
            (T @ T.T, "Text → Text",              "Text (query)",  "Text (gallery)"),
            (B @ B.T, "Brain → Brain",             "Brain (query)", "Brain (gallery)"),
            (T @ B.T, "Text → Brain  (retrieval)", "Brain",         "Text (query)"),
        ]

        for col, (mat, title, xlabel, ylabel) in enumerate(panels):
            ax = axes[row][col]
            im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=-1, vmax=1,
                           interpolation="nearest")
            ax.set_title(f"{split_name.upper()} — {title}", fontsize=11)
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = output_dir / "similarity_matrices.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Similarity matrices saved to {out_path}")


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir",    required=True)
    parser.add_argument("--test_dir",     required=True)
    parser.add_argument("--batch_size",   type=int,   default=128)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--num_epochs",   type=int,   default=1)
    parser.add_argument("--output_dim",   type=int,   default=512)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(os.getcwd())
    print(f"🚀 Using device: {device}")

    full_train_dataset = BrainPatchDataset(args.train_dir)
    test_dataset       = BrainPatchDataset(args.test_dir)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=safe_collate,
    )

    validation_size = 1000
    k_fold = KFold(
        n_splits=max(2, len(full_train_dataset) // validation_size),
        shuffle=True,
        random_state=42,
    )
    recall_fn = partial(recall_n, thresh=0.95, reduce_mean=True)
    metrics   = defaultdict(lambda: defaultdict(list))

    # Accumulated loss curves across folds (only fold 0 here, but kept general)
    all_train_losses = []
    all_val_losses   = []

    for fold, (train_idx, val_idx) in enumerate(
        k_fold.split(np.arange(len(full_train_dataset)))
    ):
        if fold >= 1:
            break

        print(f"\n{'=' * 40}\nFOLD {fold}\n{'=' * 40}")
        train_sub = torch.utils.data.Subset(full_train_dataset, train_idx)
        val_sub   = torch.utils.data.Subset(full_train_dataset, val_idx[:validation_size])

        train_loader = DataLoader(
            train_sub,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=safe_collate,
        )
        val_loader = DataLoader(
            val_sub,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=safe_collate,
        )

        model = ClipModel(
            image_model=PatchEncoder(
                input_dim=768,
                output_dim=args.output_dim,
                dropout=0.6,
            ),
            text_model=nn.Sequential(
                ProjectionHead(4096, args.output_dim, dropout=0.6)
            ),
        ).to(device)

        model.apply(init_weights)

        with torch.no_grad():
            model.logit_scale.fill_(2.6593)

        register_nan_guard(model)

        print(f"Model Parameters: {count_parameters(model)}")
        print(f"Initial logit_scale: {model.logit_scale.item():.4f}")

        run_diagnostic(model, train_loader, device)

        base_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            eps=1e-6,
        )
        optimizer = WarmupOptimizer(base_optimizer, warmup_steps=2, factor=0.01)
        criterion = TextToBrainLoss(eps=1e-6)

        clip_model, clip_train_loss, clip_val_loss, callback_outputs = train(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=None,
            criterion=criterion,
            num_epochs=args.num_epochs,
            device=device,
            verbose=True,
            output_dir=output_dir,
            callbacks=[],
            clip_grad_norm=1.0,
        )

        # Accumulate loss histories returned by train()
        # clip_train_loss and clip_val_loss are lists of per-epoch mean losses
        all_train_losses.extend(clip_train_loss)
        all_val_losses.extend(clip_val_loss)

        # -----------------------------------------------------------------
        # Evaluation — collect embeddings from all three splits
        # -----------------------------------------------------------------
        # embeddings_by_split holds raw (un-normalised) arrays for the
        # single combined similarity figure drawn after the loop.
        embeddings_by_split: dict = {}

        for loader_name, loader, weights_path in [
            (
                "train",
                DataLoader(
                    torch.utils.data.Subset(train_sub, np.arange(1000)),
                    batch_size=args.batch_size,
                    collate_fn=safe_collate,
                ),
                output_dir / "last.pt",
            ),
            ("validation", val_loader, output_dir / "best_val.pt"),
            ("test",       test_loader, output_dir / "best_val.pt"),
        ]:
            if not weights_path.exists():
                continue

            print(f"\nEvaluating {loader_name}...")
            clip_model.load_state_dict(torch.load(weights_path, map_location=device))

            # predict() returns (brain_embeddings, text_embeddings) as np arrays
            brain_embeddings, text_embeddings = predict(clip_model, loader, device=device)

            # Store raw embeddings for the combined similarity figure
            embeddings_by_split[loader_name] = (brain_embeddings, text_embeddings)

            # text→brain cosine similarity for retrieval metrics
            text_emb_n  = text_embeddings  / (np.linalg.norm(text_embeddings,  axis=1, keepdims=True) + 1e-8)
            brain_emb_n = brain_embeddings / (np.linalg.norm(brain_embeddings, axis=1, keepdims=True) + 1e-8)
            similarity  = text_emb_n @ brain_emb_n.T  # ndarray @ ndarray = ndarray
            similarity  = torch.from_numpy(np.array(similarity)).softmax(dim=1).numpy()

            n = len(similarity)
            metrics[loader_name]["recall@10"].append(
                recall_fn(similarity, np.eye(n), n_first=10)
            )
            metrics[loader_name]["recall@100"].append(
                recall_fn(similarity, np.eye(n), n_first=100)
            )
            metrics[loader_name]["mix_match"].append(100 * mix_match(similarity))

    # -------------------------------------------------------------------------
    # Loss curves + similarity matrices — both plotted exactly once
    # -------------------------------------------------------------------------
    print("\nPlotting loss curves...")
    if all_train_losses and all_val_losses:
        plot_loss_curves(all_train_losses, all_val_losses, output_dir)
    else:
        print("  (no loss data available — training.py may not return loss lists)")

    print("\nPlotting similarity matrices...")
    if embeddings_by_split:
        plot_similarity_matrices(embeddings_by_split, output_dir, max_samples=256)
    else:
        print("  (no embeddings collected — skipping)")

    # -------------------------------------------------------------------------
    # Final metrics summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 30 + "\nFINAL METRICS\n" + "=" * 30)
    for loader_name in ["train", "validation", "test"]:
        print(f"--- {loader_name.upper()} ---")
        for metric_name, values in metrics[loader_name].items():
            if values:
                print(f"  {metric_name}: {np.mean(values):.3f} +- {np.std(values):.3f}")


if __name__ == "__main__":
    main()
