"""
Calculate dice statistics of all your NIFTI files and create a csv 
"""

import os
import csv
import pickle
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

# ==========================================
# CONFIG
# ==========================================
EMBEDDINGS_TEST_DIR = "your/embeddings" #Put your embeddings
KDE_DIR             = "/your/KDE" #put your KDE
OUTPUT_CSV          = "./results/dice_results.csv"
BINARIZE_THRESHOLD  = 0.8   #you can change your threshold [0,1]

# ==========================================
# HELPERS
# ==========================================
def binarize(path: str, threshold: float = BINARIZE_THRESHOLD) -> np.ndarray:
    """Charge a NIFTI, calculating z-score and binarize him using threshold"""
    img  = nib.load(path)
    data = img.get_fdata()
    std  = data.std()
    if std < 1e-8:
        return np.zeros(data.shape, dtype=np.uint8)
    data_z = (data - data.mean()) / std
    return (data_z > threshold).astype(np.uint8)

def dice_score(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Dice between 2 kde"""
    intersection = np.logical_and(mask1, mask2).sum()
    total = mask1.sum() + mask2.sum()
    if total == 0:
        return 0.0
    return float(2.0 * intersection / total)

# ==========================================
# 1. INITIALISATION
# ==========================================
test_pkl_files = sorted(Path(EMBEDDINGS_TEST_DIR).glob("*.pkl"))
if not test_pkl_files:
    raise FileNotFoundError(f"No .pkl found in {EMBEDDINGS_TEST_DIR}")

test_pmids = []
for pkl_path in test_pkl_files:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    pmid = str(data["pmid"])
    test_pmids.append(pmid)

print(f"✅ {len(test_pmids)} PMIDs found in {EMBEDDINGS_TEST_DIR}")

# ==========================================
# 2. FIND ALL KDE PATHS
# ==========================================
all_kde_paths = {
    p.name.replace("pmid_", "").replace(".nii.gz", ""): str(p)
    for p in Path(KDE_DIR).glob("pmid_*.nii.gz")
}

if not all_kde_paths:
    raise FileNotFoundError(f"Aucun pmid_*.nii.gz trouvé dans {KDE_DIR}")

print(f"✅ {len(all_kde_paths)} NIfTI disponibles dans {KDE_DIR}")

# Search all kde available with the same pmid as your embeddings
valid_test_pmids = [p for p in test_pmids if p in all_kde_paths]
missing = set(test_pmids) - set(valid_test_pmids)
if missing:
    print(f"⚠️  {len(missing)} PMIDs test sans NIfTI correspondant (ignorés) : {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
print(f"✅ {len(valid_test_pmids)} PMIDs test avec NIfTI trouvé\n")

# ==========================================
# 3. Pre-load all masks
# ==========================================
print("Loading NIFTI Masks...")
mask_cache: dict[str, np.ndarray] = {}

all_pmids_needed = set(valid_test_pmids)
for pmid in tqdm(all_pmids_needed, desc="Reading NIfTI"):
    if pmid in all_kde_paths:
        try:
            mask_cache[pmid] = binarize(all_kde_paths[pmid])
        except Exception as e:
            print(f"  ⚠️  Error reading pmid_{pmid}.nii.gz : {e}")

print(f"✅ {len(mask_cache)} masques en mémoire\n")

# ==========================================
# 4. Calculate all queries (1 pmid and compare with all others)
# ==========================================
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
results = []

print("Calcul des Dice scores...")
for query_pmid in tqdm(valid_test_pmids, desc="Queries"):
    if query_pmid not in mask_cache:
        continue

    query_mask = mask_cache[query_pmid]
    scores: list[tuple[str, float]] = []   # (pmid, dice)

    for candidate_pmid, candidate_mask in mask_cache.items():
        if candidate_pmid == query_pmid:
            continue  
        d = dice_score(query_mask, candidate_mask)
        scores.append((candidate_pmid, d))

    if not scores:
        continue

    scores.sort(key=lambda x: x[1], reverse=True)

    top3   = scores[:3]
    worst  = scores[-1]

    # In case there is less than 3 candidates 
    while len(top3) < 3:
        top3.append(("N/A", float("nan")))

    results.append({
        "query_pmid":  query_pmid,
        "top1_pmid":   top3[0][0], "top1_dice":  round(top3[0][1], 6),
        "top2_pmid":   top3[1][0], "top2_dice":  round(top3[1][1], 6),
        "top3_pmid":   top3[2][0], "top3_dice":  round(top3[2][1], 6),
        "worst_pmid":  worst[0],   "worst_dice": round(worst[1], 6),
    })

# ==========================================
# 5. Save csv
# ==========================================
fieldnames = [
    "query_pmid",
    "top1_pmid", "top1_dice",
    "top2_pmid", "top2_dice",
    "top3_pmid", "top3_dice",
    "worst_pmid", "worst_dice",
]

with open(OUTPUT_CSV, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\n✅ CSV saved : {OUTPUT_CSV}")
print(f"   {len(results)} lines has been writte")

# ==========================================
# 6. Statitics
# ==========================================
if results:
    top1_dices = [r["top1_dice"] for r in results if not np.isnan(r["top1_dice"])]
    worst_dices = [r["worst_dice"] for r in results if not np.isnan(r["worst_dice"])]
    print(f"\n Statistics top-1 Dice :")
    print(f"   mean  = {np.mean(top1_dices):.4f}")
    print(f"   median= {np.median(top1_dices):.4f}")
    print(f"   max   = {np.max(top1_dices):.4f}")
    print(f"   min   = {np.min(top1_dices):.4f}")
    print(f"\n Statistics worst Dice :")
    print(f"   mean  = {np.mean(worst_dices):.4f}")
    print(f"   median= {np.median(worst_dices):.4f}")