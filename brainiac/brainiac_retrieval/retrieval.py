"""
Visualize top-3 queries (using CSV created).
"""

import os
import re
import csv
import tempfile
import requests
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from nilearn import plotting
from pathlib import Path
from xml.etree import ElementTree as ET

# ==========================================
# CONFIG
# ==========================================
CSV_PATH   = "./results/dice_results.csv"
KDE_DIR    = "your/KDE" #path to kde
OUTPUT_DIR = "./results"
THRESHOLD  = 0.8   #[0,1]

# list of brain regions
BRAIN_REGIONS = [
    "prefrontal", "frontal", "parietal", "temporal", "occipital",
    "hippocampus", "hippocampal", "amygdala", "insula", "cingulate",
    "anterior cingulate", "posterior cingulate", "thalamus", "thalamic",
    "striatum", "striatal", "caudate", "putamen", "nucleus accumbens",
    "cerebellum", "cerebellar", "brainstem", "brain stem",
    "motor cortex", "somatosensory", "visual cortex", "auditory cortex",
    "default mode", "dmn", "dlpfc", "vlpfc", "ofc", "orbitofrontal",
    "parahippocampal", "entorhinal", "fusiform", "precuneus", "cuneus",
    "angular gyrus", "supramarginal", "broca", "wernicke",
    "superior temporal", "inferior temporal", "middle temporal",
    "superior frontal", "inferior frontal", "middle frontal",
    "superior parietal", "inferior parietal",
    "lateral", "medial", "dorsal", "ventral", "bilateral",
    "left hemisphere", "right hemisphere", "corpus callosum",
    "white matter", "gray matter", "grey matter", "cortex", "subcortical",
]

# ==========================================
# HELPERS
# ==========================================
def normalize_img(path):
    img  = nib.load(path)
    data = img.get_fdata()
    data = (data - data.mean()) / (data.std() + 1e-8)
    return nib.Nifti1Image(data, img.affine)

def binarize(path, threshold=THRESHOLD):
    img  = nib.load(path)
    data = img.get_fdata()
    std  = data.std()
    if std < 1e-8:
        return np.zeros(data.shape, dtype=np.uint8)
    data_z = (data - data.mean()) / std
    return (data_z > threshold).astype(np.uint8)

def dice_score(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    total = mask1.sum() + mask2.sum()
    if total == 0:
        return 0.0
    return float(2.0 * intersection / total)

def kde_path(pmid):
    return os.path.join(KDE_DIR, f"pmid_{pmid}.nii.gz")

def render_kde_to_img(path):
    """Render KDE to img"""
    img_norm = normalize_img(path)
    plotting.plot_img_on_surf(
        img_norm,
        cmap="bwr",
        vmax=6,
        views=["lateral"],
        colorbar=False,
        alpha=1.0,
        bg_on_data=True,
    )
    fig = plt.gcf()
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=100, bbox_inches="tight")
    plt.close("all")
    arr = mpimg.imread(tmp.name)
    os.unlink(tmp.name)
    return arr

def get_abstract(pmid, max_chars=1500):
    try:
        url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            f"?db=pubmed&id={pmid}&rettype=xml&retmode=xml"
        )
        r = requests.get(url, timeout=10)
        text = r.text.encode("utf-8", errors="replace").decode("utf-8")
        tree = ET.fromstring(text)
        abstract_texts = tree.findall(".//AbstractText")
        if abstract_texts:
            return " ".join([a.text or "" for a in abstract_texts])[:max_chars]
        return ""
    except Exception:
        try:
            matches = re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", r.text, re.DOTALL)
            if matches:
                return " ".join(matches)[:max_chars]
        except Exception:
            pass
        return ""

def extract_brain_regions(abstract):
    """Regions of the brain find in the article"""
    if not abstract:
        return ["N/A"]
    abstract_lower = abstract.lower()
    found = []
    for region in BRAIN_REGIONS:
        if region in abstract_lower and region not in found:
            found.append(region)
    if not found:
        return ["N/A"]
    return found

def wrap_regions(regions, max_per_line=3):
    """Format for the print of brains regions"""
    lines = []
    for i in range(0, len(regions), max_per_line):
        lines.append(", ".join(regions[i:i+max_per_line]))
    return "\n".join(lines)

# ==========================================
# 1. Read Csv
# ==========================================
import pandas as pd

df = pd.read_csv(CSV_PATH)
for col in ["query_pmid", "top1_pmid", "top2_pmid", "top3_pmid", "worst_pmid"]:
    df[col] = df[col].apply(lambda x: str(int(float(x))) if pd.notna(x) else x)

df_valid = df[df["query_pmid"].apply(lambda p: os.path.exists(kde_path(p)))]
top3_queries = df_valid.nlargest(3, "top1_dice").to_dict("records")

print(f"✅ Top-3 queries selected :")
for r in top3_queries:
    print(f"   query={r['query_pmid']} | top1_dice={float(r['top1_dice']):.4f}")

# ==========================================
# 2. Generate figures
# ==========================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

for row in top3_queries:
    query_pmid = row["query_pmid"]
    matches = [
        (row["top1_pmid"], float(row["top1_dice"]), "Top-1"),
        (row["top2_pmid"], float(row["top2_dice"]), "Top-2"),
        (row["top3_pmid"], float(row["top3_dice"]), "Top-3"),
    ]

    all_cols = [(query_pmid, None, "Query")] + matches

    print(f"\n🔍 Rendu pour query PMID {query_pmid}")

    q_path = kde_path(query_pmid)
    query_mask = binarize(q_path) if os.path.exists(q_path) else None

    col_data = []
    for pmid, dice_val, label in all_cols:
        path = kde_path(pmid)
        if not os.path.exists(path):
            print(f"  ⚠️  NIfTI introuvable : {path}")
            col_data.append({"pmid": pmid, "label": label, "img": None,
                             "dice": dice_val, "regions": ["N/A"]})
            continue

        if query_mask is not None and dice_val is not None:
            cand_mask = binarize(path)
            dice_real = dice_score(query_mask, cand_mask)
        else:
            dice_real = None

        print(f"  Rendering {label} (PMID {pmid})...")
        img_arr = render_kde_to_img(path)

        print(f"  Fetching abstract PMID {pmid}...")
        abstract = get_abstract(pmid)
        regions  = extract_brain_regions(abstract)

        col_data.append({
            "pmid":    pmid,
            "label":   label,
            "img":     img_arr,
            "dice":    dice_real,
            "regions": regions,
        })

    n_cols = len(col_data)
    fig = plt.figure(figsize=(6 * n_cols, 9))
    fig.patch.set_facecolor("white")

    col_w   = 1.0 / n_cols
    img_top = 0.30   
    img_h   = 0.58   

    for i, col in enumerate(col_data):
        x0 = i * col_w + 0.01
        w  = col_w - 0.02

        # --- Title---
        if col["dice"] is not None:
            title_str = f"Dice = {col['dice']:.3f}"
            title_color = plt.cm.RdYlGn(col["dice"])  #Scaling of colors of dice
        else:
            title_str  = "Query (original)"
            title_color = "white"

        fig.text(
            x0 + w / 2, img_top + img_h + 0.02,
            title_str,
            ha="center", va="bottom",
            fontsize=13, fontweight="bold",
            color=title_color,
        )

        ax_img = fig.add_axes([x0, img_top, w, img_h])
        if col["img"] is not None:
            ax_img.imshow(col["img"])
        else:
            ax_img.text(0.5, 0.5, "NIfTI\nintrouvable",
                        ha="center", va="center", color="black", fontsize=10)
            ax_img.set_facecolor("#eeeeee")
        ax_img.axis("off")

        # --- PMID ---
        fig.text(
            x0 + w / 2, img_top - 0.04,
            f"PMID: {col['pmid']}",
            ha="center", va="top",
            fontsize=10, color="#333333",
        )

        regions_str = wrap_regions(col["regions"], max_per_line=3)
        fig.text(
            x0 + w / 2, img_top - 0.09,
            regions_str,
            ha="center", va="top",
            fontsize=7.5, color="#1a5fa8",
            linespacing=1.5,
            wrap=True,
        )

    fig.suptitle(
        f"Dice KDE — Query PMID: {query_pmid}",
        fontsize=15, fontweight="bold", color="black", y=0.97,
    )

    save_path = os.path.join(OUTPUT_DIR, f"dice_viz_{query_pmid}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close("all")
    print(f"Saved in : {save_path}")

print(f"\n✅ Figures generate in {OUTPUT_DIR}/")