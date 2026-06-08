# BrainIAC — Text-to-Brain Retrieval Pipeline

End-to-end pipeline for neuroimaging retrieval: given a text query, find the most similar brain activation maps using a trained Transformer encoder and Dice score matching on KDE maps.

---

## Pipeline Overview

```
Trained model (.pt)
       │
       ▼
[Text → Brain embeddings]   →   embeddings/pmid_*.pkl
       │
       ▼
create_csv.py               →   results/dice_results.csv
       │
       ▼
retrieval.py                →   results/dice_viz_<pmid>.png
```

---

## Project Structure

```
project/
├── model/
│   └── final_model.pt               # Trained PatchEncoder weights
├── embeddings/
│   └── pmid_<pmid>.pkl              # Precomputed embeddings (one per article)
├── KDE/
│   └── pmid_<pmid>.nii.gz           # KDE brain maps (NIfTI)
├── results/
│   ├── dice_results.csv             # Computed Dice scores (output of compute_dice.py)
│   └── dice_viz_<pmid>.png          # Figures (output of visualize_top3.py)
├── model.py                         # PatchEncoder architecture
├── create_csv.py                    # Step 1 — compute Dice scores
└── retrieval.py                     # Step 2 — generate figures
```

---

## Requirements

### Python ≥ 3.8

```bash
pip install torch nibabel nilearn matplotlib numpy pandas requests tqdm
```

| Package      | Usage                                   |
|--------------|-----------------------------------------|
| `torch`      | Loading and running the PatchEncoder    |
| `nibabel`    | Reading/writing NIfTI files             |
| `nilearn`    | Projecting KDE maps onto brain surface  |
| `matplotlib` | Figure generation                       |
| `numpy`      | Mask and array computations             |
| `pandas`     | CSV reading                             |
| `requests`   | NCBI PubMed API calls                   |
| `tqdm`       | Progress bars                           |

---

## Model

### Architecture — `PatchEncoder`

The model is a Transformer-based encoder that maps brain patch embeddings to a latent space.

| Parameter     | Value  | Description                              |
|---------------|--------|------------------------------------------|
| `input_dim`   | 768    | Input dimension per patch                |
| `num_patches` | 216    | Number of brain patches                  |
| `hidden_dim`  | 512    | Internal Transformer dimension           |
| `output_dim`  | 512    | Final embedding dimension                |
| `dropout`     | 0.3    | Dropout rate                             |
| `nhead`       | 8      | Attention heads                          |
| `num_layers`  | 2      | Transformer encoder layers               |

**Forward pass:** `[B, 216, 768]` → CLS token prepended → Transformer → `[B, 512]`

### Loading the model

Download the pretrained weights and place them in the `model/` folder:

📥 **[Download model weights (final_model.pt)](https://drive.google.com/uc?export=download&id=1jHIBic2bH94cH4aCuHtaAQzky-xMMtTg)**

```
model/
└── final_model.pt
```

Load the model in Python:

```python
import torch
from model import PatchEncoder

model = PatchEncoder()
model.load_state_dict(torch.load("model/final_model.pt", map_location="cpu"))
model.eval()
```

---

## Step 1 — Compute Dice Scores (`create_csv.py`)

Reads precomputed embeddings (`.pkl`), loads the corresponding KDE NIfTI maps, binarizes them, and computes pairwise Dice scores for all queries.

### Configuration

```python
EMBEDDINGS_TEST_DIR = "your/embeddings"   # Folder with .pkl embedding files
KDE_DIR             = "your/KDE"          # Folder with pmid_*.nii.gz files
OUTPUT_CSV          = "./results/dice_results.csv"
BINARIZE_THRESHOLD  = 0.8                 # Z-score binarization threshold [0, 1]
```

### Run

```bash
python compute_dice.py
```

### Output — `results/dice_results.csv`

| Column       | Description                        |
|--------------|------------------------------------|
| `query_pmid` | PMID of the query article          |
| `top1_pmid`  | PMID of the best match             |
| `top1_dice`  | Dice score with top-1              |
| `top2_pmid`  | PMID of the 2nd best match         |
| `top2_dice`  | Dice score with top-2              |
| `top3_pmid`  | PMID of the 3rd best match         |
| `top3_dice`  | Dice score with top-3              |
| `worst_pmid` | PMID of the worst match            |
| `worst_dice` | Dice score with worst match        |

Summary statistics (mean, median, min, max) for top-1 and worst Dice scores are printed to the terminal at the end.

---

## Step 2 — Visualize Top-3 (`visualize_top3.py`)

Reads `dice_results.csv`, selects the 3 queries with the highest top-1 Dice score, and generates one figure per query.

For each PMID, the script:
- Loads and normalizes the KDE NIfTI map
- Projects it onto a brain surface (nilearn)
- Fetches the PubMed abstract via the NCBI API
- Extracts brain regions mentioned in the abstract
- Generates a 4-column figure color-coded by Dice score

### Configuration

```python
CSV_PATH   = "./results/dice_results.csv"
KDE_DIR    = "your/KDE"
OUTPUT_DIR = "./results"
THRESHOLD  = 0.8
```

### Run

```bash
python retrieval.py
```

### Output — `results/dice_viz_<query_pmid>.png`

Each figure contains 4 columns:

| Column | Content                                            |
|--------|----------------------------------------------------|
| Query  | KDE map of the query, no Dice score                |
| Top-1  | Best match — score color-coded (red→green)         |
| Top-2  | 2nd best match                                     |
| Top-3  | 3rd best match                                     |

Below each brain: the PMID and brain regions extracted from the abstract (in blue).

---

## Technical Details

### Binarization
NIfTI images are z-scored and binarized at `THRESHOLD`. Images with near-zero standard deviation return an array of zeros.

### Dice Score
```
Dice = 2 × |A ∩ B| / (|A| + |B|)
```
Returns `0.0` if both masks are empty.

### Brain Region Extraction
Regions are detected by substring matching against a list of ~60 neuroanatomical terms in the PubMed abstract (fetched via NCBI Entrez efetch API).

---

## Limitations

- Requires an internet connection to fetch PubMed abstracts.
- PMIDs must exactly match the NIfTI filenames (`pmid_<pmid>.nii.gz`).
- Surface projection (`plot_img_on_surf`) may be slow for large volumes.
- Queries with no available NIfTI are skipped silently.
