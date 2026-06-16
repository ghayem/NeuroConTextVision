# NeuroConTextVision
NeuroConText with Rich Image Representation using pre-trained foundation models (FM)

BrainIAC: https://github.com/AIM-KannLab/BrainIAC <br>
SwinBrain: https://github.com/MAI-Lab-West-China-Hospital/SwinBrain <br>
SegVol: (https://github.com/BAAI-DCAI/SegVol)<br>

## Setup

1. **Clone the repository**:

```bash
git clone https://github.com/ghayem/NeuroConTextVision.git
cd NeuroConTextVision
````

2. **Set up environment with UV**:

```bash
# Install UV if not already installed
curl -LsSf https://astral-sh.uv.install.sh | sh

# Sync the environment
uv sync
```

3. **Add your KDE NIfTI files** in:

```text
data/kde_samples/
```

Files must be named like: `pmid_<PMID>.nii.gz`.

---

## Embeddings Extraction

* **Docs**: See `brainiac/README.md`
* **Dataset**: [here](https://filesender.renater.fr/?s=download&token=b260710c-03ae-41af-9e02-87834fb48ffb)

> Both pipelines save embeddings progressively; partial results are preserved in case of interruption.
