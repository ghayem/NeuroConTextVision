# BrainIAC KDE Embeddings Extraction:
Extract deep volumetric embeddings from KDE brain maps using BrainIAC   

1. download BrainIAC.ckpt from https://www.dropbox.com/scl/fo/i51xt63roognvt7vuslbl/AG99uZljziHss5zJz4HiFis?rlkey=9w55le6tslwxlfz6c0viylmjb&e=1&st=b9cnvwh8&dl=0 and place it in brainiac/checkpoints

2. run the following command:
```bash
python extract_brainiac_embeddings.py
```
> results are saved inside `brainiac/embeddings`



##  UNETR Segmentation Inference & Feature Extraction

Extract spatially-aligned segmentation masks using the pretrained UNETR model (`segmentation.ckpt`).

### Setup
1. Download `segmentation.ckpt` and place it in: `brainiac/checkpoints/segmentation.ckpt`
2. Ensure your input data is in: `kde_data/kde_samples/kde_processed/` (format: `pmid_XXXXXX.nii.gz`)

### Files
The main inference script is located in: `brainiac/unetr_inference/`

| File | Purpose |
|------|---------|
| `inference_segmentation.py` | Processes **all** `.nii.gz` files in the input folder automatically and saves results to `predictions/` |

###  Run Inference
```bash
python brainiac/unetr_inference/inference_segmentation.py

###   Visualize Predictions
```bash
python brainiac/unetr_inference/kde_visualization.py



### 🔹 Extract UNETR Encoder Embeddings
```bash
python brainiac/unetr_inference/extract_unetr_embeddings.py
