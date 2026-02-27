# BrainIAC KDE Embeddings Extraction:
Extract deep volumetric embeddings from KDE brain maps using BrainIAC   

1. download BrainIAC.ckpt from https://www.dropbox.com/scl/fo/i51xt63roognvt7vuslbl/AG99uZljziHss5zJz4HiFis?rlkey=9w55le6tslwxlfz6c0viylmjb&e=1&st=b9cnvwh8&dl=0 and place it in brainiac/checkpoints

2. run the following command:
```bash
python apply_segvol.py
```
> results are saved inside `brainiac/embeddings`