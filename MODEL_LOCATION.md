# Pre-trained Model Weights

The three pre-trained model weights are **large (~134 MB each)** and are **not**
committed to this repository (they exceed GitHub's 100 MB per-file limit).

## What's missing

| Folder          | Weight file     |
| --------------- | --------------- |
| `pneumonia/`    | `best_model.pth`    |
| `tumour/`       | `best_mrimodel.pth` |
| `skin_cancer/`  | `best_skinmodel.pth` |

## Options to get them

1. **Re-train from scratch.** Open the matching `training.ipynb` in each
   detector folder (e.g. `pneumonia/training.ipynb`) and run it. This
   regenerates `best_model.pth` in that folder. Requires:
   - `torch`, `torchvision`
   - `scikit-learn`
   - image data for that disease (documented inside the notebook)

2. **Download from a release / artifact store.** The weights are exported
   (zip/artifact) from the training environment and can be uploaded to a
   GitHub Release or model host. Place the downloaded `.pth` into the matching
   folder so `app.py` can load it.

## How `app.py` finds them

`app.py` imports `predict_*` from each detector's `model.py`; each
`model.py` loads its own `best_model.pth` from the same folder at import time,
so after placing the weights the app works out of the box.
