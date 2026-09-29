# medical-ai

A multi-model medical AI application with three disease detectors:

- **Pneumonia** (chest X-ray) — ResNet18
- **Brain tumour** (MRI) — ResNet18
- **Skin cancer** (dermoscopy) — ResNet18

## Run the app

```bash
pip install -r requirements.txt 2>/dev/null || pip install streamlit torch torchvision
streamlit run app.py
```

## Run the evaluation

```bash
pip install -r requirements.txt 2>/dev/null || pip install torch torchvision scikit-learn
python evaluate.py
```

## Models

The pre-trained model weights (`*.pth`) are **large (~134 MB each)** and are
*not* stored in this repository. See [MODEL_LOCATION.md](MODEL_LOCATION.md) for
how to obtain them, or train them from scratch using the `training.ipynb`
notebooks in each detector folder.

## Data

The full test image datasets (`chest_xray/`, `mri/`, `my_data/`) are large
directories and are excluded from this repository via `.gitignore`. The
`training.ipynb` files in each detector folder contain the training data and
code.

## Contents

| File                     | Purpose                                            |
| ------------------------ | -------------------------------------------------- |
| `app.py`                 | Streamlit UI for the three detectors               |
| `evaluate.py`            | Evaluates the models on the local test sets        |
| `MODEL_LOCATION.md`      | Where to get the pre-trained model weights         |
| `training.ipynb` (×3)    | Training notebooks for each detector               |
| `model.py` (×3, in each folder) | Model definition + prediction function     |
| `MEDICAL.pdf`            | Medical documentation                              |
| `evaluation_report.txt`  | Output of the last `evaluate.py` run               |

## Creating the model weights (one-line summary)

Each `model.py` function is loaded by `app.py` from a `best_model.pth` file in
the same folder. To regenerate any of the three weights, open the matched
`training.ipynb` and run it (after installing the `torch`/`torchvision` +
`scikit-learn` + `opencv-python` stack and preparing the small test folder of
images in that folder).
"# Medical-AI" 
