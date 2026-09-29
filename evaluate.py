"""
Evaluate the three Medical AI models on their local test sets.

Computes: accuracy, error rate, precision / recall / F1 (macro & weighted),
per-class report, confusion matrix and average cross-entropy loss.
Saves a full report to evaluation_report.txt
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 128
RESULTS = []


def build_model(num_classes):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def evaluate_model(name, model_dir, ckpt_name, data_dir, num_classes, batch_size=32):
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {name}")
    print(f"{'=' * 60}")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float32),
    ])

    test_ds = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(num_classes)
    ckpt = torch.load(os.path.join(model_dir, ckpt_name), map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()

    all_preds, all_labels, all_losses = [], [], []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            all_losses.append(loss.item())
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    class_names = test_ds.classes

    acc = accuracy_score(y_true, y_pred)
    err = 1 - acc
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_w = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    avg_loss = float(np.mean(all_losses))
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, digits=4)

    print(f"Test images          : {len(y_true)}  ({num_classes} classes)")
    print(f"Accuracy             : {acc:.4f}  ({acc * 100:.2f} %)")
    print(f"Error rate           : {err:.4f}  ({err * 100:.2f} %)")
    print(f"Precision (macro)    : {prec_macro:.4f}")
    print(f"Recall    (macro)    : {rec_macro:.4f}")
    print(f"F1-score  (macro)    : {f1_macro:.4f}")
    print(f"Precision (weighted) : {prec_w:.4f}")
    print(f"Recall    (weighted) : {rec_w:.4f}")
    print(f"F1-score  (weighted) : {f1_w:.4f}")
    print(f"Avg test loss (CE)   : {avg_loss:.4f}")
    print(f"\nPer-class report:\n{report}")
    print(f"Confusion matrix (rows=true, cols=predicted)  classes={class_names}")
    for cname, row in zip(class_names, cm):
        print(f"  {cname:<30s} {row.tolist()}")

    RESULTS.append({
        "name": name, "n": len(y_true), "acc": acc, "err": err,
        "prec_macro": prec_macro, "rec_macro": rec_macro, "f1_macro": f1_macro,
        "prec_w": prec_w, "rec_w": rec_w, "f1_w": f1_w, "loss": avg_loss, "cm": cm,
    })


if __name__ == "__main__":
    evaluate_model("Pneumonia (chest X-ray)", "pneumonia", "best_model.pth",
                   "chest_xray/test", 2)
    evaluate_model("Brain Tumour (MRI)", "tumour", "best_mrimodel.pth",
                   "mri/Testing", 4)
    evaluate_model("Skin Cancer (dermoscopy)", "skin_cancer", "best_skinmodel.pth",
                   "my_data/test", 14)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Model':<26s}{'Test N':>7s}{'Acc %':>8s}{'Err %':>8s}{'P(w)':>8s}{'R(w)':>8s}{'F1(w)':>8s}{'Loss':>8s}")
    for r in RESULTS:
        print(f"{r['name']:<26s}{r['n']:>7d}{r['acc'] * 100:>8.2f}{r['err'] * 100:>8.2f}"
              f"{r['prec_w']:>8.4f}{r['rec_w']:>8.4f}{r['f1_w']:>8.4f}{r['loss']:>8.4f}")
