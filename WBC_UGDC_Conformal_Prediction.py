# ==============================
# Deep Ensemble + Conformal Prediction (ICP) for WBC Classification
# ==============================

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

from efficientnet_pytorch import EfficientNet

# ------------------------------
# Reproducibility
# ------------------------------
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ------------------------------
# Device
# ------------------------------
device = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
print(f"Using device: {device}")

# ------------------------------
# Parameters
# ------------------------------
data_dir = "/path/dataset"  
batch_size = 128
num_classes = 9         
learning_rate = 1e-3
num_epochs = 10
train_frac = 0.8        # 80% train, 10% calib, 10% test
alpha_list = [0.05, 0.10, 0.20]  # conformal risks to try

# ------------------------------
# Transforms
# ------------------------------
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "eval": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# ------------------------------
# Dataset & Split: Train / Calib / Test
# ------------------------------
full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms["train"])
class_names = full_dataset.classes
assert len(class_names) == num_classes, f"num_classes={num_classes} but found {len(class_names)} folders."

N = len(full_dataset)
n_train = int(train_frac * N)
n_rest = N - n_train
n_calib = n_rest // 2
n_test  = n_rest - n_calib

train_dataset, rest_dataset = random_split(full_dataset, [n_train, n_rest])
calib_dataset, test_dataset = random_split(rest_dataset, [n_calib, n_test])

# Set eval transforms for calib and test subsets
train_dataset.dataset.transform = data_transforms["train"]
calib_dataset.dataset.transform = data_transforms["eval"]
test_dataset.dataset.transform  = data_transforms["eval"]

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4)
calib_loader = DataLoader(calib_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4)

print(f"Dataset sizes: train={len(train_dataset)}, calib={len(calib_dataset)}, test={len(test_dataset)}")

# ------------------------------
# Models (3-model Deep Ensemble)
# ------------------------------
def build_models(num_classes):
    m1 = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    m1.classifier[3] = nn.Linear(m1.classifier[3].in_features, num_classes)

    m2 = EfficientNet.from_pretrained("efficientnet-b0")
    m2._fc = nn.Linear(m2._fc.in_features, num_classes)

    m3 = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    m3.fc = nn.Linear(m3.fc.in_features, num_classes)

    return [m1.to(device), m2.to(device), m3.to(device)]

models_list = build_models(num_classes)

# ------------------------------
# Training
# ------------------------------
criterion = nn.CrossEntropyLoss()
optimizers = [optim.Adam(m.parameters(), lr=learning_rate) for m in models_list]
schedulers = [optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5) for opt in optimizers]

def train_one_model(model, optimizer, scheduler, num_epochs, name="model"):
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    start = time.time()

    for epoch in range(num_epochs):
        # Train
        model.train()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            ep_loss += loss.item() * x.size(0)
            ep_correct += (logits.argmax(1) == y).sum().item()
            ep_total += y.size(0)

        tr_loss = ep_loss / len(train_loader.dataset)
        tr_acc = ep_correct / ep_total
        train_loss.append(tr_loss)
        train_acc.append(tr_acc)

        
        model.eval()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in calib_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                ep_loss += loss.item() * x.size(0)
                ep_correct += (logits.argmax(1) == y).sum().item()
                ep_total += y.size(0)

        vl_loss = ep_loss / len(calib_loader.dataset) if len(calib_loader.dataset) > 0 else float("nan")
        vl_acc = ep_correct / ep_total if ep_total > 0 else float("nan")
        val_loss.append(vl_loss)
        val_acc.append(vl_acc)

        scheduler.step()

        print(f"{name} | Epoch {epoch+1}/{num_epochs} "
              f"| Train Loss {tr_loss:.4f} Acc {tr_acc:.4f} "
              f"| Calib Loss {vl_loss:.4f} Acc {vl_acc:.4f}")

    print(f"⏱ {name} training time: {time.time()-start:.2f}s")
    return train_loss, train_acc, val_loss, val_acc

print("\n=== Training models ===")
all_metrics = []
for i, (m, opt, sch) in enumerate(zip(models_list, optimizers, schedulers), start=1):
    tl, ta, vl, va = train_one_model(m, opt, sch, num_epochs, name=f"Model_{i}")
    all_metrics.append({"train_loss": tl, "train_acc": ta, "val_loss": vl, "val_acc": va})

# ------------------------------
# Plot training curves
# ------------------------------
def plot_training_curves(metrics_list, num_epochs):
    for i, metrics in enumerate(metrics_list, start=1):
        epochs = range(1, num_epochs + 1)
        plt.figure(figsize=(12, 4))
        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, metrics["train_loss"], label="Train")
        plt.plot(epochs, metrics["val_loss"], label="Calib")
        plt.title(f"Model {i} Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
        # Acc
        plt.subplot(1, 2, 2)
        plt.plot(epochs, metrics["train_acc"], label="Train")
        plt.plot(epochs, metrics["val_acc"], label="Calib")
        plt.title(f"Model {i} Accuracy")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
        plt.tight_layout()
        plt.show()

plot_training_curves(all_metrics, num_epochs)

# ------------------------------
# Ensemble helpers
# ------------------------------
@torch.no_grad()
def ensemble_predict_proba(models, inputs):
    probs_accum = None
    for m in models:
        m.eval()
        logits = m(inputs)
        probs = F.softmax(logits, dim=1)
        probs_accum = probs if probs_accum is None else probs_accum + probs
    return probs_accum / len(models)

def plot_confusion(y_true, y_pred, class_names, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_calibration(y_true, prob_max, title="Calibration Curve"):
    y_true = np.array(y_true)
    # correctness wrt predicted class
    y_pred = prob_max.argmax(axis=1) if prob_max.ndim == 2 else None
    if y_pred is not None:
        correct = (y_true == y_pred).astype(int)
        conf = prob_max.max(axis=1)
    else:
        # If we only get a (N,) conf vector
        correct = y_true  # (not used this branch in this script)
        conf = prob_max
    prob_true, prob_pred = calibration_curve(correct, conf, n_bins=10, strategy="uniform")
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], "--", label="Perfect")
    plt.xlabel("Mean Predicted Confidence")
    plt.ylabel("Empirical Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------------------
# Evaluate Base Models on TEST
# ------------------------------
print("\n=== Base model evaluation on TEST ===")
for i, m in enumerate(models_list, start=1):
    y_true, y_pred, prob_all = [], [], []
    m.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = m(x)
            probs = F.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            prob_all.append(probs.cpu().numpy())
    prob_all = np.concatenate(prob_all, axis=0)

    print(f"\nClassification Report — Model {i}")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    plot_confusion(y_true, y_pred, class_names, title=f"Model {i} Confusion Matrix")
    plot_calibration(y_true, prob_all, title=f"Model {i} Calibration Curve")

# ------------------------------
# Evaluate Ensemble on TEST
# ------------------------------
print("\n=== Ensemble evaluation on TEST ===")
y_true, y_pred, prob_all = [], [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        probs = ensemble_predict_proba(models_list, x)
        pred = probs.argmax(dim=1)

        y_true.extend(y.numpy())
        y_pred.extend(pred.cpu().numpy())
        prob_all.append(probs.cpu().numpy())
prob_all = np.concatenate(prob_all, axis=0)

print("\nClassification Report — Deep Ensemble")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
plot_confusion(y_true, y_pred, class_names, title="Deep Ensemble Confusion Matrix")
plot_calibration(y_true, prob_all, title="Deep Ensemble Calibration Curve")

# =========================================================
#              INDUCTIVE CONFORMAL PREDICTION
# =========================================================
def nonconformity_scores_on_calib(models, loader):
    """ s = 1 - p_true(x) on calibration set """
    scores = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            probs = ensemble_predict_proba(models, x)  # [B, C]
            p_true = probs.gather(1, y.unsqueeze(1)).squeeze(1)  # [B]
            s = 1.0 - p_true
            scores.append(s.cpu().numpy())
    scores = np.concatenate(scores, axis=0) if len(scores) else np.array([])
    scores = np.clip(scores, 0.0, 1.0)
    return scores

def qhat_from_scores(scores, alpha):
    """ q̂ = quantile_{ceil((n+1)*(1-alpha))/n} of scores """
    n = len(scores)
    if n == 0:
        return 1.0  # degenerate case; produce full sets
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return np.partition(scores, k-1)[k-1]

@torch.no_grad()
def conformal_predict_sets(models, loader, qhat):
    """ Γ(x) = {k : p_k(x) >= 1 - qhat} """
    thresh = 1.0 - qhat
    y_true_all, set_sizes, top1_all, in_set_masks = [], [], [], []
    for x, y in loader:
        x = x.to(device)
        probs = ensemble_predict_proba(models, x)  # [B, C]
        in_set = (probs >= thresh)                  # [B, C] bool
        set_size = in_set.sum(dim=1).cpu().numpy()
        top1 = probs.argmax(dim=1).cpu().numpy()

        in_set_masks.append(in_set.cpu().numpy())
        set_sizes.extend(list(set_size))
        top1_all.extend(list(top1))
        y_true_all.extend(list(y.numpy()))
    in_set_masks = np.concatenate(in_set_masks, axis=0) if len(in_set_masks) else np.zeros((0, num_classes), bool)
    return np.array(y_true_all), in_set_masks, np.array(set_sizes), np.array(top1_all)

def cp_metrics(y_true, in_set_mask, set_sizes):
    """ Coverage and average set size """
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0
    covered = in_set_mask[np.arange(n), y_true].astype(np.float32)
    coverage = covered.mean()
    avg_size = set_sizes.mean() if len(set_sizes) else 0.0
    return coverage, avg_size

print("\n=== Building conformal predictor (using CALIBRATION split) ===")
cal_scores = nonconformity_scores_on_calib(models_list, calib_loader)
print(f"Computed {len(cal_scores)} calibration scores.")

for alpha in alpha_list:
    qhat = qhat_from_scores(cal_scores, alpha)
    y_true, in_set_mask, set_sizes, top1 = conformal_predict_sets(models_list, test_loader, qhat)
    coverage, avg_size = cp_metrics(y_true, in_set_mask, set_sizes)
    print(f"\nalpha={alpha:.2f} -> q̂={qhat:.4f} (threshold p >= {1-qhat:.4f})")
    print(f"Coverage on TEST: {coverage:.3f} (target ≥ {1-alpha:.2f})")
    print(f"Average set size: {avg_size:.2f}")

    # Histogram of set sizes
    plt.figure(figsize=(6, 4))
    bins = np.arange(1, num_classes+2)
    plt.hist(set_sizes, bins=bins, align="left", rwidth=0.9)
    plt.xticks(bins[:-1])
    plt.xlabel("|Γ(x)|"); plt.ylabel("Count")
    plt.title(f"Conformal Prediction Set Sizes (alpha={alpha:.2f})")
    plt.tight_layout()
    plt.show()

# Risk–Coverage curve
coverages, risks, grid = [], [], np.linspace(0.0, 0.4, 21)
for a in grid:
    q = qhat_from_scores(cal_scores, a)
    y_true, mask, sizes, _ = conformal_predict_sets(models_list, test_loader, q)
    cov, _ = cp_metrics(y_true, mask, sizes)
    coverages.append(cov)
    risks.append(1 - cov)
plt.figure(figsize=(6, 4))
plt.plot(coverages, risks, marker="o")
plt.gca().invert_xaxis()  # higher coverage to the left
plt.xlabel("Coverage"); plt.ylabel("Risk (= 1 - coverage)")
plt.title("Risk–Coverage (Deep Ensemble + ICP)")
plt.grid(True)
plt.tight_layout()
plt.show()
