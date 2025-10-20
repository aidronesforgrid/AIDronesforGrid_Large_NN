# AIDronesforGrid_Large_NN
Creating a Large Neural Network to train an AI for image classification of power line components. 
 =========================
# 0) Colab Setup (Installs) (for tuning: adding albumentation pip and install commands)
# =========================
!pip -q install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip -q install scikit-learn pandas
!pip -q install albumentations==1.4.3 opencv-python-headless

# =========================
# 1) Mount Drive & Unzip
# =========================
from google.colab import drive
drive.mount('/content/drive')

import os, zipfile, sys
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---- UPDATE THIS if your zip is in a different Drive folder ----
ZIP_PATH = "/content/drive/MyDrive/unsupervised_anomaly_detection.zip"

# Unzip into /content/ (so we'll get /content/glass-insulator/test/good and .../missingcap)
if not Path(ZIP_PATH).exists():
    raise FileNotFoundError(f"Could not find: {ZIP_PATH}. Move the zip to that path or update ZIP_PATH.")

print("Unzipping dataset (first time only)...")
!unzip -q -o "/content/drive/MyDrive/unsupervised_anomaly_detection.zip" -d /content/

# =========================
# 2) Imports & Config
# =========================
import os, math, time, random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets, models
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---- Your dataset root: contains 'good' and 'missingcap' subfolders ----
IMAGEFOLDER_ROOT = "/content/glass-insulator/test"  # <- This matches your Windows paths once unzipped

# Training configuration
# MODEL_NAME = "resnet18"     # "resnet18", "resnet50", or "efficientnet_v2_s"
# FREEZE_BACKBONE = False     # True = feature extraction; False = full fine-tune
# IMG_SIZE = 224
# BATCH_SIZE = 32
# LR = 3e-4
# EPOCHS = 10
# VAL_SPLIT = 0.2
# RANDOM_SEED = 42
# USE_MIXED_PRECISION = True

SAVE_DIR = Path("/content/nn_runs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
BEST_CKPT = SAVE_DIR / "best_model.pt"

# =========================
# For tuning: Adding dictionary below to house all hyperparameters
# =========================

HP = dict(
    MODEL_NAME="resnet18",
    FREEZE_BACKBONE=False,
    IMG_SIZE=224,
    BATCH_SIZE=32,
    LR=3e-4,
    WEIGHT_DECAY=5e-4,
    EPOCHS=20,
    VAL_SPLIT=0.2,
    LABEL_SMOOTH=0.05,
    MIXUP_ALPHA=0.2,      # set >0 to enable mixup
    DROPOUT=0.0,          # we’ll add to head
    AUG_STRENGTH=0.5,     # 0..1 controls jitter/blur probs
    SCHEDULER="cosine",   # ["cosine","onecycle","none"]
    WARMUP_EPOCHS=1,
    USE_MIXED_PRECISION=True,
    RANDOM_SEED=42,
# for tuning: adding the following line
    USE_WEIGHTED_SAMPLER=True,   # True = balanced mini-batches

    )
# Define RANDOM_SEED before set_seed is called
RANDOM_SEED = HP["RANDOM_SEED"]

# =========================
# 3) Reproducibility
# =========================
def set_seed(seed=RANDOM_SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
set_seed()

# =========================
# 4) Transforms
# =========================
# def get_transforms(img_size):
#     train_tf = transforms.Compose([
#         transforms.Resize((img_size, img_size)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
#     ])
#     val_tf = transforms.Compose([
#         transforms.Resize((img_size, img_size)),
#         transforms.CenterCrop(img_size),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
#     ])
#     return train_tf, val_tf

# =========================
# HP4) Transforms
# =========================

def get_transforms_albu(img_size: int, aug_strength: float):
    """
    Albumentations transforms.
    - Returns callables that accept a PIL.Image and return a CHW torch.Tensor (float, normalized).
    - aug_strength in [0,1] scales probabilities/intensities.
    """
    jitter = 0.1 + 0.4 * aug_strength          # brightness/contrast/saturation factor
    p_aug  = 0.2 + 0.6 * aug_strength          # generic prob knob
    erase_scale = (0.02, 0.12)

    train_tf = A.Compose([
        A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=p_aug),
        A.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=0.02, p=p_aug),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3 if aug_strength > 0.3 else 0.0),
        # Optional “drone-ish” effects (enable if useful for your data)
        # A.MotionBlur(blur_limit=7, p=0.15),
        # A.RandomShadow(p=0.15),
        # A.RandomRain(p=0.1),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])

    val_tf = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.CenterCrop(height=img_size, width=img_size),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])

    # Adapter so ImageFolder-style Dataset can call transform(img) where img is a PIL.Image
    class _AlbuAdapter:
        def __init__(self, albu_compose):
            self.albu = albu_compose
        def __call__(self, pil_img):
            # Albumentations expects numpy (H, W, C) in RGB
            np_img = np.array(pil_img)  # converts PIL->np RGB
            out = self.albu(image=np_img)["image"]  # torch.Tensor (C,H,W), float normalized
            return out

    return _AlbuAdapter(train_tf), _AlbuAdapter(val_tf)

    # Validation should be deterministic
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return train_tf, val_tf

# =========================
# 5) Data Preparation (tuning denoted)
# =========================
def prepare_dataloaders_imagefolder(root, val_split=0.2):
    if not Path(root).exists():
        raise FileNotFoundError(f"Expected dataset root at {root} with class subfolders.")
    full_ds = datasets.ImageFolder(root, transform=None)
    if len(full_ds.samples) == 0:
        raise RuntimeError(f"No images found under {root}. Check subfolders like {root}/good and {root}/missingcap.")
    classes = full_ds.classes
    class_to_idx = full_ds.class_to_idx
    print("Found classes:", classes)

    # Stratified split
    paths = [s[0] for s in full_ds.samples]
    labels = [s[1] for s in full_ds.samples]

    train_idx, val_idx = train_test_split(
        np.arange(len(paths)),
        test_size=val_split,
        random_state=RANDOM_SEED,
        stratify=labels
    )

    train_tf, val_tf = get_transforms_albu(HP["IMG_SIZE"], HP["AUG_STRENGTH"])

    class SubsetFolder(Dataset):
        def __init__(self, indices, transform):
            self.indices = indices
            self.transform = transform
            self.samples = [full_ds.samples[i] for i in indices]
        def __len__(self): return len(self.samples)
        def __getitem__(self, i):
            path, y = self.samples[i]
            img = Image.open(path).convert("RGB")
            if self.transform: img = self.transform(img)
            return img, y

    train_ds = SubsetFolder(train_idx, train_tf)
    val_ds   = SubsetFolder(val_idx,   val_tf)

    train_counts = Counter([y for _, y in train_ds.samples])
    print("Train counts by class index:", dict(train_counts))
    print({classes[k]: v for k, v in train_counts.items()})

    # replacing (tuning) =====================
    # train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    # val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    # return train_loader, val_loader, classes, class_to_idx, train_counts
    # ====================

    if HP.get("USE_WEIGHTED_SAMPLER", False):
        class_freq = Counter([y for _, y in train_ds.samples])            # {class_idx: count}
        inv_freq = {c: 1.0 / max(n, 1) for c, n in class_freq.items()}    # avoid div-by-zero
        sample_weights = [inv_freq[y] for _, y in train_ds.samples]       # one weight per sample
        sampler = WeightedRandomSampler(sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=HP["BATCH_SIZE"], sampler=sampler, # Use HP["BATCH_SIZE"]
            num_workers=2, pin_memory=True, drop_last=False
    )
    else:
        # fallback: classic shuffle (no balancing)
        train_loader = DataLoader(
            train_ds, batch_size=HP["BATCH_SIZE"], shuffle=True, # Use HP["BATCH_SIZE"]
            num_workers=2, pin_memory=True, drop_last=False
    )

    val_loader = DataLoader(
        val_ds, batch_size=HP["BATCH_SIZE"], shuffle=False, # Use HP["BATCH_SIZE"]
        num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, classes, class_to_idx, train_counts


train_loader, val_loader, classes, class_to_idx, train_counts = prepare_dataloaders_imagefolder(IMAGEFOLDER_ROOT, HP["VAL_SPLIT"]) # Use HP["VAL_SPLIT"]

# =========================
# 6) Model Factory
# =========================
def build_model(num_classes):
    if HP["MODEL_NAME"] == "resnet18": # Use HP["MODEL_NAME"]
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
    elif HP["MODEL_NAME"] == "resnet50": # Use HP["MODEL_NAME"]
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
    elif HP["MODEL_NAME"] == "efficientnet_v2_s": # Use HP["MODEL_NAME"]
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        in_feats = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feats, num_classes)
    else:
        raise ValueError("Unsupported MODEL_NAME.")

    if HP["FREEZE_BACKBONE"]: # Use HP["FREEZE_BACKBONE"]
        for name, p in model.named_parameters():
            if not any(k in name for k in ["fc", "classifier"]):
                p.requires_grad = False
    return model

model = build_model(num_classes=len(classes)).to(device)

# =========================
# 7) Loss (with class weighting)
# =========================
# total = sum(train_counts.values())
# weights = [total / train_counts.get(i, 1) for i in range(len(classes))]
# class_weight = torch.tensor(weights, dtype=torch.float32, device=device)
# criterion = nn.CrossEntropyLoss(weight=class_weight)

# optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
# scaler = torch.cuda.amp.GradScaler(enabled=(USE_MIXED_PRECISION and device.type == "cuda"))

# =========================
# HP7) Loss (for tuning)
# =========================

# for tuning: replacing the following line ==========
# criterion = nn.CrossEntropyLoss()
# ==================
criterion = nn.CrossEntropyLoss(label_smoothing=HP["LABEL_SMOOTH"]) # Use HP["LABEL_SMOOTH"]
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=HP["LR"], # Use HP["LR"]
    weight_decay=HP["WEIGHT_DECAY"],
)
scaler = torch.cuda.amp.GradScaler(enabled=(HP["USE_MIXED_PRECISION"] and device.type == "cuda")) # Use HP["USE_MIXED_PRECISION"]

steps_per_epoch = max(1, len(train_loader))
total_steps = steps_per_epoch * HP["EPOCHS"] # Use HP["EPOCHS"]
warmup_steps = max(1, int(HP["WARMUP_EPOCHS"] * steps_per_epoch))

if HP["SCHEDULER"].lower() == "cosine":
    def lr_lambda(step):
        # linear warmup
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # cosine decay 1 -> 0 over remaining steps
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

elif HP["SCHEDULER"].lower() == "onecycle":
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=HP["LR"], # Use HP["LR"]
        steps_per_epoch=steps_per_epoch,
        epochs=HP["EPOCHS"], # Use HP["EPOCHS"]
        pct_start=max(1, warmup_steps)/float(max(1, total_steps)),  # warmup fraction
        anneal_strategy="cos",
        div_factor=25.0,        # initial_lr = max_lr/div_factor
        final_div_factor=1e4,   # min_lr = initial_lr/final_div_factor
    )

else:
    scheduler = None

# =========================
# 8) Train / Eval Loops (tuning line-by-line)
# =========================

# for tuning: adding the following block of code before next comment

def mixup_data(x, y, alpha: float):
    """Returns mixed inputs, paired targets, and lambda."""
    if alpha <= 0.0:
        return x, (y, y), 1.0  # no-op, keeps API consistent
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, (y_a, y_b), lam

def mixup_criterion(criterion, pred, targets, lam: float):
    y_a, y_b = targets
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)

# for tuning: replacing train_one_epoch with version that will use MixUp"
def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler=None,
    scheduler=None,
    mixup_alpha: float = 0.0,
):
    model.train()
    run_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        # ---- MixUp augmentation ----
        x_in, targets, lam = mixup_data(x, y, mixup_alpha)

        if scaler:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x_in)
                if mixup_alpha > 0.0:
                    loss = mixup_criterion(criterion, logits, targets, lam)
                else:
                    loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x_in)
            if mixup_alpha > 0.0:
                loss = mixup_criterion(criterion, logits, targets, lam)
            else:
                loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        # ---- Scheduler update (runs once per batch) ----
        if scheduler is not None:
            scheduler.step()

        # ---- Track training loss and accuracy ----
        run_loss += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    # ---- Return average loss and accuracy ----
    return run_loss / total, correct / total

@torch.no_grad()
def eval_one_epoch(model, loader, criterion):
    model.eval()
    run_loss, correct, total = 0.0, 0, 0
    ys, preds = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        run_loss += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        ys.append(y.cpu().numpy()); preds.append(pred.cpu().numpy())
    ys = np.concatenate(ys); preds = np.concatenate(preds)
    return run_loss/total, correct/total, ys, preds

# =========================
# 9) Run Training with Early Stop
# =========================
best_val_acc = 0.0
patience, bad_epochs = 3, 0

print("Classes:", classes)
for epoch in range(1, HP["EPOCHS"] + 1): # Use HP["EPOCHS"]
# tuning edit: adding "scheduler", and "mixup_alpha=HP["MIXUP_ALPHA"]"
    tr_loss, tr_acc = train_one_epoch(
    model, train_loader, criterion, optimizer,
    scaler, scheduler, mixup_alpha=HP["MIXUP_ALPHA"])
    va_loss, va_acc, y_true, y_pred = eval_one_epoch(model, val_loader, criterion)

# tuning edit: adding command to pring LR$ each epoch just below
    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch:02d}/{HP['EPOCHS']} | lr={current_lr:.2e} | train_loss={tr_loss:.4f} acc={tr_acc:.3f} " # Use HP["EPOCHS"]
      f"| val_loss={va_loss:.4f} acc={va_acc:.3f}")

    if va_acc > best_val_acc:
        best_val_acc = va_acc
        torch.save({
            "model_state": model.state_dict(),
            "classes": classes,
            "model_name": HP["MODEL_NAME"], # Use HP["MODEL_NAME"]
            "img_size": HP["IMG_SIZE"], # Use HP["IMG_SIZE"]
        }, BEST_CKPT)
        bad_epochs = 0
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            print("Early stopping.")
            break

print(f"\nBest validation accuracy: {best_val_acc:.3f}")
print(f"Saved checkpoint: {BEST_CKPT}")

# =========================
# 10) Final Validation Report
# =========================
print("\nClassification report on validation set:")
print(classification_report(y_true, y_pred, target_names=classes, digits=4))
print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred))

# =========================
# 11) Minimal Inference Helper
# =========================
@torch.no_grad()
def predict_image(path):
    model.eval()
    ckpt = torch.load(BEST_CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    tf = transforms.Compose([
        transforms.Resize((HP["IMG_SIZE"], HP["IMG_SIZE"])), # Use HP["IMG_SIZE"]
        transforms.CenterCrop(HP["IMG_SIZE"]), # Use HP["IMG_SIZE"]
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    img = Image.open(path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    logits = model(x)
    pred = logits.argmax(1).item()
    return classes[pred]

# Example usage after training (update path to a real image):
# print(predict_image("/content/glass-insulator/test/good/your_image.jpg"))
