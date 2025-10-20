
# =========================
# 2) Imports & Config
# =========================

import albumentations as A
from albumentations.pytorch import ToTensorV2

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
from sklearn.metrics import classification_report, confusion_matrix, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---- Your dataset roots: contain 14 class subfolders each ----
TRAIN_ROOT = r"C:\Users\wyatt\OneDrive\Documents\Senior Design\SD Dataset\train"  # Updated path
VAL_ROOT = r"C:\Users\wyatt\OneDrive\Documents\Senior Design\SD Dataset\val"    # Updated path

SAVE_DIR = Path(r"C:\Users\wyatt\OneDrive\Documents\Senior Design\SD Dataset")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
BEST_CKPT = SAVE_DIR / "best_model.pt"

# =========================
# For tuning: Adding dictionary below to house all hyperparameters
# =========================

HP = dict(
    MODEL_NAME="resnet18",
    FREEZE_BACKBONE=False,
    IMG_SIZE=224,
    BATCH_SIZE=128, # Increased batch size based on 32GB VRAM
    LR=3e-4,
    WEIGHT_DECAY=5e-4,
    EPOCHS=20,
    VAL_SPLIT=0.2, # This will no longer be used for splitting, but kept for other potential uses
    LABEL_SMOOTH=0.05,
    MIXUP_ALPHA=0.2,      # set >0 to enable mixup
    DROPOUT=0.0,          # we’ll add to head
    AUG_STRENGTH=0.5,     # 0..1 controls jitter/blur probs
    SCHEDULER="cosine",   # ["cosine","onecycle","none"]
    WARMUP_EPOCHS=1,
    USE_MIXED_PRECISION=True, # Ensure mixed precision is enabled
    RANDOM_SEED=42,
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
class _AlbuAdapter:
        def __init__(self, albu_compose):
            self.albu = albu_compose
        def __call__(self, pil_img):
            np_img = np.array(pil_img)
            out = self.albu(image=np_img)["image"]
            return out

def get_transforms_albu(img_size: int, aug_strength: float):
    jitter = 0.1 + 0.4 * aug_strength
    p_aug  = 0.2 + 0.6 * aug_strength
    erase_scale = (0.02, 0.12)

    train_tf = A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),  # <-- FIXED
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

    return _AlbuAdapter(train_tf), _AlbuAdapter(val_tf)


# =========================
# 5) Data Preparation
# =========================
def prepare_dataloaders_imagefolder(train_root, val_root):
    if not Path(train_root).exists():
        raise FileNotFoundError(f"Expected dataset root at {train_root} with class subfolders.")
    if not Path(val_root).exists():
        raise FileNotFoundError(f"Expected dataset root at {val_root} with class subfolders.")

    train_tf, val_tf = get_transforms_albu(HP["IMG_SIZE"], HP["AUG_STRENGTH"])

    train_ds = datasets.ImageFolder(train_root, transform=train_tf)
    val_ds = datasets.ImageFolder(val_root, transform=val_tf)

    assert train_ds.classes == val_ds.classes, \
        f"Class mismatch: train={train_ds.classes} vs val={val_ds.classes}"

    if len(train_ds.samples) == 0:
        raise RuntimeError(f"No images found under {train_root}. Check subfolders.")
    if len(val_ds.samples) == 0:
        raise RuntimeError(f"No images found under {val_root}. Check subfolders.")

    classes = train_ds.classes
    class_to_idx = train_ds.class_to_idx
    print("Found classes:", classes)

    train_counts = Counter([s[1] for s in train_ds.samples])
    print("Train counts by class index:", dict(train_counts))
    print({classes[k]: v for k, v in train_counts.items()})


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
            num_workers=2, pin_memory=(device.type == "cuda"), drop_last=False
    )

    val_loader = DataLoader(
        val_ds, batch_size=HP["BATCH_SIZE"], shuffle=False, # Use HP["BATCH_SIZE"]
        num_workers=2, pin_memory=(device.type == "cuda")
    )

    return train_loader, val_loader, classes, class_to_idx, train_counts

def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, mixup_alpha=0.0):
    model.train()
    running_loss, running_correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if mixup_alpha > 0:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            batch_size = x.size()[0]
            index = torch.randperm(batch_size).to(device)
            mixed_x = lam * x + (1 - lam) * x[index, :]
            y_a, y_b = y, y[index]
            with torch.cuda.amp.autocast(enabled=(HP["USE_MIXED_PRECISION"] and device.type == "cuda")):
                outputs = model(mixed_x)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
        else:
            with torch.cuda.amp.autocast(enabled=(HP["USE_MIXED_PRECISION"] and device.type == "cuda")):
                outputs = model(x)
                loss = criterion(outputs, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        preds = outputs.argmax(1)
        running_loss += loss.item() * x.size(0)
        running_correct += (preds == y).sum().item()
        total += x.size(0)
    avg_loss = running_loss / total
    avg_acc = running_correct / total
    return avg_loss, avg_acc

def eval_one_epoch(model, loader, criterion):
    model.eval()
    running_loss, running_correct, total = 0.0, 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            preds = outputs.argmax(1)
            running_loss += loss.item() * x.size(0)
            running_correct += (preds == y).sum().item()
            total += x.size(0)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    avg_loss = running_loss / total
    avg_acc = running_correct / total
    return avg_loss, avg_acc, y_true, y_pred

def build_model(num_classes):
    if HP["MODEL_NAME"] == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif HP["MODEL_NAME"] == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif HP["MODEL_NAME"] == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {HP['MODEL_NAME']}")
    if HP["FREEZE_BACKBONE"]:
        for name, param in model.named_parameters():
            if "fc" not in name and "classifier" not in name:
                param.requires_grad = False
    return model

def main():
    train_loader, val_loader, classes, class_to_idx, train_counts = prepare_dataloaders_imagefolder(TRAIN_ROOT, VAL_ROOT)
    model = build_model(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=HP["LABEL_SMOOTH"])
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=HP["LR"],
        weight_decay=HP["WEIGHT_DECAY"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(HP["USE_MIXED_PRECISION"] and device.type == "cuda"))
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * HP["EPOCHS"]
    warmup_steps = max(1, int(HP["WARMUP_EPOCHS"] * steps_per_epoch))

    if HP["SCHEDULER"].lower() == "cosine":
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif HP["SCHEDULER"].lower() == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=HP["LR"],
            steps_per_epoch=steps_per_epoch,
            epochs=HP["EPOCHS"],
            pct_start=max(1, warmup_steps)/float(max(1, total_steps)),
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e4,
        )
    else:
        scheduler = None

    best_val_f1 = -1.0
    patience, bad_epochs = 3, 0

    print("Classes:", classes)
    for epoch in range(1, HP["EPOCHS"] + 1):
        print(f"Epoch {epoch}/{HP['EPOCHS']}")
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scaler, scheduler, mixup_alpha=HP["MIXUP_ALPHA"])
        va_loss, va_acc, y_true, y_pred = eval_one_epoch(model, val_loader, criterion)
        val_macro_f1 = f1_score(y_true, y_pred, average="macro")
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:02d}/{HP['EPOCHS']} | lr={current_lr:.2e} | train_loss={tr_loss:.4f} acc={tr_acc:.3f} "
              f"| val_loss={va_loss:.4f} acc={va_acc:.3f} | val_macro_f1={val_macro_f1:.3f}")

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            torch.save({
                "model_state": model.state_dict(),
                "classes": classes,
                "model_name": HP["MODEL_NAME"],
                "img_size": HP["IMG_SIZE"],
            }, BEST_CKPT)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping.")
                break

    print(f"\nBest validation macro-F1: {best_val_f1:.3f}")
    print(f"Saved checkpoint: {BEST_CKPT}")

    print("\nClassification report on validation set:")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


    # =========================
    # Test Set Evaluation
    # =========================
    print("\n" + "="*60 + "\n")

    print("Evaluating on TEST set with the best saved model...")

    TEST_ROOT = r"C:\Users\wyatt\OneDrive\Documents\Senior Design\SD Dataset\test"
    # Load the best model
    checkpoint = torch.load(BEST_CKPT, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Prepare test loader
    _, test_tf = get_transforms_albu(HP["IMG_SIZE"], HP["AUG_STRENGTH"])
    test_ds = datasets.ImageFolder(TEST_ROOT, transform=test_tf)
    test_loader = DataLoader(
        test_ds, batch_size=HP["BATCH_SIZE"], shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda")
    )

    # Evaluate on test set
    test_loss, test_acc, y_true_test, y_pred_test = eval_one_epoch(model, test_loader, criterion)
    print(f"\nTest loss: {test_loss:.4f} | Test acc: {test_acc:.3f}")
    print("Classification report on test set:")
    print(classification_report(
    y_true_test,
    y_pred_test,
    labels=np.arange(len(classes)),
    target_names=classes,
    digits=4,
    zero_division=0
))
    print("Confusion matrix:")
    print(confusion_matrix(y_true_test, y_pred_test))
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

if __name__ == "__main__":
    main()

# Example usage after training (update path to a real image):
# print(predict_image("/content/glass-insulator/test/good/your_image.jpg"))