#!/usr/bin/env python3
"""
Distill a MobileNetV3-Small student from your TEACHER checkpoint, then export:
- PyTorch: best/last .pth
- ONNX: student.onnx (+ simplified)
- (optional) TFLite INT8 (for Coral USB TPU via edgetpu_compiler)

Quick start (Windows paths already default to your setup):
  python train_distill_export.py ^
    --data_root "C:\\Users\\wyatt\\OneDrive\\Documents\\Senior Design\\SD Dataset" ^
    --teacher_ckpt "C:\\Users\\wyatt\\OneDrive\\Documents\\Senior Design\\SD Dataset\\best_model.pt" ^
    --epochs 40 --batch_size 64 --to_tflite 1 --calib_images 300

Requirements:
  pip install torch torchvision onnx onnxsim onnx2tf tensorflow==2.* pillow
"""

import os, sys, math, time, argparse, random, json, shutil, subprocess
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ---------- DEFAULTS FOR YOUR ENV (Wyatt) ----------
DEFAULT_DATA_ROOT = r"C:\Users\wyatt\OneDrive\Documents\Senior Design\SD Dataset"
DEFAULT_TRAIN_SUBDIR = "train"
DEFAULT_VAL_SUBDIR   = "val"
DEFAULT_TEACHER_CKPT = str(Path(DEFAULT_DATA_ROOT) / "best_model.pt")
DEFAULT_OUTDIR       = str(Path(DEFAULT_DATA_ROOT) / "artifacts_student_mnv3s")


# ---------------------------
# Args
# ---------------------------
def get_args():
    p = argparse.ArgumentParser()
    # IO & data
    p.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT,
                   help="Root folder containing train/ and val/ subfolders")
    p.add_argument("--train_subdir", type=str, default=DEFAULT_TRAIN_SUBDIR)
    p.add_argument("--val_subdir", type=str, default=DEFAULT_VAL_SUBDIR)
    p.add_argument("--teacher_ckpt", type=str, default=DEFAULT_TEACHER_CKPT,
                   help="Path to teacher .pt/.pth saved by your teacher script")
    p.add_argument("--num_classes", type=int, default=0,
                   help="0 = infer from teacher ckpt or dataset")
    # Train
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--alpha", type=float, default=0.4, help="Weight for hard CE (0..1)")
    p.add_argument("--T", type=float, default=4.0, help="Temperature for KD")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    p.add_argument("--skip_train", action="store_true", help="Only export using latest student.pth")
    # Export
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--to_tflite", type=int, default=1, help="If 1, try onnx->tf->tflite INT8 here")
    p.add_argument("--calib_images", type=int, default=300, help="Representative dataset images for PTQ")
    return p.parse_args()


# ---------------------------
# Utils
# ---------------------------
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


class DistillationLoss(nn.Module):
    def __init__(self, T=4.0, alpha=0.4):
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, s_logits, t_logits, y):
        hard = self.ce(s_logits, y)
        soft = self.kl(
            F.log_softmax(s_logits / self.T, dim=1),
            F.softmax(t_logits / self.T, dim=1)
        ) * (self.T * self.T)
        return self.alpha * hard + (1 - self.alpha) * soft, hard.detach(), soft.detach()


# ---------------------------
# Teacher loading (matches your checkpoint format)
# ---------------------------
def _build_backbone(name: str, num_classes: int):
    name = (name or "").lower()
    if name == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name in {"efficientnet_v2_s", "efficientnet-v2-s"}:
        m = models.efficientnet_v2_s(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    # Fallback (safe default)
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def load_teacher(ckpt_path: str, num_classes_hint: int = 0):
    """
    Expects the teacher checkpoint saved like your code:
      torch.save({
         "model_state": model.state_dict(),
         "classes": classes,
         "model_name": HP["MODEL_NAME"],
         "img_size": HP["IMG_SIZE"],
      }, BEST_CKPT)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)

    classes = ckpt.get("classes", None)
    model_name = ckpt.get("model_name", "resnet50")
    if classes is not None and len(classes) > 0:
        num_classes = len(classes)
    else:
        num_classes = num_classes_hint if num_classes_hint > 0 else 1000

    teacher = _build_backbone(model_name, num_classes)
    state = ckpt.get("model_state", ckpt)  # allow raw state_dict fallback
    teacher.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
    teacher.eval().to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)

    meta = {
        "classes": classes,
        "model_name": model_name,
        "img_size": ckpt.get("img_size", 224),
        "num_classes": num_classes
    }
    return teacher, meta


# ---------------------------
# Student
# ---------------------------
def build_student(num_classes):
    m = models.mobilenet_v3_small(weights=None)
    in_feats = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_feats, num_classes)
    return m


# ---------------------------
# Data
# ---------------------------
def make_loaders(data_root, train_subdir, val_subdir, image_size, batch_size, num_workers):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(image_size*1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dir = Path(data_root)/train_subdir
    val_dir   = Path(data_root)/val_subdir
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Missing train/val dirs under {data_root}")

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds   = datasets.ImageFolder(val_dir, transform=val_tf)

    if len(train_ds.samples) == 0 or len(val_ds.samples) == 0:
        raise RuntimeError("Empty train/val set; check folder structure and file types.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_ds, val_ds


# ---------------------------
# Eval
# ---------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred==y).sum().item()
        total += y.numel()
    return correct/total if total>0 else 0.0


# ---------------------------
# Train KD
# ---------------------------
def train_kd(teacher, student, train_loader, val_loader, epochs, lr, weight_decay, T, alpha, outdir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher.eval().to(device)
    student.to(device)

    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    kd_loss = DistillationLoss(T=T, alpha=alpha)

    best_acc = 0.0
    (Path(outdir)).mkdir(parents=True, exist_ok=True)

    for ep in range(1, epochs+1):
        student.train()
        running = []
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits = student(x)
            loss, _, _ = kd_loss(s_logits, t_logits, y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            opt.step()
            running.append(loss.item())
        sched.step()

        val_acc = evaluate(student, val_loader, device)
        print(f"[{ep:03d}/{epochs}] train_kd_loss={np.mean(running):.4f}  val_acc={val_acc:.4f}")

        # checkpoints
        ck = {"epoch": ep, "val_acc": val_acc, "state_dict": student.state_dict()}
        torch.save(ck, Path(outdir)/"student_last.pth")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(ck, Path(outdir)/"student_best.pth")

    print(f"Best val acc: {best_acc:.4f}")
    return best_acc


# ---------------------------
# Export: ONNX
# ---------------------------
def export_onnx(student_pth, num_classes, image_size, outdir, opset=17):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_student(num_classes)
    ckpt = torch.load(student_pth, map_location=device)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)

    dummy = torch.randn(1,3,image_size,image_size, device=device)
    onnx_path = str(Path(outdir)/"student.onnx")
    torch.onnx.export(model, dummy, onnx_path,
                      input_names=["input"], output_names=["logits"],
                      opset_version=opset,
                      dynamic_axes={"input":{0:"batch"}, "logits":{0:"batch"}})
    print(f"Exported ONNX -> {onnx_path}")

    # Try simplify
    try:
        import onnx, onnxsim
        model_onnx = onnx.load(onnx_path)
        model_simplified, check = onnxsim.simplify(model_onnx)
        if check:
            onnx_path = str(Path(outdir)/"student_simplified.onnx")
            onnx.save(model_simplified, onnx_path)
            print(f"Simplified ONNX -> {onnx_path}")
    except Exception as e:
        print(f"[warn] onnxsim simplify skipped: {e}")

    return onnx_path


# ---------------------------
# Export: ONNX -> TF -> TFLite INT8 (PTQ)
# ---------------------------
def convert_onnx_to_tflite_int8(onnx_path, outdir, image_size, calib_paths):
    """
    Convert ONNX -> TF SavedModel (onnx2tf CLI) -> TFLite INT8 (with representative dataset).
    Produces: student_saved_model/, student_int8.tflite
    """
    saved_model_dir = str(Path(outdir)/"student_saved_model")

    # 1) ONNX -> TF SavedModel
    try:
        cmd = ["onnx2tf", "-i", onnx_path, "-o", saved_model_dir, "--output_signaturedefs"]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    except Exception as e:
        print("onnx2tf failed. Ensure 'pip install onnx2tf' and TensorFlow are installed.")
        raise e

    # 2) TFLite INT8 conversion
    import tensorflow as tf

    def rep_ds():
        # Feed representative samples roughly matching training preprocessing.
        mean = np.array([0.485,0.456,0.406], dtype=np.float32)
        std  = np.array([0.229,0.224,0.225], dtype=np.float32)
        for p in calib_paths:
            try:
                img = Image.open(p).convert("RGB")
                img = img.resize((image_size, image_size), Image.BILINEAR)
                arr = np.asarray(img).astype(np.float32) / 255.0
                arr = (arr - mean) / std
                # NHWC expected by TF graph from onnx2tf
                arr = arr[None, ...]  # [1, H, W, C]
                yield [arr]
            except Exception:
                continue

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_ds
    # Force INT8 kernels; set I/O to uint8 for Coral-friendly edges
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    tfl_path = str(Path(outdir)/"student_int8.tflite")
    open(tfl_path, "wb").write(tflite_model)
    print(f"Exported TFLite INT8 -> {tfl_path}")
    return tfl_path, saved_model_dir


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(Path(args.outdir)/"args.json","w"), indent=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Data
    train_loader, val_loader, train_ds, val_ds = make_loaders(
        args.data_root, args.train_subdir, args.val_subdir,
        args.image_size, args.batch_size, args.num_workers
    )
    dataset_classes = train_ds.classes
    print(f"Dataset classes ({len(dataset_classes)}): {dataset_classes}")

    # 2) Teacher / Student
    teacher, tmeta = load_teacher(args.teacher_ckpt, num_classes_hint=len(dataset_classes))
    if args.num_classes == 0:
        args.num_classes = tmeta["num_classes"] if tmeta["classes"] else len(dataset_classes)

    # Warn on class mismatch (folder names vs teacher ckpt)
    if tmeta["classes"] is not None and list(tmeta["classes"]) != list(dataset_classes):
        print("[WARN] Teacher checkpoint classes differ from dataset folder names.")
        print("       Teacher:", list(tmeta["classes"]))
        print("       Dataset:", list(dataset_classes))
        print("       Proceeding: KD uses teacher logits aligned to the current dataset indices.")

    student = build_student(args.num_classes)

    # 3) Train (KD)
    if not args.skip_train:
        best = train_kd(teacher, student, train_loader, val_loader,
                        args.epochs, args.lr, args.weight_decay, args.T, args.alpha, args.outdir)
        print("KD training done.")

    # 4) Export ONNX (from best or last)
    cand = Path(args.outdir)/"student_best.pth"
    if not cand.exists():
        cand = Path(args.outdir)/"student_last.pth"
    onnx_path = export_onnx(str(cand), args.num_classes, args.image_size, args.outdir, opset=args.opset)

    # 5) Optional: Build representative dataset list from your TRAIN/ for TFLite PTQ
    if args.to_tflite:
        # Collect calibration samples
        train_root = Path(args.data_root) / args.train_subdir
        candidates = []
        for root, _, files in os.walk(train_root):
            for f in files:
                if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp")):
                    candidates.append(str(Path(root)/f))
        random.shuffle(candidates)
        candidates = candidates[:args.calib_images]
        (Path(args.outdir)/"cache_sample_paths.json").write_text(json.dumps(candidates, indent=2))

        # Convert ONNX -> TF -> TFLite INT8
        try:
            tfl_path, sm_dir = convert_onnx_to_tflite_int8(onnx_path, args.outdir, args.image_size, candidates)
            print("\nNext (on Linux x86_64):")
            print(f"  edgetpu_compiler -s {tfl_path}")
            print("This will emit: student_int8_edgetpu.tflite  (deploy this on the Pi with pycoral).")
        except Exception as e:
            print("\n[TFLite INT8 conversion failed]")
            print("You can still do it manually:")
            print("  1) onnx2tf -i student.onnx -o student_saved_model --output_signaturedefs")
            print("  2) Use a small TF script to convert that SavedModel to INT8 with a representative dataset.")
            print(e)