"""
train_vit.py
-------------
Entrenamiento de un Vision Transformer (ViT-B/16, torchvision, preentrenado
en ImageNet) para clasificación binaria real/fake sobre el dataset
StyleGAN3 Real vs Fake Faces.

Diseñado para correr como trabajo SBATCH no interactivo en Kabré
(partición nukwa-l40s, límite de 4h por trabajo). Guarda checkpoints
cada época para poder continuar el entrenamiento si el job se corta por
el límite de tiempo — basta con volver a lanzar el mismo script y
detecta automáticamente el último checkpoint.

Uso:
    python train_vit.py [--epochs N] [--batch-size N] [--lr F] [--resume]

Salidas:
    checkpoints/vit_best.pt        — mejor modelo según AUC en validación
    checkpoints/vit_last.pt        — checkpoint más reciente (para resume)
    results/training_log.csv       — métricas por época
    results/gpu_stats.csv          — utilización de GPU muestreada durante el entrenamiento
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16, ViT_B_16_Weights
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

# ── Rutas del proyecto ─────────────────────────────────────────────────────
PROJECT_ROOT = Path("/data/ulead-04/proyecto_paralela")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SRC_DIR = PROJECT_ROOT / "src"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC_DIR))
from preprocessing_worker import ChunkedNpyIterableDataset  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# Argumentos
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--resume", action="store_true",
                    help="Continuar desde checkpoints/vit_last.pt si existe")
    p.add_argument("--max-minutes", type=float, default=220,
                    help="Detener y guardar checkpoint antes de este límite "
                         "(margen de seguridad bajo el límite de 240 min del job)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Modelo
# ══════════════════════════════════════════════════════════════════════════════

def build_model(device):
    """ViT-B/16 preentrenado en ImageNet, cabeza reemplazada para clasificación binaria."""
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)

    # Reemplazar la cabeza de clasificación (1000 clases ImageNet -> 2 clases)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, 2)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Modelo: ViT-B/16 | Parámetros entrenables: {n_params:,}")

    return model.to(device)


# ══════════════════════════════════════════════════════════════════════════════
# GPU monitor (para el análisis de rendimiento del curso)
# ══════════════════════════════════════════════════════════════════════════════

class GPUMonitor:
    """Muestrea utilización/memoria de GPU en segundo plano vía nvidia-smi."""

    def __init__(self, log_path):
        self.log_path = log_path
        self.rows = []

    def sample(self, epoch, step):
        if not torch.cuda.is_available():
            return
        mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        self.rows.append({
            "epoch": epoch, "step": step,
            "mem_allocated_gb": round(mem_alloc, 3),
            "mem_reserved_gb": round(mem_reserved, 3),
            "timestamp": time.time(),
        })

    def save(self):
        if not self.rows:
            return
        with open(self.log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)


# ══════════════════════════════════════════════════════════════════════════════
# Entrenamiento / Validación
# ══════════════════════════════════════════════════════════════════════════════

def run_epoch(model, loader, device, optimizer, scaler, criterion,
              train=True, gpu_monitor=None, epoch=0):
    model.train() if train else model.eval()

    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    n_batches = 0
    t0 = time.time()

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for step, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                logits = model(imgs)
                loss = criterion(logits, labels)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            n_batches += 1

            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            if gpu_monitor is not None and step % 20 == 0:
                gpu_monitor.sample(epoch, step)

    elapsed = time.time() - t0
    n_samples = len(all_labels)
    throughput = n_samples / elapsed if elapsed > 0 else 0.0

    metrics = {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "auc_roc": roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0,
        "n_samples": n_samples,
        "time_s": round(elapsed, 2),
        "throughput_img_s": round(throughput, 2),
    }
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Checkpointing
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(path, model, optimizer, scaler, epoch, best_auc):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_auc": best_auc,
    }, path)


def load_checkpoint(path, model, optimizer, scaler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt["epoch"], ckpt["best_auc"]


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    print("=" * 70)
    print("ENTRENAMIENTO ViT-B/16 — Real vs Fake Faces (StyleGAN3)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("\n📦 Cargando datasets...")
    train_ds = ChunkedNpyIterableDataset(PROCESSED_DIR / "train", shuffle=True, seed=42)
    val_ds = ChunkedNpyIterableDataset(PROCESSED_DIR / "val", shuffle=False)
    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                num_workers=args.num_workers, pin_memory=True,
                                persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                              num_workers=max(args.num_workers // 2, 1), pin_memory=True)

    # ── Modelo ────────────────────────────────────────────────────────────────
    print("\n🧠 Construyendo modelo...")
    model = build_model(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    start_epoch = 0
    best_auc = 0.0

    last_ckpt_path = CHECKPOINTS_DIR / "vit_last.pt"
    best_ckpt_path = CHECKPOINTS_DIR / "vit_best.pt"

    if args.resume and last_ckpt_path.exists():
        print(f"\n♻️  Reanudando desde {last_ckpt_path}...")
        start_epoch, best_auc = load_checkpoint(last_ckpt_path, model, optimizer, scaler, device)
        start_epoch += 1
        print(f"  Continuando desde época {start_epoch}, mejor AUC previo: {best_auc:.4f}")

    # ── Loop de entrenamiento ────────────────────────────────────────────────
    gpu_monitor = GPUMonitor(RESULTS_DIR / "gpu_stats.csv")
    log_path = RESULTS_DIR / "training_log.csv"
    log_rows = []

    if log_path.exists() and args.resume:
        with open(log_path) as f:
            log_rows = list(csv.DictReader(f))

    training_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        elapsed_min = (time.time() - training_start) / 60
        if elapsed_min > args.max_minutes:
            print(f"\n⏰ Límite de tiempo de seguridad alcanzado ({args.max_minutes} min). "
                  f"Guardando checkpoint y terminando.")
            save_checkpoint(last_ckpt_path, model, optimizer, scaler, epoch - 1, best_auc)
            break

        print(f"\n{'=' * 70}")
        print(f"ÉPOCA {epoch + 1}/{args.epochs}")
        print("=" * 70)

        train_metrics = run_epoch(model, train_loader, device, optimizer, scaler,
                                    criterion, train=True, gpu_monitor=gpu_monitor, epoch=epoch)
        print(f"  [TRAIN] loss={train_metrics['loss']:.4f}  acc={train_metrics['accuracy']:.4f}  "
              f"auc={train_metrics['auc_roc']:.4f}  {train_metrics['throughput_img_s']:.1f} img/s")

        val_metrics = run_epoch(model, val_loader, device, optimizer, scaler,
                                  criterion, train=False, epoch=epoch)
        print(f"  [VAL]   loss={val_metrics['loss']:.4f}  acc={val_metrics['accuracy']:.4f}  "
              f"auc={val_metrics['auc_roc']:.4f}  f1={val_metrics['f1']:.4f}")

        row = {"epoch": epoch + 1}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        log_rows.append(row)

        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerows(log_rows)

        save_checkpoint(last_ckpt_path, model, optimizer, scaler, epoch, best_auc)

        if val_metrics["auc_roc"] > best_auc:
            best_auc = val_metrics["auc_roc"]
            save_checkpoint(best_ckpt_path, model, optimizer, scaler, epoch, best_auc)
            print(f"  ⭐ Nuevo mejor modelo (AUC={best_auc:.4f}) guardado en {best_ckpt_path.name}")

    gpu_monitor.save()

    total_time = (time.time() - training_start) / 60
    print(f"\n{'=' * 70}")
    print(f"ENTRENAMIENTO FINALIZADO — {total_time:.1f} min totales")
    print(f"Mejor AUC-ROC en validación: {best_auc:.4f}")
    print("=" * 70)

    with open(RESULTS_DIR / "training_summary.json", "w") as f:
        json.dump({
            "epochs_completed": len(log_rows),
            "best_val_auc": best_auc,
            "total_time_minutes": round(total_time, 2),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
        }, f, indent=2)


if __name__ == "__main__":
    main()
