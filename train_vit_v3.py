"""
train_vit_v3.py
----------------
Tercera iteración del entrenamiento. Corrige dos problemas de optimización
identificados en el análisis del run v2.

DIAGNÓSTICO DEL RUN v2
----------------------
  · El mejor checkpoint quedó en la ÉPOCA 1 (AUC-val 0.9263) y nunca mejoró.
  · El AUC de entrenamiento subió de 0.85 a 0.99 mientras el de validación
    se estancaba → el modelo memorizaba desde el primer momento.
  · Resultado: recall sobre fake de 93% en test pero 60.9% en test_hard.

Causa 1 — Learning rate excesivo para fine-tuning.
  Se usaba lr=1e-4 para ajustar los 85.8M parámetros de un ViT preentrenado.
  El rango habitual para fine-tuning completo de un ViT es 1e-5 a 3e-5. Con
  1e-4 los pesos preentrenados se degradan en la primera época y a partir de
  ahí el modelo solo sobreajusta.

Causa 2 — El criterio de early stopping medía la señal equivocada.
  El conjunto `val` comparte la distribución de truncation del entrenamiento
  (psi ~ U(0.4, 1.0)), así que el modelo podía verse estable acertando los
  casos fáciles (psi bajo) mientras fallaba en los difíciles (psi=1.0).

CORRECCIONES APLICADAS
----------------------
  1. lr = 2e-5 con planificador coseno y warmup lineal de 1 época.
  2. Early stopping guiado por el AUC sobre val_hard (psi=1.0), el escenario
     adverso que realmente queremos optimizar.
  3. Más épocas disponibles (20) y mayor paciencia (5): con lr bajo el modelo
     necesita más pasos para converger.
  4. Se registran ambas validaciones (val y val_hard) por época, lo que
     documenta la divergencia entre distribución fácil y difícil.

Uso:
    python train_vit_v3.py [--epochs N] [--lr F] [--patience N]
"""

import argparse
import csv
import json
import math
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
    confusion_matrix,
)

PROJECT_ROOT = Path("/data/ulead-04/proyecto_paralela")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_v2"
SRC_DIR = PROJECT_ROOT / "src"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC_DIR))
from preprocessing_worker import ChunkedNpyIterableDataset, ChunkedNpyDataset  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5,
                    help="LR pico. Rango recomendado para fine-tuning de ViT: 1e-5 a 3e-5")
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--max-minutes", type=float, default=200)
    return p.parse_args()


def build_model(device):
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads.head = nn.Linear(model.heads.head.in_features, 2)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ViT-B/16 | parámetros entrenables: {n:,}")
    return model.to(device)


def make_scheduler(optimizer, steps_per_epoch, args):
    """Warmup lineal seguido de decaimiento coseno."""
    total_steps = max(steps_per_epoch * args.epochs, 1)
    warmup_steps = max(int(steps_per_epoch * args.warmup_epochs), 1)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_train_epoch(model, loader, device, optimizer, scheduler, scaler, criterion):
    model.train()
    total_loss, n_batches = 0.0, 0
    preds, labels_all, probs = [], [], []
    t0 = time.time()

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16,
                             enabled=torch.cuda.is_available()):
            logits = model(imgs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1
        probs.extend(torch.softmax(logits, 1)[:, 1].detach().cpu().numpy().tolist())
        preds.extend(logits.argmax(1).detach().cpu().numpy().tolist())
        labels_all.extend(labels.cpu().numpy().tolist())

    elapsed = time.time() - t0
    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": accuracy_score(labels_all, preds),
        "auc_roc": roc_auc_score(labels_all, probs) if len(set(labels_all)) > 1 else 0.0,
        "time_s": round(elapsed, 2),
        "throughput_img_s": round(len(labels_all) / elapsed, 2) if elapsed else 0.0,
        "lr": optimizer.param_groups[0]["lr"],
    }


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    total_loss, n_batches = 0.0, 0
    preds, labels_all, probs = [], [], []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                             enabled=torch.cuda.is_available()):
            logits = model(imgs)
            if criterion is not None:
                total_loss += criterion(logits, labels).item()
                n_batches += 1
        probs.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy().tolist())
        preds.extend(logits.argmax(1).cpu().numpy().tolist())
        labels_all.extend(labels.cpu().numpy().tolist())

    cm = confusion_matrix(labels_all, preds)
    recall_fake = cm[0, 0] / cm[0].sum() if cm[0].sum() else 0.0
    recall_real = cm[1, 1] / cm[1].sum() if cm[1].sum() else 0.0

    return {
        "loss": total_loss / n_batches if n_batches else None,
        "accuracy": round(accuracy_score(labels_all, preds), 4),
        "precision": round(precision_score(labels_all, preds, zero_division=0), 4),
        "recall": round(recall_score(labels_all, preds, zero_division=0), 4),
        "f1": round(f1_score(labels_all, preds, zero_division=0), 4),
        "auc_roc": round(roc_auc_score(labels_all, probs), 4) if len(set(labels_all)) > 1 else 0.0,
        "recall_fake": round(float(recall_fake), 4),
        "recall_real": round(float(recall_real), 4),
        "n": len(labels_all),
        "confusion_matrix": cm.tolist(),
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("ENTRENAMIENTO ViT v3 — LR CORREGIDO + EARLY STOPPING SOBRE val_hard")
    print("=" * 72)
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  LR pico: {args.lr:.1e}  (v2 usaba 1e-4)  |  warmup: {args.warmup_epochs} época(s)")
    print(f"  Weight decay: {args.weight_decay}  |  label smoothing: {args.label_smoothing}")

    val_hard_dir = PROCESSED_DIR / "val_hard"
    if not val_hard_dir.exists():
        raise FileNotFoundError(
            "Falta data/processed_v2/val_hard. Ejecuta primero split_test_hard.py"
        )

    print("\n📦 Datasets:")
    train_ds = ChunkedNpyIterableDataset(PROCESSED_DIR / "train", shuffle=True, seed=42)
    val_ds = ChunkedNpyDataset(PROCESSED_DIR / "val")
    val_hard_ds = ChunkedNpyDataset(val_hard_dir)
    print(f"   train={len(train_ds):,}  val={len(val_ds):,}  val_hard={len(val_hard_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               num_workers=args.num_workers, pin_memory=True,
                               persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)
    val_hard_loader = DataLoader(val_hard_ds, batch_size=args.batch_size,
                                   shuffle=False, num_workers=4, pin_memory=True)

    print("\n🧠 Modelo:")
    model = build_model(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    steps_per_epoch = max(len(train_ds) // args.batch_size, 1)
    scheduler = make_scheduler(optimizer, steps_per_epoch, args)

    best_hard_auc, epochs_no_improve = 0.0, 0
    best_ckpt = CHECKPOINTS_DIR / "vit_v3_best.pt"
    log_path = RESULTS_DIR / "training_log_v3.csv"
    log_rows = []
    t_start = time.time()

    for epoch in range(args.epochs):
        if (time.time() - t_start) / 60 > args.max_minutes:
            print("\n⏰ Límite de tiempo alcanzado.")
            break

        print(f"\n{'=' * 72}\nÉPOCA {epoch + 1}/{args.epochs}\n{'=' * 72}")

        tr = run_train_epoch(model, train_loader, device, optimizer,
                              scheduler, scaler, criterion)
        print(f"  [TRAIN]     loss={tr['loss']:.4f} acc={tr['accuracy']:.4f} "
              f"auc={tr['auc_roc']:.4f}  lr={tr['lr']:.2e}  {tr['throughput_img_s']:.0f} img/s")

        va = evaluate(model, val_loader, device, criterion)
        print(f"  [VAL]       loss={va['loss']:.4f} acc={va['accuracy']:.4f} "
              f"auc={va['auc_roc']:.4f}  recall_fake={va['recall_fake']:.4f}")

        vh = evaluate(model, val_hard_loader, device, criterion)
        print(f"  [VAL_HARD]  loss={vh['loss']:.4f} acc={vh['accuracy']:.4f} "
              f"auc={vh['auc_roc']:.4f}  recall_fake={vh['recall_fake']:.4f}  ← criterio")

        row = {"epoch": epoch + 1, "lr": tr["lr"]}
        row.update({f"train_{k}": v for k, v in tr.items() if k != "lr"})
        row.update({f"val_{k}": v for k, v in va.items() if k != "confusion_matrix"})
        row.update({f"valhard_{k}": v for k, v in vh.items() if k != "confusion_matrix"})
        log_rows.append(row)
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            w.writeheader()
            w.writerows(log_rows)

        if vh["auc_roc"] > best_hard_auc:
            best_hard_auc = vh["auc_roc"]
            epochs_no_improve = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                         "best_hard_auc": best_hard_auc}, best_ckpt)
            print(f"  ⭐ Mejor modelo por AUC en val_hard = {best_hard_auc:.4f}")
        else:
            epochs_no_improve += 1
            print(f"  · Sin mejora en val_hard ({epochs_no_improve}/{args.patience})")
            if epochs_no_improve >= args.patience:
                print(f"\n🛑 Early stopping tras {args.patience} épocas sin mejora.")
                break

    total_min = (time.time() - t_start) / 60

    # ── Evaluación final ───────────────────────────────────────────────────
    print(f"\n{'=' * 72}\nEVALUACIÓN FINAL (mejor checkpoint según val_hard)\n{'=' * 72}")
    ck = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"])
    print(f"  Checkpoint de la época {ck['epoch'] + 1}")

    evals = {}
    for split in ("test", "test_hard_final"):
        d = PROCESSED_DIR / split
        if not d.exists():
            continue
        ds = ChunkedNpyDataset(d)
        ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        r = evaluate(model, ld, device)
        evals[split] = r
        print(f"\n  ── {split} (n={r['n']:,}) ──")
        print(f"     accuracy={r['accuracy']:.4f}  f1={r['f1']:.4f}  auc={r['auc_roc']:.4f}")
        print(f"     recall fake={r['recall_fake']:.4f}   recall real={r['recall_real']:.4f}")

    print(f"\n{'=' * 72}\nEVOLUCIÓN DEL PROYECTO — recall sobre FAKE\n{'=' * 72}")
    print(f"{'Versión':<10} {'test (fácil)':>14} {'test_hard':>14} {'brecha':>10}")
    print("-" * 72)
    print(f"{'v1':<10} {'82.0%':>14} {'23.0%':>14} {'-59.0pp':>10}")
    print(f"{'v2':<10} {'93.0%':>14} {'60.9%':>14} {'-32.1pp':>10}")
    if "test" in evals and "test_hard_final" in evals:
        t, h = evals["test"], evals["test_hard_final"]
        gap = (h["recall_fake"] - t["recall_fake"]) * 100
        print(f"{'v3':<10} {t['recall_fake']:>13.1%} {h['recall_fake']:>14.1%} {gap:>9.1f}pp")

    summary = {
        "version": "v3",
        "epochs_run": len(log_rows),
        "best_epoch": ck["epoch"] + 1,
        "best_val_hard_auc": best_hard_auc,
        "total_time_minutes": round(total_min, 2),
        "hyperparams": {
            "lr_peak": args.lr, "warmup_epochs": args.warmup_epochs,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "batch_size": args.batch_size, "patience": args.patience,
            "scheduler": "linear warmup + cosine decay",
            "early_stopping_criterion": "AUC-ROC sobre val_hard (psi=1.0)",
        },
        "evaluations": evals,
    }
    out = RESULTS_DIR / "evaluation_summary_v3.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  ✅ Guardado: {out}")
    print(f"  ⏱️  Tiempo total: {total_min:.1f} min")


if __name__ == "__main__":
    main()
