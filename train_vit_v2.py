"""
train_vit_v2.py
----------------
Entrenamiento del ViT-B/16 sobre el dataset v2, que corrige el sesgo de
truncation identificado experimentalmente en la versión anterior.

CAMBIOS RESPECTO A train_vit.py (v1)
------------------------------------
  1. Dataset: data/processed_v2 (fakes con psi ~ U(0.4, 1.0) además de los
     fakes originales de Kaggle, que usaban truncation bajo).
  2. Early stopping (paciencia configurable) sobre el AUC de validación:
     en v1 el modelo empezaba a sobreajustar desde la época ~10 y se
     entrenaban 15 sin beneficio.
  3. Label smoothing (0.05) para reducir la sobreconfianza observada en v1
     (loss de validación creciendo de 0.34 a 0.79 mientras el AUC se estancaba).
  4. Evaluación final sobre DOS conjuntos:
       - test       : distribución igual a la de entrenamiento
       - test_hard  : 1,000 reales reservados + 1,000 sintéticos a psi=1.0
                      con semillas disjuntas (prueba de generalización estricta)

La comparación test vs test_hard es el resultado central: mide si el modelo
aprendió características del generador o solo de la porción del espacio
latente que vio durante el entrenamiento.

Uso:
    python train_vit_v2.py [--epochs N] [--patience N] [--resume]
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
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=3,
                    help="Épocas sin mejora en AUC-val antes de detener")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-minutes", type=float, default=200)
    return p.parse_args()


def build_model(device):
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads.head = nn.Linear(model.heads.head.in_features, 2)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ViT-B/16 | parámetros entrenables: {n:,}")
    return model.to(device)


def run_epoch(model, loader, device, optimizer, scaler, criterion, train=True):
    model.train() if train else model.eval()

    total_loss, n_batches = 0.0, 0
    preds_all, labels_all, probs_all = [], [], []
    t0 = time.time()

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                 enabled=torch.cuda.is_available()):
                logits = model(imgs)
                loss = criterion(logits, labels)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            n_batches += 1

            probs_all.extend(torch.softmax(logits, 1)[:, 1].detach().cpu().numpy().tolist())
            preds_all.extend(logits.argmax(1).detach().cpu().numpy().tolist())
            labels_all.extend(labels.cpu().numpy().tolist())

    elapsed = time.time() - t0
    n = len(labels_all)

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": accuracy_score(labels_all, preds_all),
        "precision": precision_score(labels_all, preds_all, zero_division=0),
        "recall": recall_score(labels_all, preds_all, zero_division=0),
        "f1": f1_score(labels_all, preds_all, zero_division=0),
        "auc_roc": roc_auc_score(labels_all, probs_all) if len(set(labels_all)) > 1 else 0.0,
        "n_samples": n,
        "time_s": round(elapsed, 2),
        "throughput_img_s": round(n / elapsed, 2) if elapsed > 0 else 0.0,
    }


def evaluate_set(model, split_name, device, batch_size):
    """Evalúa un split map-style y devuelve métricas + recall por clase."""
    ds = ChunkedNpyDataset(PROCESSED_DIR / split_name)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    preds, labels, probs = [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, lbl in loader:
            logits = model(imgs.to(device))
            probs.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy().tolist())
            preds.extend(logits.argmax(1).cpu().numpy().tolist())
            labels.extend(lbl.numpy().tolist())

    cm = confusion_matrix(labels, preds)
    # recall por clase: fake=0, real=1
    recall_fake = cm[0, 0] / cm[0].sum() if cm[0].sum() else 0.0
    recall_real = cm[1, 1] / cm[1].sum() if cm[1].sum() else 0.0

    return {
        "split": split_name,
        "n": len(labels),
        "accuracy": round(accuracy_score(labels, preds), 4),
        "precision": round(precision_score(labels, preds, zero_division=0), 4),
        "recall": round(recall_score(labels, preds, zero_division=0), 4),
        "f1": round(f1_score(labels, preds, zero_division=0), 4),
        "auc_roc": round(roc_auc_score(labels, probs), 4),
        "recall_fake": round(float(recall_fake), 4),
        "recall_real": round(float(recall_real), 4),
        "confusion_matrix": cm.tolist(),
    }


def save_ckpt(path, model, optimizer, scaler, epoch, best_auc):
    torch.save({"epoch": epoch, "model_state": model.state_dict(),
                 "optimizer_state": optimizer.state_dict(),
                 "scaler_state": scaler.state_dict(), "best_auc": best_auc}, path)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("ENTRENAMIENTO ViT v2 — DATASET CON TRUNCATION DIVERSO")
    print("=" * 70)
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print("\n📦 Datasets:")
    train_ds = ChunkedNpyIterableDataset(PROCESSED_DIR / "train", shuffle=True, seed=42)
    val_ds = ChunkedNpyIterableDataset(PROCESSED_DIR / "val", shuffle=False)
    print(f"   train={len(train_ds):,}  val={len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               num_workers=args.num_workers, pin_memory=True,
                               persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                             num_workers=max(args.num_workers // 2, 1), pin_memory=True)

    print("\n🧠 Modelo:")
    model = build_model(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    best_auc, start_epoch, epochs_no_improve = 0.0, 0, 0
    last_ckpt = CHECKPOINTS_DIR / "vit_v2_last.pt"
    best_ckpt = CHECKPOINTS_DIR / "vit_v2_best.pt"

    if args.resume and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        scaler.load_state_dict(ck["scaler_state"])
        start_epoch, best_auc = ck["epoch"] + 1, ck["best_auc"]
        print(f"\n♻️  Reanudado desde época {start_epoch} (AUC previo {best_auc:.4f})")

    log_rows = []
    log_path = RESULTS_DIR / "training_log_v2.csv"
    t_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        if (time.time() - t_start) / 60 > args.max_minutes:
            print("\n⏰ Límite de tiempo alcanzado; guardando checkpoint.")
            save_ckpt(last_ckpt, model, optimizer, scaler, epoch - 1, best_auc)
            break

        print(f"\n{'=' * 70}\nÉPOCA {epoch + 1}/{args.epochs}\n{'=' * 70}")

        tr = run_epoch(model, train_loader, device, optimizer, scaler, criterion, True)
        print(f"  [TRAIN] loss={tr['loss']:.4f} acc={tr['accuracy']:.4f} "
              f"auc={tr['auc_roc']:.4f}  {tr['throughput_img_s']:.0f} img/s")

        va = run_epoch(model, val_loader, device, optimizer, scaler, criterion, False)
        print(f"  [VAL]   loss={va['loss']:.4f} acc={va['accuracy']:.4f} "
              f"auc={va['auc_roc']:.4f} f1={va['f1']:.4f}")

        row = {"epoch": epoch + 1}
        row.update({f"train_{k}": v for k, v in tr.items()})
        row.update({f"val_{k}": v for k, v in va.items()})
        log_rows.append(row)
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            w.writeheader()
            w.writerows(log_rows)

        save_ckpt(last_ckpt, model, optimizer, scaler, epoch, best_auc)

        if va["auc_roc"] > best_auc:
            best_auc = va["auc_roc"]
            epochs_no_improve = 0
            save_ckpt(best_ckpt, model, optimizer, scaler, epoch, best_auc)
            print(f"  ⭐ Mejor modelo (AUC={best_auc:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  · Sin mejora ({epochs_no_improve}/{args.patience})")
            if epochs_no_improve >= args.patience:
                print(f"\n🛑 Early stopping: {args.patience} épocas sin mejora.")
                break

    total_min = (time.time() - t_start) / 60

    # ── Evaluación final sobre test y test_hard ────────────────────────────
    print(f"\n{'=' * 70}")
    print("EVALUACIÓN FINAL (cargando el mejor checkpoint)")
    print("=" * 70)

    ck = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"])

    evals = {}
    for split in ("test", "test_hard"):
        if (PROCESSED_DIR / split).exists():
            r = evaluate_set(model, split, device, args.batch_size)
            evals[split] = r
            print(f"\n  ── {split} (n={r['n']:,}) ──")
            print(f"     accuracy={r['accuracy']:.4f}  f1={r['f1']:.4f}  auc={r['auc_roc']:.4f}")
            print(f"     recall fake={r['recall_fake']:.4f}   recall real={r['recall_real']:.4f}")

    print(f"\n{'=' * 70}")
    print("COMPARACIÓN CLAVE — ¿generaliza fuera de su distribución?")
    print("=" * 70)
    if "test" in evals and "test_hard" in evals:
        t, h = evals["test"], evals["test_hard"]
        gap = t["recall_fake"] - h["recall_fake"]
        print(f"  Recall sobre FAKE en test      : {t['recall_fake']:.1%}")
        print(f"  Recall sobre FAKE en test_hard : {h['recall_fake']:.1%}")
        print(f"  Brecha                          : {gap:+.1%}")
        print(f"\n  Referencia v1 (antes de la corrección): 82% → 23%  (brecha −59pp)")
        if abs(gap) < 0.10:
            print("  ✅ El modelo generaliza: la brecha se cerró.")
        elif abs(gap) < 0.25:
            print("  ⚠  Generalización parcial; queda una brecha moderada.")
        else:
            print("  ❌ Persiste una brecha grande; revisar composición del dataset.")

    summary = {
        "version": "v2",
        "epochs_run": len(log_rows),
        "best_val_auc": best_auc,
        "total_time_minutes": round(total_min, 2),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "hyperparams": {
            "batch_size": args.batch_size, "lr": args.lr,
            "label_smoothing": args.label_smoothing, "patience": args.patience,
        },
        "evaluations": evals,
    }
    out = RESULTS_DIR / "evaluation_summary_v2.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  ✅ Guardado: {out}")
    print(f"  ⏱️  Tiempo total: {total_min:.1f} min")


if __name__ == "__main__":
    main()
