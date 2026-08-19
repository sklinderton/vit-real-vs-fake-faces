"""
train_vit_v4.py
----------------
Cuarta iteración. Objetivo: superar el 95 % de aciertos sobre el conjunto
difícil (rostros sintéticos con psi = 1.0 y semillas nunca vistas).

PUNTO DE PARTIDA (v3)
---------------------
    aciertos 89.8 %   ·   AUC 0.9534   ·   detecta sintéticas 87.0 %

La brecha entre un AUC de 0.9534 y unos aciertos de 89.8 % indica que parte
del margen no está en el modelo sino en dónde se corta la decisión: el
umbral 0.5 es una convención, no un óptimo.

CAMBIOS
-------
1. Datos (build_dataset_v3.py). La mitad de las sintéticas del entrenamiento
   se generan con psi ∈ [0.85, 1.0], el mismo régimen que mide la evaluación.
   Antes solo una sexta parte caía ahí.

2. Augmentación en tiempo de carga, idéntica para ambas clases: compresión
   JPEG de calidad variable, brillo, contraste, saturación, y desenfoque o
   realce. La compresión es la pieza clave — impide que el modelo se apoye en
   detalles de alta frecuencia que no sobreviven a una imagen real.

3. Umbral ajustado sobre val_hard. Se recorre el rango y se elige el corte
   que maximiza los aciertos, luego se aplica sin cambios a test_hard. El
   umbral se decide sobre validación, nunca sobre la prueba final.

4. Predicción sobre la imagen y su espejo, promediando ambas. Duplica el
   costo de inferencia y suele añadir uno o dos puntos.

Uso:
    python train_vit_v4.py [--epochs N] [--lr F] [--no-tta]
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
PROCESSED = PROJECT_ROOT / "data" / "processed_v3"
SRC_DIR = PROJECT_ROOT / "src"
CKPT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS = PROJECT_ROOT / "results"

CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC_DIR))
from preprocessing_worker import ChunkedNpyIterableDataset, ChunkedNpyDataset  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--no-tta", action="store_true",
                    help="Desactiva la predicción sobre la imagen y su espejo")
    p.add_argument("--max-minutes", type=float, default=210)
    return p.parse_args()


def build_model(device):
    m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    m.heads.head = nn.Linear(m.heads.head.in_features, 2)
    print(f"  ViT-B/16 · {sum(p.numel() for p in m.parameters()):,} parámetros")
    return m.to(device)


def make_scheduler(opt, steps_per_epoch, args):
    total = max(steps_per_epoch * args.epochs, 1)
    warm = max(int(steps_per_epoch * args.warmup_epochs), 1)

    def f(step):
        if step < warm:
            return step / warm
        prog = (step - warm) / max(total - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def train_epoch(model, loader, device, opt, sched, scaler, crit):
    model.train()
    tot, nb = 0.0, 0
    preds, labs, probs = [], [], []
    t0 = time.time()

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)

        with torch.autocast("cuda", dtype=torch.float16,
                             enabled=torch.cuda.is_available()):
            logits = model(imgs)
            loss = crit(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()

        tot += loss.item()
        nb += 1
        probs.extend(torch.softmax(logits, 1)[:, 1].detach().cpu().numpy().tolist())
        preds.extend(logits.argmax(1).detach().cpu().numpy().tolist())
        labs.extend(labels.cpu().numpy().tolist())

    el = time.time() - t0
    return {"loss": tot / max(nb, 1),
            "accuracy": accuracy_score(labs, preds),
            "auc_roc": roc_auc_score(labs, probs) if len(set(labs)) > 1 else 0.0,
            "time_s": round(el, 1),
            "throughput": round(len(labs) / el, 1) if el else 0.0,
            "lr": opt.param_groups[0]["lr"]}


@torch.no_grad()
def collect_probs(model, loader, device, tta=True):
    """
    Devuelve (probabilidad_de_real, etiquetas).

    Con tta activo promedia la predicción sobre la imagen original y su
    espejo horizontal. Es una transformación que no altera si un rostro es
    auténtico o generado, así que el promedio de ambas vistas reduce la
    varianza de la predicción sin introducir sesgo.
    """
    model.eval()
    probs, labs = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16,
                             enabled=torch.cuda.is_available()):
            p = torch.softmax(model(imgs), 1)[:, 1]
            if tta:
                p2 = torch.softmax(model(torch.flip(imgs, dims=[3])), 1)[:, 1]
                p = (p + p2) / 2
        probs.extend(p.float().cpu().numpy().tolist())
        labs.extend(labels.numpy().tolist())
    return np.array(probs), np.array(labs)


def metrics_at(probs_real, labels, thr):
    """Métricas con un umbral dado sobre la probabilidad de ser auténtica."""
    preds = (probs_real >= thr).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    rf = cm[0, 0] / cm[0].sum() if cm[0].sum() else 0.0
    rr = cm[1, 1] / cm[1].sum() if cm[1].sum() else 0.0
    return {
        "threshold": round(float(thr), 4),
        "accuracy": round(accuracy_score(labels, preds), 4),
        "precision": round(precision_score(labels, preds, zero_division=0), 4),
        "recall": round(recall_score(labels, preds, zero_division=0), 4),
        "f1": round(f1_score(labels, preds, zero_division=0), 4),
        "auc_roc": round(roc_auc_score(labels, probs_real), 4) if len(set(labels)) > 1 else 0.0,
        "recall_fake": round(float(rf), 4),
        "recall_real": round(float(rr), 4),
        "n": int(len(labels)),
        "confusion_matrix": cm.tolist(),
    }


def best_threshold(probs_real, labels):
    """Umbral que maximiza los aciertos. Se decide solo sobre validación."""
    grid = np.linspace(0.05, 0.95, 181)
    accs = [(accuracy_score(labels, (probs_real >= t).astype(int)), t) for t in grid]
    acc, thr = max(accs)
    return float(thr), float(acc)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tta = not args.no_tta

    print("=" * 74)
    print("ENTRENAMIENTO v4 — DATOS DIFÍCILES, AUGMENTACIÓN, TTA Y UMBRAL")
    print("=" * 74)
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  LR {args.lr:.1e} · weight decay {args.weight_decay} · "
          f"label smoothing {args.label_smoothing}")
    print(f"  Augmentación: activa (JPEG, brillo, contraste, desenfoque)")
    print(f"  TTA: {'activo' if tta else 'desactivado'}")

    if not (PROCESSED / "val_hard").exists():
        raise FileNotFoundError(
            f"Falta {PROCESSED / 'val_hard'}. Ejecuta antes build_dataset_v3.py")

    print("\n📦 Datos:")
    train_ds = ChunkedNpyIterableDataset(PROCESSED / "train", shuffle=True,
                                          seed=42, augment=True)
    val_ds = ChunkedNpyDataset(PROCESSED / "val")
    vh_ds = ChunkedNpyDataset(PROCESSED / "val_hard")
    print(f"   train {len(train_ds):,} (con augmentación) · "
          f"val {len(val_ds):,} · val_hard {len(vh_ds):,}")

    train_ld = DataLoader(train_ds, batch_size=args.batch_size,
                           num_workers=args.num_workers, pin_memory=True,
                           persistent_workers=args.num_workers > 0)
    val_ld = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4,
                         pin_memory=True)
    vh_ld = DataLoader(vh_ds, batch_size=64, shuffle=False, num_workers=4,
                        pin_memory=True)

    print("\n🧠 Modelo:")
    model = build_model(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
    sched = make_scheduler(opt, max(len(train_ds) // args.batch_size, 1), args)

    best_auc, no_improve = 0.0, 0
    best_ckpt = CKPT_DIR / "vit_v4_best.pt"
    log_path = RESULTS / "training_log_v4.csv"
    rows = []
    t_start = time.time()

    for ep in range(args.epochs):
        if (time.time() - t_start) / 60 > args.max_minutes:
            print("\n⏰ Límite de tiempo alcanzado.")
            break

        print(f"\n{'=' * 74}\nÉPOCA {ep + 1}/{args.epochs}\n{'=' * 74}")
        tr = train_epoch(model, train_ld, device, opt, sched, scaler, crit)
        print(f"  [TRAIN]    loss {tr['loss']:.4f} · acc {tr['accuracy']:.4f} · "
              f"auc {tr['auc_roc']:.4f} · lr {tr['lr']:.2e} · {tr['throughput']:.0f} img/s")

        pv, lv = collect_probs(model, val_ld, device, tta=tta)
        mv = metrics_at(pv, lv, 0.5)
        print(f"  [VAL]      acc {mv['accuracy']:.4f} · auc {mv['auc_roc']:.4f}")

        ph, lh = collect_probs(model, vh_ld, device, tta=tta)
        mh = metrics_at(ph, lh, 0.5)
        thr, acc_thr = best_threshold(ph, lh)
        print(f"  [VAL_HARD] acc {mh['accuracy']:.4f} · auc {mh['auc_roc']:.4f} · "
              f"detecta sintéticas {mh['recall_fake']:.4f}")
        print(f"             con umbral {thr:.3f} → acc {acc_thr:.4f}  ← criterio")

        rows.append({"epoch": ep + 1, "lr": tr["lr"],
                      "train_loss": tr["loss"], "train_accuracy": tr["accuracy"],
                      "train_auc_roc": tr["auc_roc"],
                      "train_throughput_img_s": tr["throughput"],
                      "val_accuracy": mv["accuracy"], "val_auc_roc": mv["auc_roc"],
                      "valhard_accuracy": mh["accuracy"],
                      "valhard_auc_roc": mh["auc_roc"],
                      "valhard_recall_fake": mh["recall_fake"],
                      "valhard_best_threshold": thr,
                      "valhard_accuracy_tuned": round(acc_thr, 4)})
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

        if mh["auc_roc"] > best_auc:
            best_auc = mh["auc_roc"]
            no_improve = 0
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                         "best_hard_auc": best_auc, "threshold": thr,
                         "tta": tta}, best_ckpt)
            print(f"  ⭐ Mejor AUC en val_hard: {best_auc:.4f} (umbral {thr:.3f})")
        else:
            no_improve += 1
            print(f"  · Sin mejora ({no_improve}/{args.patience})")
            if no_improve >= args.patience:
                print(f"\n🛑 Early stopping.")
                break

    total_min = (time.time() - t_start) / 60

    # ── Evaluación final ───────────────────────────────────────────────────
    print(f"\n{'=' * 74}\nEVALUACIÓN FINAL\n{'=' * 74}")
    ck = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"])
    print(f"  Checkpoint de la época {ck['epoch'] + 1}")

    # Umbral fijado sobre val_hard, nunca sobre la prueba final
    ph, lh = collect_probs(model, vh_ld, device, tta=tta)
    thr, _ = best_threshold(ph, lh)
    print(f"  Umbral elegido sobre val_hard: {thr:.4f}")

    evals = {}
    for split in ("test", "test_hard"):
        d = PROCESSED / split
        if not d.exists():
            continue
        ld = DataLoader(ChunkedNpyDataset(d), batch_size=64, shuffle=False,
                         num_workers=4)
        p, l = collect_probs(model, ld, device, tta=tta)
        m_default = metrics_at(p, l, 0.5)
        m_tuned = metrics_at(p, l, thr)
        evals[split] = {"default": m_default, "tuned": m_tuned}
        print(f"\n  ── {split} (n={m_tuned['n']:,}) ──")
        print(f"     umbral 0.5   → acc {m_default['accuracy']:.4f} · "
              f"f1 {m_default['f1']:.4f} · detecta sintéticas {m_default['recall_fake']:.4f}")
        print(f"     umbral {thr:.3f} → acc {m_tuned['accuracy']:.4f} · "
              f"f1 {m_tuned['f1']:.4f} · detecta sintéticas {m_tuned['recall_fake']:.4f}")
        print(f"     AUC {m_tuned['auc_roc']:.4f}")

    print(f"\n{'=' * 74}\nEVOLUCIÓN — conjunto difícil (psi = 1.0, semillas nuevas)\n{'=' * 74}")
    print(f"{'Versión':<8} {'Aciertos':>11} {'AUC':>9} {'Detecta sintéticas':>21}")
    print("-" * 74)
    print(f"{'v1':<8} {'—':>11} {'—':>9} {'23.0 %':>21}")
    print(f"{'v2':<8} {'—':>11} {'—':>9} {'60.9 %':>21}")
    print(f"{'v3':<8} {'89.8 %':>11} {'0.9534':>9} {'87.0 %':>21}")
    if "test_hard" in evals:
        t = evals["test_hard"]["tuned"]
        print(f"{'v4':<8} {t['accuracy'] * 100:>10.1f} % {t['auc_roc']:>9.4f} "
              f"{t['recall_fake'] * 100:>20.1f} %")
        if t["accuracy"] >= 0.95:
            print("\n  ✅ Objetivo alcanzado: 95 % o más de aciertos.")
        else:
            falta = (0.95 - t["accuracy"]) * 100
            print(f"\n  Faltan {falta:.1f} puntos para el 95 %.")

    summary = {
        "version": "v4",
        "epochs_run": len(rows),
        "best_epoch": ck["epoch"] + 1,
        "best_val_hard_auc": best_auc,
        "decision_threshold": thr,
        "tta": tta,
        "total_time_minutes": round(total_min, 2),
        "hyperparams": {
            "lr_peak": args.lr, "warmup_epochs": args.warmup_epochs,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "batch_size": args.batch_size, "patience": args.patience,
            "augmentation": "flip, JPEG 70-100, brillo, contraste, saturación, desenfoque/realce",
            "dataset": "processed_v3 (psi estratificado, 50 % en [0.85, 1.0])",
        },
        "evaluations": evals,
    }
    with open(RESULTS / "evaluation_summary_v4.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  ✅ {RESULTS / 'evaluation_summary_v4.json'}")
    print(f"  ⏱️  {total_min:.1f} min")


if __name__ == "__main__":
    main()
