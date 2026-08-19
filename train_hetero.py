"""
train_hetero.py
----------------
Entrena modelos de arquitectura distinta al ViT para diversificar el ensamble.

POR QUÉ
-------
El ensamble actual promedia tres ViT-B/16 que solo difieren en la semilla.
Comparten arquitectura, así que ven la imagen del mismo modo y se equivocan
en casos parecidos. Por eso pasar de un modelo (93.3 %) a tres (94.0 %)
rindió tan poco.

El diagnóstico lo confirma: incluso eligiendo el umbral mirando directamente
la prueba final —algo que no se debe hacer— el techo de ese ensamble es
94.4 %. El margen ya no está en dónde se corta la decisión sino en cuánta
información distinta aportan los miembros.

    ViT-B/16     parches de 16x16 comparados entre sí de forma global
    ConvNeXt     convoluciones jerárquicas, sensibles a textura local
    Swin-T       ventanas desplazadas, escala intermedia entre ambos

Las texturas de piel y los bordes del cabello, donde StyleGAN3 deja sus
rastros más persistentes, son precisamente lo que una red convolucional
capta mejor que un transformador de parches grandes. Sus errores deberían
solaparse menos con los del ViT.

TAMBIÉN SE CORRIGE
------------------
Los tres ViT pararon en las épocas 3, 4 y 5. El criterio de parada mira el
AUC sobre val_hard, que tiene 1 000 muestras y es ruidoso, de modo que
detenía el entrenamiento antes de tiempo. Aquí se sube la paciencia y se
promedian los pesos de las mejores épocas, que estabiliza el resultado sin
coste adicional de cómputo.

Uso:
    python train_hetero.py --arch convnext --tag cnx
    python train_hetero.py --arch swin     --tag swin
"""

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score,
)

PROJECT_ROOT = Path("/data/ulead-04/proyecto_paralela")
PROCESSED = PROJECT_ROOT / "data" / "processed_v4"
SRC_DIR = PROJECT_ROOT / "src"
CKPT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS = PROJECT_ROOT / "results"

CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC_DIR))
os.environ["VIT_IMG_SIZE"] = "256"
from preprocessing_worker import ChunkedNpyIterableDataset, ChunkedNpyDataset  # noqa: E402

IMG_SIZE = 256


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["convnext", "swin", "convnext_small"],
                    required=True)
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--swa-top", type=int, default=3,
                    help="Cuántas de las mejores épocas promediar al final")
    p.add_argument("--max-minutes", type=float, default=210)
    return p.parse_args()


def build_model(arch, device):
    """
    Construye la arquitectura pedida con pesos de ImageNet y dos salidas.

    ConvNeXt y Swin aceptan cualquier resolución divisible por su factor de
    reducción, así que a 256 px no hace falta interpolar nada, a diferencia
    del ViT.
    """
    import torchvision.models as tvm

    if arch == "convnext":
        m = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        in_f = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(in_f, 2)
        name = "ConvNeXt-Tiny"

    elif arch == "convnext_small":
        m = tvm.convnext_small(weights=tvm.ConvNeXt_Small_Weights.IMAGENET1K_V1)
        in_f = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(in_f, 2)
        name = "ConvNeXt-Small"

    else:  # swin
        # Swin-T v2 exige que la resolución coincida con la del preentrenamiento;
        # la v1 tolera entradas distintas, así que se usa esa.
        m = tvm.swin_t(weights=tvm.Swin_T_Weights.IMAGENET1K_V1)
        in_f = m.head.in_features
        m.head = nn.Linear(in_f, 2)
        name = "Swin-T"

    n = sum(p.numel() for p in m.parameters())
    print(f"  {name} · {n:,} parámetros")
    return m.to(device), name


def make_scheduler(opt, steps, args):
    total = max(steps * args.epochs, 1)
    warm = max(int(steps * args.warmup_epochs), 1)

    def f(s):
        if s < warm:
            return s / warm
        prog = (s - warm) / max(total - warm, 1)
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
            "throughput": round(len(labs) / el, 1) if el else 0.0,
            "lr": opt.param_groups[0]["lr"]}


@torch.no_grad()
def predict(model, loader, device, tta=True):
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


def metrics_at(probs, labels, thr=0.5):
    preds = (probs >= thr).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    rf = cm[0, 0] / cm[0].sum() if cm[0].sum() else 0.0
    rr = cm[1, 1] / cm[1].sum() if cm[1].sum() else 0.0
    return {"threshold": round(float(thr), 4),
            "accuracy": round(accuracy_score(labels, preds), 4),
            "precision": round(precision_score(labels, preds, zero_division=0), 4),
            "recall": round(recall_score(labels, preds, zero_division=0), 4),
            "f1": round(f1_score(labels, preds, zero_division=0), 4),
            "auc_roc": round(roc_auc_score(labels, probs), 4),
            "recall_fake": round(float(rf), 4),
            "recall_real": round(float(rr), 4),
            "n": int(len(labels)),
            "confusion_matrix": cm.tolist()}


def average_states(states):
    """
    Promedia varios diccionarios de pesos.

    Promediar los pesos de las mejores épocas suele dar un modelo algo más
    estable que quedarse con una sola, porque suaviza la posición final en el
    espacio de parámetros. Solo tiene sentido entre épocas cercanas del mismo
    entrenamiento, donde los pesos siguen siendo comparables.
    """
    avg = copy.deepcopy(states[0])
    for k in avg:
        if avg[k].dtype.is_floating_point:
            avg[k] = torch.stack([s[k].float() for s in states]).mean(0).to(avg[k].dtype)
    return avg


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 74)
    print(f"ENTRENAMIENTO HETEROGÉNEO — {args.arch.upper()}")
    print("=" * 74)
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Resolución {IMG_SIZE}px · lr {args.lr:.1e} · semilla {args.seed}")
    print(f"  Paciencia {args.patience} · promedia las {args.swa_top} mejores épocas")

    print("\n📦 Datos:")
    train_ds = ChunkedNpyIterableDataset(PROCESSED / "train", shuffle=True,
                                          seed=args.seed, augment=True)
    val_ds = ChunkedNpyDataset(PROCESSED / "val")
    vh_ds = ChunkedNpyDataset(PROCESSED / "val_hard")
    print(f"   train {len(train_ds):,} · val {len(val_ds):,} · val_hard {len(vh_ds):,}")

    train_ld = DataLoader(train_ds, batch_size=args.batch_size,
                           num_workers=args.num_workers, pin_memory=True,
                           persistent_workers=args.num_workers > 0)
    val_ld = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)
    vh_ld = DataLoader(vh_ds, batch_size=64, shuffle=False, num_workers=4)

    print("\n🧠 Modelo:")
    model, arch_name = build_model(args.arch, device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
    sched = make_scheduler(opt, max(len(train_ds) // args.batch_size, 1), args)

    best_auc, no_improve = 0.0, 0
    top_states = []          # (auc, epoch, state_dict)
    rows = []
    ckpt_path = CKPT_DIR / f"vit_v5_{args.tag}.pt"
    log_path = RESULTS / f"training_log_{args.tag}.csv"
    t0 = time.time()

    for ep in range(args.epochs):
        if (time.time() - t0) / 60 > args.max_minutes:
            print("\n⏰ Límite de tiempo alcanzado.")
            break

        print(f"\n{'=' * 74}\nÉPOCA {ep + 1}/{args.epochs}\n{'=' * 74}")
        tr = train_epoch(model, train_ld, device, opt, sched, scaler, crit)
        print(f"  [TRAIN]    loss {tr['loss']:.4f} · acc {tr['accuracy']:.4f} · "
              f"auc {tr['auc_roc']:.4f} · lr {tr['lr']:.2e} · {tr['throughput']:.0f} img/s")

        pv, lv = predict(model, val_ld, device)
        mv = metrics_at(pv, lv)
        print(f"  [VAL]      acc {mv['accuracy']:.4f} · auc {mv['auc_roc']:.4f}")

        ph, lh = predict(model, vh_ld, device)
        mh = metrics_at(ph, lh)
        print(f"  [VAL_HARD] acc {mh['accuracy']:.4f} · auc {mh['auc_roc']:.4f} · "
              f"detecta sintéticas {mh['recall_fake']:.4f}  ← criterio")

        rows.append({"epoch": ep + 1, "lr": tr["lr"],
                      "train_loss": tr["loss"], "train_accuracy": tr["accuracy"],
                      "train_auc_roc": tr["auc_roc"],
                      "train_throughput_img_s": tr["throughput"],
                      "val_accuracy": mv["accuracy"], "val_auc_roc": mv["auc_roc"],
                      "valhard_accuracy": mh["accuracy"],
                      "valhard_auc_roc": mh["auc_roc"],
                      "valhard_recall_fake": mh["recall_fake"]})
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

        # Guardar las mejores épocas para promediarlas al final
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        top_states.append((mh["auc_roc"], ep + 1, state))
        top_states.sort(key=lambda x: -x[0])
        top_states = top_states[:args.swa_top]

        if mh["auc_roc"] > best_auc:
            best_auc = mh["auc_roc"]
            no_improve = 0
            print(f"  ⭐ Mejor AUC en val_hard: {best_auc:.4f}")
        else:
            no_improve += 1
            print(f"  · Sin mejora ({no_improve}/{args.patience})")
            if no_improve >= args.patience:
                print("\n🛑 Early stopping.")
                break

    # ── Promedio de las mejores épocas ─────────────────────────────────────
    eps = [e for _, e, _ in top_states]
    print(f"\n{'=' * 74}\nPROMEDIANDO PESOS DE LAS ÉPOCAS {eps}\n{'=' * 74}")
    model.load_state_dict(average_states([s for _, _, s in top_states]))
    model.to(device)

    ph, lh = predict(model, vh_ld, device)
    m_avg = metrics_at(ph, lh)
    best_single = top_states[0][0]
    print(f"  AUC val_hard · mejor época sola {best_single:.4f} · "
          f"promedio {m_avg['auc_roc']:.4f}")

    if m_avg["auc_roc"] < best_single:
        print("  El promedio no mejoró; se conserva la mejor época individual.")
        model.load_state_dict(top_states[0][2])
        model.to(device)
        ph, lh = predict(model, vh_ld, device)
        m_avg = metrics_at(ph, lh)

    torch.save({"epoch": top_states[0][1], "arch": args.arch,
                 "model_state": model.state_dict(),
                 "best_hard_auc": m_avg["auc_roc"],
                 "averaged_epochs": eps}, ckpt_path)
    print(f"  ✅ {ckpt_path}")

    # ── Evaluación ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 74}\nEVALUACIÓN\n{'=' * 74}")
    evals = {}
    for split in ("test", "test_hard"):
        d = PROCESSED / split
        if not d.exists():
            continue
        ld = DataLoader(ChunkedNpyDataset(d), batch_size=64, shuffle=False,
                         num_workers=4)
        p, l = predict(model, ld, device)
        m = metrics_at(p, l)
        evals[split] = m
        print(f"\n  ── {split} (n={m['n']:,}) ──")
        print(f"     aciertos {m['accuracy']:.4f} · f1 {m['f1']:.4f} · "
              f"AUC {m['auc_roc']:.4f}")
        print(f"     detecta sintéticas {m['recall_fake']:.4f} · "
              f"auténticas {m['recall_real']:.4f}")

    total = (time.time() - t0) / 60
    with open(RESULTS / f"evaluation_summary_{args.tag}.json", "w") as f:
        json.dump({"arch": arch_name, "tag": args.tag, "seed": args.seed,
                    "image_size": IMG_SIZE, "epochs_run": len(rows),
                    "averaged_epochs": eps,
                    "best_val_hard_auc": m_avg["auc_roc"],
                    "total_time_minutes": round(total, 2),
                    "evaluations": evals}, f, indent=2)

    print(f"\n  ⏱️  {total:.1f} min")
    print(f"\n  Ahora vuelve a evaluar el ensamble incluyendo este modelo:")
    print(f"      python evaluate_ensemble.py --tags a b c {args.tag}")


if __name__ == "__main__":
    main()
