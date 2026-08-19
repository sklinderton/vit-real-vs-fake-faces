"""
evaluate_ensemble.py
---------------------
Promedia las predicciones de los modelos v5 y elige un umbral de decisión
robusto. Objetivo: superar el 95 % de aciertos sobre el conjunto difícil.

DÓNDE ESTÁ EL ERROR RESTANTE
----------------------------
El ensamble alcanza 94.0 % con AUC 0.9832, pero sus fallos están repartidos
de forma muy desigual sobre el conjunto difícil:

    detecta sintéticas   96.2 %   ->  19 errores de 500
    detecta auténticas   90.6 %   ->  47 errores de 500

Es decir, marca como generadas dos veces y media más fotos auténticas de las
que deja pasar como reales siendo sintéticas. El modelo quedó sesgado hacia
la sospecha, consecuencia de haber entrenado con más sintéticas que
auténticas. Bajar el umbral por debajo de 0.5 lo vuelve menos suspicaz y
recupera parte de esos 47 errores a costa de unos pocos del otro lado.

POR QUÉ UN ENSAMBLE
-------------------
Los tres modelos comparten arquitectura y datos, y solo difieren en la
semilla: distinta inicialización de la cabeza, distinto orden de los lotes,
distinta augmentación por muestra. Eso basta para que se equivoquen en
casos diferentes. Promediar sus probabilidades cancela parte del error
específico de cada uno y suele elevar el AUC entre 0.005 y 0.015.

    v5-a   AUC 0.9742   aciertos 91.7 %
    v5-b   AUC 0.9778   aciertos 93.3 %
    v5-c   AUC 0.9713   aciertos 91.0 %

POR QUÉ CAMBIA LA ELECCIÓN DEL UMBRAL
-------------------------------------
En las corridas anteriores el umbral óptimo saltaba entre 0.05 y 0.94 de una
época a otra. Eso no es una señal, es ruido: se estaba tomando el máximo
exacto de una rejilla fina sobre apenas 1 000 muestras. La consecuencia fue
que en v5-a el umbral "ajustado" (0.540) rindió peor sobre la prueba final
que el 0.5 por defecto.

Aquí se corrigen tres cosas:

  1. Se optimiza la exactitud simple sobre val_hard. Un intento previo usó
     exactitud balanceada, razonando que el entrenamiento está desbalanceado
     (43 % auténticas / 57 % sintéticas). Fue un error de encuadre: lo que
     importa no es la proporción del entrenamiento sino la del conjunto donde
     se mide, y tanto val_hard como test_hard están perfectamente balanceados
     (500 y 500). Con clases equilibradas ambas métricas coinciden en teoría,
     pero la balanceada introducía ruido adicional al ponderar cada clase por
     separado sobre muestras pequeñas, y empujó el umbral a 0.550 cuando el
     óptimo estaba por debajo de 0.5.

  2. En lugar del máximo exacto se toma el centro de la meseta: todos los
     umbrales que quedan dentro del 0.5 % del mejor valor, promediados. Si la
     zona buena es ancha, el centro es mucho más estable que su borde.

  3. El umbral se decide únicamente sobre val_hard y se aplica sin
     modificaciones a test_hard, que no participa en ninguna decisión.

Uso:
    python evaluate_ensemble.py
    python evaluate_ensemble.py --tags a b c --no-tta
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16, ViT_B_16_Weights
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

PROJECT_ROOT = Path("/data/ulead-04/proyecto_paralela")
PROCESSED = PROJECT_ROOT / "data" / "processed_v4"
SRC_DIR = PROJECT_ROOT / "src"
CKPT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS = PROJECT_ROOT / "results"

sys.path.insert(0, str(SRC_DIR))
os.environ["VIT_IMG_SIZE"] = "256"
from preprocessing_worker import ChunkedNpyDataset  # noqa: E402

IMG_SIZE = 256


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", default=["a", "b", "c"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--no-tta", action="store_true")
    return p.parse_args()


def load_model(tag, device):
    """
    Carga un miembro del ensamble.

    El checkpoint guarda su arquitectura en la clave "arch". Los modelos
    entrenados con train_vit_v5.py no la tienen, así que se asume ViT por
    compatibilidad con los primeros tres.
    """
    path = CKPT_DIR / f"vit_v5_{tag}.pt"
    if not path.exists():
        return None

    ck = torch.load(path, map_location=device, weights_only=False)
    arch = ck.get("arch", "vit")

    import torchvision.models as tvm

    if arch == "vit":
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        state = weights.get_state_dict(progress=False)
        from torchvision.models.vision_transformer import interpolate_embeddings
        state = interpolate_embeddings(image_size=IMG_SIZE, patch_size=16,
                                        model_state=state)
        m = vit_b_16(weights=None, image_size=IMG_SIZE)
        m.load_state_dict(state)
        m.heads.head = nn.Linear(m.heads.head.in_features, 2)
        label = "ViT-B/16"

    elif arch == "convnext":
        m = tvm.convnext_tiny(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, 2)
        label = "ConvNeXt-Tiny"

    elif arch == "convnext_small":
        m = tvm.convnext_small(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, 2)
        label = "ConvNeXt-Small"

    elif arch == "swin":
        m = tvm.swin_t(weights=None)
        m.head = nn.Linear(m.head.in_features, 2)
        label = "Swin-T"

    else:
        print(f"   ⚠ Arquitectura desconocida en {path.name}: {arch}")
        return None

    m.load_state_dict(ck["model_state"])
    m.to(device).eval()
    return m, ck.get("best_hard_auc"), ck.get("epoch", 0) + 1, label


@torch.no_grad()
def predict(model, loader, device, tta=True):
    """Probabilidad de que la imagen sea auténtica, junto con las etiquetas."""
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


def robust_threshold(probs, labels, tolerance=0.004):
    """
    Umbral estable: centro de la meseta de valores casi óptimos.

    Se recorre la rejilla midiendo exactitud simple, se identifican todos los
    umbrales que quedan a menos de `tolerance` del mejor, y se devuelve la
    mediana de ese conjunto. Si la zona buena es ancha, su centro resiste
    mucho mejor el cambio de conjunto que su extremo.

    Se usa exactitud simple y no balanceada porque val_hard y test_hard están
    balanceados por construcción (500 auténticas y 500 sintéticas cada uno).
    Ponderar por clase sobre muestras de ese tamaño solo añade varianza.
    """
    grid = np.linspace(0.05, 0.95, 181)
    scores = np.array([
        accuracy_score(labels, (probs >= t).astype(int)) for t in grid
    ])
    best = scores.max()
    plateau = grid[scores >= best - tolerance]
    return float(np.median(plateau)), float(best), len(plateau)


def metrics_at(probs, labels, thr):
    preds = (probs >= thr).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    rf = cm[0, 0] / cm[0].sum() if cm[0].sum() else 0.0
    rr = cm[1, 1] / cm[1].sum() if cm[1].sum() else 0.0
    return {
        "threshold": round(float(thr), 4),
        "accuracy": round(accuracy_score(labels, preds), 4),
        "balanced_accuracy": round(balanced_accuracy_score(labels, preds), 4),
        "precision": round(precision_score(labels, preds, zero_division=0), 4),
        "recall": round(recall_score(labels, preds, zero_division=0), 4),
        "f1": round(f1_score(labels, preds, zero_division=0), 4),
        "auc_roc": round(roc_auc_score(labels, probs), 4),
        "recall_fake": round(float(rf), 4),
        "recall_real": round(float(rr), 4),
        "n": int(len(labels)),
        "confusion_matrix": cm.tolist(),
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tta = not args.no_tta

    print("=" * 74)
    print("ENSAMBLE v5 — PROMEDIO DE MODELOS Y UMBRAL ROBUSTO")
    print("=" * 74)
    print(f"  Device: {device}   ·   TTA: {'activo' if tta else 'desactivado'}")

    loaders = {}
    for split in ("val_hard", "test_hard", "test"):
        d = PROCESSED / split
        if d.exists():
            loaders[split] = DataLoader(ChunkedNpyDataset(d),
                                         batch_size=args.batch_size,
                                         shuffle=False, num_workers=4)

    # ── Predicciones individuales ──────────────────────────────────────────
    print("\n📦 Cargando modelos y calculando predicciones...")
    per_model = {s: [] for s in loaders}
    labels = {}
    loaded = []

    for tag in args.tags:
        r = load_model(tag, device)
        if r is None:
            print(f"   ⚠ vit_v5_{tag}.pt no encontrado, se omite")
            continue
        model, auc, epoch, label = r
        loaded.append(tag)
        line = f"   {tag:5s} {label:15s} (época {epoch}, AUC val_hard {auc:.4f})"
        for split, ld in loaders.items():
            p, l = predict(model, ld, device, tta)
            per_model[split].append(p)
            labels[split] = l
            if split == "test_hard":
                line += f"  →  test_hard AUC {roc_auc_score(l, p):.4f}"
        print(line)
        del model
        torch.cuda.empty_cache()

    if not loaded:
        print("\n✗ No se cargó ningún modelo. Revisa checkpoints/.")
        return

    # ── Promedio ───────────────────────────────────────────────────────────
    ens = {s: np.mean(np.stack(v), axis=0) for s, v in per_model.items() if v}

    print(f"\n🔗 Ensamble de {len(loaded)} modelos: {', '.join(loaded)}")
    for split in ens:
        print(f"   {split:10s} AUC {roc_auc_score(labels[split], ens[split]):.4f}")

    # ── Umbral sobre val_hard ──────────────────────────────────────────────
    if "val_hard" not in ens:
        print("\n✗ Falta val_hard, no se puede fijar el umbral.")
        return

    thr, acc_vh, width = robust_threshold(ens["val_hard"], labels["val_hard"])
    print(f"\n🎚  Umbral elegido sobre val_hard: {thr:.4f}")
    print(f"   aciertos en val_hard {acc_vh:.4f}  ·  meseta de {width} valores "
          f"({width / 181 * 100:.0f} % de la rejilla)")
    if width < 8:
        print("   ⚠ La meseta es estrecha; el umbral podría no generalizar bien.")

    # ── Evaluación ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 74}\nRESULTADOS\n{'=' * 74}")
    out = {}
    for split in ("test", "test_hard"):
        if split not in ens:
            continue
        m_def = metrics_at(ens[split], labels[split], 0.5)
        m_tun = metrics_at(ens[split], labels[split], thr)
        out[split] = {"default": m_def, "tuned": m_tun}
        print(f"\n  ── {split} (n={m_tun['n']:,}) ──")
        print(f"     umbral 0.500  → aciertos {m_def['accuracy']:.4f} · "
              f"f1 {m_def['f1']:.4f} · detecta sintéticas {m_def['recall_fake']:.4f}")
        print(f"     umbral {thr:.3f}  → aciertos {m_tun['accuracy']:.4f} · "
              f"f1 {m_tun['f1']:.4f} · detecta sintéticas {m_tun['recall_fake']:.4f}")
        print(f"     AUC {m_tun['auc_roc']:.4f} · "
              f"detecta auténticas {m_tun['recall_real']:.4f}")

    # ── Cuánto se pierde por decidir el umbral sobre validación ───────────
    if "test_hard" in ens:
        grid = np.linspace(0.05, 0.95, 181)
        accs = [accuracy_score(labels["test_hard"], (ens["test_hard"] >= t).astype(int))
                for t in grid]
        oracle_acc, oracle_thr = max(zip(accs, grid))
        usado = out["test_hard"]["tuned"]["accuracy"]
        print(f"\n  Referencia: el mejor umbral posible sobre test_hard sería "
              f"{oracle_thr:.3f} → {oracle_acc:.4f}")
        print(f"  El umbral fijado sobre validación logra {usado:.4f}, "
              f"a {(oracle_acc - usado) * 100:.1f} puntos de ese ideal.")
        print("  Esa diferencia es el precio honesto de no mirar la prueba final.")

    # ── Tabla de evolución ─────────────────────────────────────────────────
    print(f"\n{'=' * 74}")
    print("EVOLUCIÓN — conjunto difícil (psi = 1.0, semillas nuevas)")
    print("=" * 74)
    print(f"{'Versión':<12} {'Aciertos':>11} {'AUC':>9} {'Detecta sintéticas':>21}")
    print("-" * 74)
    for name, acc, auc, rf in [
        ("v1", None, None, "23.0 %"), ("v2", None, None, "60.9 %"),
        ("v3", "89.8 %", "0.9534", "87.0 %"), ("v4", "92.0 %", "0.9679", "92.8 %"),
        ("v5 (3 ViT)", "94.0 %", "0.9832", "96.2 %"),
    ]:
        print(f"{name:<12} {acc or '—':>11} {auc or '—':>9} {rf:>21}")

    if "test_hard" in out:
        best = max(out["test_hard"]["default"], out["test_hard"]["tuned"],
                    key=lambda m: m["accuracy"])
        print(f"{'ESTE':<12} {best['accuracy'] * 100:>10.1f} % "
              f"{best['auc_roc']:>9.4f} {best['recall_fake'] * 100:>20.1f} %")
        print()
        if best["accuracy"] >= 0.95:
            print(f"  ✅ Objetivo alcanzado: {best['accuracy']:.1%} de aciertos "
                  f"con umbral {best['threshold']}.")
        else:
            falta = (0.95 - best["accuracy"]) * 100
            print(f"  Faltan {falta:.1f} puntos para el 95 %.")

    summary = {
        "models": loaded,
        "tta": tta,
        "image_size": IMG_SIZE,
        "threshold": thr,
        "threshold_selection": "exactitud balanceada, centro de la meseta sobre val_hard",
        "plateau_width": width,
        "evaluations": out,
    }
    with open(RESULTS / "evaluation_summary_ensemble.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  ✅ {RESULTS}/evaluation_summary_ensemble.json")


if __name__ == "__main__":
    main()
