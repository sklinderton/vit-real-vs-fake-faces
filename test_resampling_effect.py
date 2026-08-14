"""
test_resampling_effect.py
--------------------------
Experimento de ablación: mide cómo el FILTRO DE REMUESTREO usado al
redimensionar las imágenes generadas por StyleGAN3 (1024x1024 -> 256x256)
afecta la tasa de detección del clasificador ViT.

Contexto:
  El diagnóstico de compresión descartó un sesgo JPEG (todo el dataset es
  PNG sin pérdida). Sin embargo, la energía en altas frecuencias de las
  imágenes generadas con LANCZOS (6.8056) queda fuera del rango de ambas
  clases del dataset de entrenamiento (real=6.6726, fake=6.6064), con un
  tamaño de efecto grande (d≈0.80) respecto a las fakes de Kaggle.

Hipótesis:
  El dataset de Kaggle fue redimensionado a 256x256 con un filtro más
  suavizante que LANCZOS. El clasificador aprendió a asociar mayor nitidez
  con "real", por lo que las imágenes generadas con LANCZOS (más nítidas
  que todo lo visto en entrenamiento) se clasifican erróneamente como reales.

Predicción falsable:
  Si la hipótesis es correcta, la tasa de detección debe variar de forma
  sustancial según el filtro usado, y debe ser mayor con filtros más
  suavizantes (AREA, BILINEAR) que con LANCZOS.

Uso:
    python test_resampling_effect.py --num 200 --seed-start 100000
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.models import vit_b_16

PROJECT_ROOT = Path("/data/ulead-04/proyecto_paralela")
STYLEGAN3_REPO = Path("/data/ulead-04/stylegan3")
STYLEGAN3_PKL = Path("/data/ulead-04/stylegan3_models/stylegan3-r-ffhq-1024x1024.pkl")
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

for d in (RESULTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(STYLEGAN3_REPO))

IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Filtros a comparar. "none" = sin paso intermedio por 256x256:
# se va directo de 1024 a 224, como haría un pipeline sin el dataset de Kaggle.
RESAMPLING_METHODS = {
    "lanczos":  Image.LANCZOS,    # el usado originalmente (más nítido)
    "bicubic":  Image.BICUBIC,
    "bilinear": Image.BILINEAR,
    "area":     Image.BOX,        # promediado por área (más suavizante)
    "nearest":  Image.NEAREST,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=200)
    p.add_argument("--seed-start", type=int, default=100000)
    p.add_argument("--truncation", type=float, default=1.0)
    return p.parse_args()


def load_generator(device):
    print(f"📦 Cargando generador StyleGAN3...")
    with open(STYLEGAN3_PKL, "rb") as f:
        G = pickle.load(f)["G_ema"].to(device)
    G.eval()
    return G


def load_classifier(device):
    ckpt_path = CHECKPOINTS_DIR / "vit_best.pt"
    model = vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, 2)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"🧠 ViT cargado (época {ckpt['epoch']+1}, AUC-val {ckpt['best_auc']:.4f})")
    return model


def high_freq_energy(pil_img):
    """Energía media en el anillo de altas frecuencias del espectro FFT."""
    gray = np.asarray(pil_img.convert("L").resize((256, 256)), dtype=np.float32)
    f = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.abs(f)
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.hypot(x - cx, y - cy)
    mask = r > (r.max() * 0.4)
    return float(np.log1p(magnitude[mask]).mean())


def classify(pil_img, classifier, device):
    """Aplica el preprocesamiento del entrenamiento y clasifica."""
    img_224 = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img_224, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)
    tensor = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    with torch.no_grad():
        logits = classifier(tensor.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return float(probs[0]), float(probs[1])   # p_fake, p_real


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("EXPERIMENTO DE ABLACIÓN — EFECTO DEL FILTRO DE REMUESTREO")
    print("=" * 70)
    print(f"  Device: {device}")
    print(f"  Semillas: {args.seed_start}..{args.seed_start + args.num - 1}")
    print(f"  Filtros a comparar: {', '.join(RESAMPLING_METHODS)} + directo(1024->224)")

    G = load_generator(device)
    classifier = load_classifier(device)

    # Acumuladores por método
    stats = {name: {"p_fake": [], "hf": []} for name in RESAMPLING_METHODS}
    stats["direct_1024"] = {"p_fake": [], "hf": []}

    print(f"\n🎨 Generando y evaluando {args.num} rostros con cada filtro...")
    t0 = time.time()

    for i in range(args.num):
        seed = args.seed_start + i
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        label = torch.zeros([1, G.c_dim], device=device)

        with torch.no_grad():
            img = G(z, label, truncation_psi=args.truncation, noise_mode="const")

        img_uint8 = (img.clamp(-1, 1) + 1) * (255 / 2)
        img_uint8 = img_uint8.permute(0, 2, 3, 1).to(torch.uint8)[0].cpu().numpy()
        pil_1024 = Image.fromarray(img_uint8, "RGB")

        # ── Variante A: 1024 -> 256 (con cada filtro) -> clasificar ────────
        for name, resample in RESAMPLING_METHODS.items():
            img_256 = pil_1024.resize((256, 256), resample)
            p_fake, _ = classify(img_256, classifier, device)
            stats[name]["p_fake"].append(p_fake)
            stats[name]["hf"].append(high_freq_energy(img_256))

        # ── Variante B: 1024 -> 224 directo, sin pasar por 256 ─────────────
        p_fake, _ = classify(pil_1024, classifier, device)
        stats["direct_1024"]["p_fake"].append(p_fake)
        stats["direct_1024"]["hf"].append(high_freq_energy(pil_1024))

        if (i + 1) % 50 == 0:
            print(f"   {i + 1}/{args.num}...")

    elapsed = time.time() - t0
    print(f"\n⏱️  Completado en {elapsed:.1f}s")

    # ── Resultados ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTADOS POR FILTRO DE REMUESTREO")
    print("=" * 70)
    print(f"{'Filtro':<14} {'Detección':>11} {'P(fake) media':>15} {'Energía HF':>13}")
    print("-" * 70)

    rows = []
    for name, d in stats.items():
        pf = np.array(d["p_fake"])
        hf = np.array(d["hf"])
        detection = float((pf > 0.5).mean())
        rows.append({
            "method": name,
            "detection_rate": round(detection, 4),
            "mean_p_fake": round(float(pf.mean()), 4),
            "std_p_fake": round(float(pf.std()), 4),
            "mean_high_freq": round(float(hf.mean()), 4),
        })
        print(f"{name:<14} {detection:>10.1%} {pf.mean():>15.4f} {hf.mean():>13.4f}")

    print("-" * 70)
    print("  Referencia del dataset de entrenamiento:")
    print(f"  {'Real (Kaggle)':<14} {'—':>10} {'—':>15} {6.6726:>13.4f}")
    print(f"  {'Fake (Kaggle)':<14} {'—':>10} {'—':>15} {6.6064:>13.4f}")

    # ── Veredicto ──────────────────────────────────────────────────────────
    detections = [r["detection_rate"] for r in rows]
    spread = max(detections) - min(detections)
    best = max(rows, key=lambda r: r["detection_rate"])
    worst = min(rows, key=lambda r: r["detection_rate"])

    print("\n" + "=" * 70)
    print("  INTERPRETACIÓN")
    print("=" * 70)
    print(f"  Mejor filtro : {best['method']:<12} → {best['detection_rate']:.1%} detección")
    print(f"  Peor filtro  : {worst['method']:<12} → {worst['detection_rate']:.1%} detección")
    print(f"  Rango total  : {spread:.1%} puntos porcentuales de diferencia")

    if spread > 0.20:
        print("\n  ⚠️  HIPÓTESIS CONFIRMADA: el filtro de remuestreo altera drásticamente")
        print("      la predicción del modelo. Esto demuestra que el clasificador es")
        print("      sensible a artefactos del pipeline de procesamiento, no solo al")
        print("      contenido de la imagen. Se requiere augmentation de remuestreo")
        print("      aleatorio en el entrenamiento v2 para forzar invarianza.")
    elif spread > 0.10:
        print("\n  ⚠  Efecto moderado del remuestreo. Contribuye al problema pero")
        print("     probablemente no es la única causa de la brecha de generalización.")
    else:
        print("\n  ✓  El remuestreo NO explica la brecha de generalización.")
        print("     La causa debe buscarse en otro factor (distribución de semillas,")
        print("     post-procesado del dataset original, o el generador específico).")

    out = {
        "n_images": args.num,
        "seed_range": [args.seed_start, args.seed_start + args.num - 1],
        "truncation_psi": args.truncation,
        "results_by_method": rows,
        "detection_spread": round(spread, 4),
        "reference_high_freq": {"real_kaggle": 6.6726, "fake_kaggle": 6.6064},
    }
    out_path = RESULTS_DIR / "resampling_ablation.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  ✅ Guardado: {out_path}")

    # ── Figura ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = [r["method"] for r in rows]
    dets = [r["detection_rate"] for r in rows]
    hfs = [r["mean_high_freq"] for r in rows]

    colors = ["#2ca02c" if d > 0.5 else "#d62728" for d in dets]
    axes[0].bar(names, dets, color=colors, alpha=0.85)
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1.2, label="Azar")
    axes[0].set_ylabel("Tasa de detección de 'Fake'")
    axes[0].set_title("Detección según filtro de remuestreo")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend()
    for i, v in enumerate(dets):
        axes[0].text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=9)

    axes[1].bar(names, hfs, color="#1f77b4", alpha=0.85)
    axes[1].axhline(6.6726, color="#2ca02c", linestyle="--", label="Real (Kaggle)")
    axes[1].axhline(6.6064, color="#d62728", linestyle="--", label="Fake (Kaggle)")
    axes[1].set_ylabel("Energía en altas frecuencias")
    axes[1].set_title("Nitidez resultante vs. rango del dataset")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend()
    axes[1].set_ylim(min(hfs + [6.55]) - 0.05, max(hfs + [6.72]) + 0.05)

    plt.suptitle("Ablación: sensibilidad del ViT al pipeline de remuestreo",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = FIGURES_DIR / "resampling_ablation.png"
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Figura: {fig_path}")


if __name__ == "__main__":
    main()
