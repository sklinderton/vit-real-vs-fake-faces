"""
compare_generators.py
----------------------
Experimento decisivo para explicar la brecha de generalización del ViT
(82% de recall sobre fakes de Kaggle vs ~20% sobre StyleGAN3 generado).

Descartado previamente:
  - Sesgo de compresión JPEG  → todo el dataset es PNG sin pérdida (d=0.093)
  - Sesgo de remuestreo       → la detección es plana (~20%) con 6 filtros
                                distintos, pese a variar la energía HF de
                                6.38 a 7.32 (ablación, spread=3pp)

Hipótesis restante:
  Las imágenes "fake" del dataset de Kaggle NO provienen del mismo generador
  (o de la misma configuración) que stylegan3-r-ffhq. Este script contrasta
  las poblaciones en el dominio de frecuencia y barre valores de truncation.

Dos análisis:
  A) Comparación espectral: perfil de potencia radial (FFT 2D) de
     Kaggle-real vs Kaggle-fake vs StyleGAN3-generado. Si las fakes de Kaggle
     tienen una firma espectral que las generadas no comparten, confirma
     que son generadores distintos.
  B) Barrido de truncation psi: mide si generar con psi menor (caras más
     cercanas a la media del espacio latente) aumenta la detección.

Uso:
    python compare_generators.py --num 200
"""

import argparse
import json
import pickle
import sys
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
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

for d in (RESULTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(STYLEGAN3_REPO))

IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
N_BINS = 64
TRUNCATIONS = [0.4, 0.55, 0.7, 0.85, 1.0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=200)
    p.add_argument("--seed-start", type=int, default=300000)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Espectro radial
# ══════════════════════════════════════════════════════════════════════════════

def radial_spectrum(pil_img, n_bins=N_BINS):
    """Perfil de potencia radial promedio del espectro FFT 2D (256x256)."""
    gray = np.asarray(pil_img.convert("L").resize((256, 256)), dtype=np.float32)
    f = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log1p(np.abs(f))

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.hypot(x - cx, y - cy)
    r_max = r.max()

    edges = np.linspace(0, r_max, n_bins + 1)
    profile = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        mask = (r >= edges[i]) & (r < edges[i + 1])
        profile[i] = magnitude[mask].mean() if mask.any() else 0.0
    return profile


def load_folder_spectra(directory, limit):
    exts = ("*.png", "*.jpg", "*.jpeg")
    files = []
    for e in exts:
        files.extend(sorted(directory.glob(e)))
    files = files[:limit]
    profiles = []
    for fp in files:
        with Image.open(fp) as im:
            profiles.append(radial_spectrum(im.convert("RGB")))
    return np.array(profiles), len(files)


# ══════════════════════════════════════════════════════════════════════════════
# Modelos
# ══════════════════════════════════════════════════════════════════════════════

def load_generator(device):
    with open(STYLEGAN3_PKL, "rb") as f:
        G = pickle.load(f)["G_ema"].to(device)
    G.eval()
    return G


def load_classifier(device):
    model = vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, 2)
    ckpt = torch.load(CHECKPOINTS_DIR / "vit_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


def classify(pil_img, classifier, device):
    img_224 = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img_224, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    with torch.no_grad():
        probs = torch.softmax(classifier(t.unsqueeze(0).to(device)), dim=1)[0].cpu().numpy()
    return float(probs[0])


def gen_face(G, seed, psi, device):
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    c = torch.zeros([1, G.c_dim], device=device)
    with torch.no_grad():
        img = G(z, c, truncation_psi=psi, noise_mode="const")
    arr = ((img.clamp(-1, 1) + 1) * (255 / 2)).permute(0, 2, 3, 1).to(torch.uint8)[0].cpu().numpy()
    return Image.fromarray(arr, "RGB").resize((256, 256), Image.BICUBIC)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("COMPARACIÓN DE GENERADORES — ANÁLISIS ESPECTRAL Y TRUNCATION")
    print("=" * 70)

    G = load_generator(device)
    classifier = load_classifier(device)

    # ── A) Espectros de las tres poblaciones ───────────────────────────────
    print("\n📊 [A] Calculando espectros radiales...")

    spec_real, n_real = load_folder_spectra(RAW_DIR / "Real faces", args.num)
    print(f"   Kaggle real      : {n_real} imágenes")

    spec_fake_k, n_fake_k = load_folder_spectra(RAW_DIR / "Fake faces", args.num)
    print(f"   Kaggle fake      : {n_fake_k} imágenes")

    print(f"   StyleGAN3 gen    : generando {args.num}...")
    spec_gen = []
    for i in range(args.num):
        img = gen_face(G, args.seed_start + i, 1.0, device)
        spec_gen.append(radial_spectrum(img))
    spec_gen = np.array(spec_gen)

    m_real, m_fake_k, m_gen = spec_real.mean(0), spec_fake_k.mean(0), spec_gen.mean(0)

    # Distancia L2 entre perfiles medios (normalizada por nº de bins)
    d_real_fake = float(np.linalg.norm(m_real - m_fake_k) / N_BINS)
    d_fake_gen = float(np.linalg.norm(m_fake_k - m_gen) / N_BINS)
    d_real_gen = float(np.linalg.norm(m_real - m_gen) / N_BINS)

    print(f"\n   Distancia espectral (L2 normalizada entre perfiles medios):")
    print(f"     Kaggle-real  vs Kaggle-fake : {d_real_fake:.5f}")
    print(f"     Kaggle-fake  vs Generado    : {d_fake_gen:.5f}")
    print(f"     Kaggle-real  vs Generado    : {d_real_gen:.5f}")

    if d_fake_gen > d_real_fake * 1.5:
        print("\n   ⚠️  Las fakes de Kaggle están espectralmente MÁS LEJOS de las")
        print("       generadas que de las reales del propio dataset.")
        print("       → Fuerte indicio de que provienen de generadores distintos.")
    else:
        print("\n   ✓  Los espectros de fakes-Kaggle y generadas son comparables.")
        print("      El generador probablemente sí es el mismo o muy similar.")

    # ── B) Barrido de truncation ───────────────────────────────────────────
    print(f"\n📊 [B] Barrido de truncation psi ({args.num} imágenes por valor)...")
    trunc_rows = []
    for psi in TRUNCATIONS:
        p_fakes = []
        for i in range(args.num):
            img = gen_face(G, args.seed_start + 50000 + i, psi, device)
            p_fakes.append(classify(img, classifier, device))
        pf = np.array(p_fakes)
        det = float((pf > 0.5).mean())
        trunc_rows.append({
            "truncation_psi": psi,
            "detection_rate": round(det, 4),
            "mean_p_fake": round(float(pf.mean()), 4),
        })
        print(f"   psi={psi:<5} → detección {det:>6.1%}   P(fake) medio {pf.mean():.4f}")

    best_trunc = max(trunc_rows, key=lambda r: r["detection_rate"])
    trunc_spread = (max(r["detection_rate"] for r in trunc_rows)
                    - min(r["detection_rate"] for r in trunc_rows))

    print(f"\n   Mejor psi: {best_trunc['truncation_psi']} "
          f"({best_trunc['detection_rate']:.1%})   |   spread: {trunc_spread:.1%}")

    if trunc_spread > 0.20:
        print("   ⚠️  El truncation influye significativamente en la detección.")
    else:
        print("   ✓  El truncation NO explica la brecha (efecto pequeño).")

    # ── Guardar ────────────────────────────────────────────────────────────
    out = {
        "n_per_group": args.num,
        "spectral_distances": {
            "kaggle_real_vs_kaggle_fake": d_real_fake,
            "kaggle_fake_vs_generated": d_fake_gen,
            "kaggle_real_vs_generated": d_real_gen,
        },
        "truncation_sweep": trunc_rows,
        "truncation_spread": round(trunc_spread, 4),
    }
    out_path = RESULTS_DIR / "generator_comparison.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n   ✅ Guardado: {out_path}")

    # ── Figuras ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    freqs = np.arange(N_BINS)
    axes[0].plot(freqs, m_real, label=f"Kaggle real (n={n_real})", color="#2ca02c", lw=2)
    axes[0].plot(freqs, m_fake_k, label=f"Kaggle fake (n={n_fake_k})", color="#d62728", lw=2)
    axes[0].plot(freqs, m_gen, label=f"StyleGAN3 generado (n={args.num})",
                  color="#1f77b4", lw=2, linestyle="--")
    axes[0].set_xlabel("Anillo de frecuencia radial")
    axes[0].set_ylabel("log(1 + |FFT|) medio")
    axes[0].set_title("Perfil de potencia radial por población")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Diferencias respecto a Kaggle-real (resalta la firma de cada generador)
    axes[1].plot(freqs, m_fake_k - m_real, label="Kaggle fake − Kaggle real",
                  color="#d62728", lw=2)
    axes[1].plot(freqs, m_gen - m_real, label="Generado − Kaggle real",
                  color="#1f77b4", lw=2, linestyle="--")
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_xlabel("Anillo de frecuencia radial")
    axes[1].set_ylabel("Diferencia espectral")
    axes[1].set_title("Firma espectral relativa a las imágenes reales")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("¿Provienen las fakes de Kaggle del mismo generador?",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    p1 = FIGURES_DIR / "spectral_comparison.png"
    plt.savefig(p1, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Figura: {p1}")

    fig2, ax = plt.subplots(figsize=(8, 5))
    psis = [r["truncation_psi"] for r in trunc_rows]
    dets = [r["detection_rate"] for r in trunc_rows]
    ax.plot(psis, dets, marker="o", lw=2, color="#1f77b4")
    ax.axhline(0.5, color="red", linestyle="--", label="Azar")
    ax.set_xlabel("Truncation psi")
    ax.set_ylabel("Tasa de detección de 'Fake'")
    ax.set_title("Detección vs. truncation del espacio latente", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    for x, y in zip(psis, dets):
        ax.annotate(f"{y:.1%}", (x, y), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=9)
    plt.tight_layout()
    p2 = FIGURES_DIR / "truncation_sweep.png"
    plt.savefig(p2, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Figura: {p2}")


if __name__ == "__main__":
    main()
