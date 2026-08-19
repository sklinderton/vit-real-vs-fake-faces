"""
generate_stylegan3.py
----------------------
Genera rostros sintéticos NUEVOS con el generador oficial preentrenado de
StyleGAN3 (NVIDIA, FFHQ 1024x1024) y evalúa el clasificador ViT sobre ellos.

Este es el experimento de generalización más fuerte del proyecto: las imágenes
generadas aquí NO existen en ningún dataset — se sintetizan desde semillas
aleatorias del espacio latente Z, distintas a las usadas para el dataset de
Kaggle con el que se entrenó el clasificador. Si el ViT las detecta como
"fake", significa que aprendió características generalizables de StyleGAN3,
no que memorizó las 10,000 imágenes específicas del entrenamiento.

Uso:
    python generate_stylegan3.py --num 100 --seed-start 100000

Salidas:
    data/stylegan3_generated/seed######.png   — rostros generados (256x256)
    results/stylegan3_generalization.csv       — predicción del ViT por imagen
    results/stylegan3_summary.json             — métricas agregadas
    figures/stylegan3_samples.png              — grilla de muestras con predicción
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

# ── Rutas ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/data/ulead-04/proyecto_paralela")
STYLEGAN3_REPO = Path("/data/ulead-04/stylegan3")
STYLEGAN3_PKL = Path("/data/ulead-04/stylegan3_models/stylegan3-r-ffhq-1024x1024.pkl")

SRC_DIR = PROJECT_ROOT / "src"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
GENERATED_DIR = PROJECT_ROOT / "data" / "stylegan3_generated"

for d in (RESULTS_DIR, FIGURES_DIR, GENERATED_DIR):
    d.mkdir(parents=True, exist_ok=True)

# StyleGAN3 necesita su propio repo en sys.path para deserializar el .pkl
sys.path.insert(0, str(STYLEGAN3_REPO))
sys.path.insert(0, str(SRC_DIR))

IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=100,
                    help="Cantidad de rostros nuevos a generar")
    p.add_argument("--seed-start", type=int, default=100000,
                    help="Semilla inicial (valores altos = fuera del rango típico de datasets públicos)")
    p.add_argument("--truncation", type=float, default=1.0,
                    help="Psi de truncamiento. 1.0 = máxima diversidad (más realista para el test), "
                         "valores <1 producen rostros más 'promedio' y fáciles")
    p.add_argument("--save-size", type=int, default=256,
                    help="Resolución a la que se guardan los PNG (el generador produce 1024x1024)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Cargar generador StyleGAN3
# ══════════════════════════════════════════════════════════════════════════════

def load_generator(device):
    print(f"📦 Cargando generador StyleGAN3 desde {STYLEGAN3_PKL.name}...")
    with open(STYLEGAN3_PKL, "rb") as f:
        G = pickle.load(f)["G_ema"].to(device)
    G.eval()
    print(f"  Resolución de salida : {G.img_resolution}x{G.img_resolution}")
    print(f"  Dimensión latente z  : {G.z_dim}")
    print(f"  Parámetros           : {sum(p.numel() for p in G.parameters()):,}")
    return G


# ══════════════════════════════════════════════════════════════════════════════
# Cargar clasificador ViT
# ══════════════════════════════════════════════════════════════════════════════

def load_classifier(device):
    ckpt_path = CHECKPOINTS_DIR / "vit_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No se encontró {ckpt_path}. Entrena primero con train_vit.py")

    model = vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, 2)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    print(f"🧠 Clasificador ViT cargado (época {ckpt['epoch'] + 1}, AUC-val {ckpt['best_auc']:.4f})")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Generación + clasificación
# ══════════════════════════════════════════════════════════════════════════════

def generate_and_classify(G, classifier, device, args):
    results = []
    sample_images = []   # guardamos las primeras 8 para la figura

    print(f"\n🎨 Generando {args.num} rostros nuevos (semillas {args.seed_start}..{args.seed_start + args.num - 1})")
    print(f"   Truncation psi = {args.truncation}")

    t0 = time.time()

    for i in range(args.num):
        seed = args.seed_start + i

        # ── Generar imagen desde una semilla del espacio latente ──────────
        z = torch.from_numpy(
            np.random.RandomState(seed).randn(1, G.z_dim)
        ).to(device)

        label = torch.zeros([1, G.c_dim], device=device)  # FFHQ no es condicional

        with torch.no_grad():
            img = G(z, label, truncation_psi=args.truncation, noise_mode="const")

        # StyleGAN3 devuelve [-1, 1] float -> convertir a uint8 [0, 255]
        img_uint8 = (img.clamp(-1, 1) + 1) * (255 / 2)
        img_uint8 = img_uint8.permute(0, 2, 3, 1).to(torch.uint8)[0].cpu().numpy()

        pil_img = Image.fromarray(img_uint8, "RGB")

        # Guardar en disco a resolución reducida (ahorra espacio, cuota limitada)
        save_img = pil_img.resize((args.save_size, args.save_size), Image.LANCZOS)
        out_path = GENERATED_DIR / f"seed{seed:06d}.png"
        save_img.save(out_path)

        # ── Clasificar con el ViT (mismo preprocesamiento del entrenamiento) ──
        img_224 = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        arr = np.asarray(img_224, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        arr = arr.transpose(2, 0, 1)
        tensor = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))

        with torch.no_grad():
            logits = classifier(tensor.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred = int(probs[1] > probs[0])   # 1 = predice Real, 0 = predice Fake

        results.append({
            "seed": seed,
            "filename": out_path.name,
            "p_fake": float(probs[0]),
            "p_real": float(probs[1]),
            "predicted": "Real" if pred == 1 else "Fake",
            "correct": pred == 0,          # ground truth: TODAS son fake
        })

        if len(sample_images) < 8:
            sample_images.append((save_img, probs, seed))

        if (i + 1) % 20 == 0:
            print(f"   {i + 1}/{args.num} generadas...")

    elapsed = time.time() - t0
    print(f"\n⏱️  Generación + clasificación: {elapsed:.1f}s ({args.num / elapsed:.2f} img/s)")

    return results, sample_images, elapsed


# ══════════════════════════════════════════════════════════════════════════════
# Reporte
# ══════════════════════════════════════════════════════════════════════════════

def report(results, sample_images, elapsed, args):
    import csv

    n = len(results)
    n_correct = sum(r["correct"] for r in results)
    detection_rate = n_correct / n
    mean_p_fake = float(np.mean([r["p_fake"] for r in results]))

    print("\n" + "=" * 70)
    print("RESULTADOS — GENERALIZACIÓN SOBRE ROSTROS STYLEGAN3 NUEVOS")
    print("=" * 70)
    print(f"  Imágenes generadas          : {n}")
    print(f"  Detectadas como Fake        : {n_correct} ({detection_rate:.1%})")
    print(f"  Clasificadas como Real (err): {n - n_correct} ({1 - detection_rate:.1%})")
    print(f"  P(fake) promedio            : {mean_p_fake:.4f}")
    print("=" * 70)
    print("\n  Nota: el ground truth es que TODAS las imágenes son sintéticas,")
    print("  por lo que la 'tasa de detección' equivale al recall sobre la clase Fake.")

    # CSV por imagen
    csv_path = RESULTS_DIR / "stylegan3_generalization.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  ✅ Detalle por imagen: {csv_path}")

    # JSON resumen
    summary = {
        "n_generated": n,
        "seed_range": [args.seed_start, args.seed_start + n - 1],
        "truncation_psi": args.truncation,
        "detection_rate_fake": detection_rate,
        "n_detected_fake": n_correct,
        "n_misclassified_real": n - n_correct,
        "mean_p_fake": mean_p_fake,
        "generation_time_s": round(elapsed, 2),
        "throughput_img_s": round(n / elapsed, 3),
        "generator": STYLEGAN3_PKL.name,
    }
    json_path = RESULTS_DIR / "stylegan3_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  ✅ Resumen: {json_path}")

    # Figura con muestras
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    fig.suptitle(
        "Rostros sintéticos NUEVOS generados con StyleGAN3 y clasificados por el ViT\n"
        "(ninguno existe en el dataset de entrenamiento)",
        fontsize=14, fontweight="bold"
    )

    for ax, (img, probs, seed) in zip(axes.flat, sample_images):
        ax.imshow(img)
        pred_fake = probs[0] > probs[1]
        color = "#2ca02c" if pred_fake else "#d62728"
        verdict = "FAKE ✓" if pred_fake else "REAL ✗"
        ax.set_title(f"seed {seed}\n{verdict}  P(fake)={probs[0]:.3f}",
                      fontsize=10, color=color, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    fig_path = FIGURES_DIR / "stylegan3_samples.png"
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Figura: {fig_path}")

    # Histograma de confianza
    fig2, ax = plt.subplots(figsize=(9, 5))
    p_fakes = [r["p_fake"] for r in results]
    ax.hist(p_fakes, bins=30, color="#1f77b4", edgecolor="white", alpha=0.85)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1.5,
                label="Umbral de decisión (0.5)")
    ax.set_xlabel("P(fake) asignada por el ViT")
    ax.set_ylabel("Cantidad de imágenes")
    ax.set_title(f"Distribución de confianza sobre {n} rostros StyleGAN3 nuevos\n"
                  f"Tasa de detección: {detection_rate:.1%}", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    hist_path = FIGURES_DIR / "stylegan3_confidence_hist.png"
    plt.savefig(hist_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Histograma: {hist_path}")


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("GENERACIÓN STYLEGAN3 + TEST DE GENERALIZACIÓN DEL ViT")
    print("=" * 70)
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    G = load_generator(device)
    classifier = load_classifier(device)

    results, samples, elapsed = generate_and_classify(G, classifier, device, args)
    report(results, samples, elapsed, args)


if __name__ == "__main__":
    main()
