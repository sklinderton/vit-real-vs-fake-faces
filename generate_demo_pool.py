"""
generate_demo_pool.py
----------------------
Genera un conjunto de rostros sintéticos con StyleGAN3 para alimentar el
dashboard interactivo, cubriendo distintos valores de truncation psi.

Se ejecuta en Kabré (GPU L40S, ~35 img/s). El dashboard, que corre en una
máquina local sin CUDA, muestrea de este pool en lugar de generar en vivo:
en CPU cada imagen tardaría entre 20 y 60 segundos, inviable para una demo.
Cada imagen queda registrada con su semilla y su psi, de modo que cualquiera
puede regenerarla de forma exacta y verificar que no fue seleccionada a mano.

IMPORTANTE — semillas disjuntas:
  Este pool usa semillas 700000+, distintas de las empleadas en el
  entrenamiento (400000–404499) y en los conjuntos de prueba (900000–900999).
  Así, ninguna imagen del dashboard fue vista por el modelo durante el ajuste.

Uso:
    python generate_demo_pool.py --num 300
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path("/data/ulead-04/proyecto_paralela")
STYLEGAN3_REPO = Path("/data/ulead-04/stylegan3")
STYLEGAN3_PKL = Path("/data/ulead-04/stylegan3_models/stylegan3-r-ffhq-1024x1024.pkl")
OUT_DIR = PROJECT_ROOT / "dashboard" / "assets" / "stylegan3_pool"

sys.path.insert(0, str(STYLEGAN3_REPO))

SEED_START = 700000
PSI_VALUES = [0.5, 0.7, 0.85, 1.0]   # se reparten equitativamente
SAVE_SIZE = 256


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=300)
    p.add_argument("--seed-start", type=int, default=SEED_START)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 66)
    print("GENERACIÓN DEL POOL DE DEMOSTRACIÓN — StyleGAN3")
    print("=" * 66)
    print(f"  Device: {device}")
    print(f"  Semillas: {args.seed_start}..{args.seed_start + args.num - 1}")
    print(f"  Valores de psi: {PSI_VALUES}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.png"):
        old.unlink()

    print("\n📦 Cargando generador...")
    with open(STYLEGAN3_PKL, "rb") as f:
        G = pickle.load(f)["G_ema"].to(device)
    G.eval()

    entries = []
    print(f"\n🎨 Generando {args.num} rostros...")

    for i in tqdm(range(args.num), unit="img"):
        seed = args.seed_start + i
        psi = PSI_VALUES[i % len(PSI_VALUES)]

        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        c = torch.zeros([1, G.c_dim], device=device)
        with torch.no_grad():
            img = G(z, c, truncation_psi=psi, noise_mode="const")

        arr = ((img.clamp(-1, 1) + 1) * (255 / 2))
        arr = arr.permute(0, 2, 3, 1).to(torch.uint8)[0].cpu().numpy()

        pil = Image.fromarray(arr, "RGB").resize((SAVE_SIZE, SAVE_SIZE), Image.BICUBIC)
        fname = f"sg3_seed{seed}_psi{psi:.2f}.png"
        pil.save(OUT_DIR / fname)

        entries.append({"filename": fname, "seed": seed, "psi": psi})

    manifest = {
        "generator": STYLEGAN3_PKL.name,
        "n_images": len(entries),
        "seed_range": [args.seed_start, args.seed_start + args.num - 1],
        "psi_values": PSI_VALUES,
        "save_size": SAVE_SIZE,
        "resize_pipeline": "1024 -> 256 (bicubic)",
        "note": ("Semillas disjuntas de entrenamiento (400000-404499) y "
                  "de los conjuntos de prueba (900000-900999)."),
        "images": entries,
    }
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    total_mb = sum(f.stat().st_size for f in OUT_DIR.glob("*.png")) / (1024 ** 2)
    print(f"\n  ✅ {len(entries)} imágenes en {OUT_DIR}")
    print(f"  📦 Tamaño total: {total_mb:.1f} MB")
    print(f"  📄 Manifiesto: {OUT_DIR / 'manifest.json'}")
    print("\n  Descarga la carpeta 'dashboard/' completa a tu PC para el Streamlit.")


if __name__ == "__main__":
    main()
