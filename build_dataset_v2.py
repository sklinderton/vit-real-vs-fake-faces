"""
build_dataset_v2.py
--------------------
Construye el dataset v2 corrigiendo el sesgo de truncation detectado
experimentalmente en la v1.

DIAGNÓSTICO QUE MOTIVA ESTA VERSIÓN
-----------------------------------
El clasificador v1 alcanzaba 82% de recall sobre fakes del dataset de Kaggle
pero solo 23% sobre rostros StyleGAN3 generados con truncation psi=1.0.
Tres hipótesis fueron evaluadas experimentalmente:

  1. Sesgo de compresión JPEG   → DESCARTADA (todo es PNG sin pérdida, d=0.093)
  2. Sesgo de filtro de remuestreo → DESCARTADA (detección plana ~20% con
                                      6 filtros distintos, spread=3pp)
  3. Sesgo de truncation psi     → CONFIRMADA (spread=77pp)

     psi=0.40 → 100.0% detección      psi=0.85 →  49.5%
     psi=0.55 →  97.5%                psi=1.00 →  23.0%
     psi=0.70 →  76.5%

El análisis espectral confirmó además que el generador SÍ es el mismo
(distancia Kaggle-fake vs generado = 0.00733, menor que Kaggle-fake vs
Kaggle-real = 0.01055). El dataset de Kaggle simplemente fue generado con
truncation bajo, produciendo rostros "promedio" y suaves. El modelo aprendió
"cara suavizada = falsa", una regla correcta pero incompleta.

CORRECCIÓN APLICADA
-------------------
Los fakes sintéticos se generan con psi muestreado uniformemente en [0.4, 1.0],
cubriendo todo el rango de salida del generador en lugar de una sola esquina.

COMPOSICIÓN
-----------
  Conjunto principal (18,000 imgs, split 80/10/10):
      9,000 reales   (Kaggle)
      4,500 falsas   (Kaggle, truncation bajo original)
      4,500 falsas   (StyleGAN3 generado, psi ~ U(0.4, 1.0))

  Test difícil (2,000 imgs) — prueba de generalización estricta:
      1,000 reales   (Kaggle, reservados, nunca vistos en entrenamiento)
      1,000 falsas   (StyleGAN3, psi=1.0, semillas nunca vistas)

Los PNG generados no se guardan: las semillas son deterministas y quedan
registradas en data/metadata/seeds_manifest_v2.json, lo que permite
regenerar exactamente el mismo conjunto sin coste de almacenamiento.

Uso:
    python build_dataset_v2.py
"""

import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path("/data/ulead-04/proyecto_paralela")
STYLEGAN3_REPO = Path("/data/ulead-04/stylegan3")
STYLEGAN3_PKL = Path("/data/ulead-04/stylegan3_models/stylegan3-r-ffhq-1024x1024.pkl")
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "processed_v2"
META_DIR = PROJECT_ROOT / "data" / "metadata"

sys.path.insert(0, str(STYLEGAN3_REPO))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from preprocessing_worker import preprocess_image  # noqa: E402

# ── Parámetros ──────────────────────────────────────────────────────────────
IMG_SIZE = 224
N_WORKERS = 16
CHUNK_SIZE = 2000
SEED = 42

N_REAL_MAIN = 9000        # reales para train/val/test
N_REAL_HARD = 1000        # reales reservados para el test difícil
N_FAKE_KAGGLE = 4500      # fakes originales de Kaggle
N_FAKE_GEN = 4500         # fakes generados con psi diverso
N_HARD_GEN = 1000         # fakes del test difícil (psi=1.0)

PSI_MIN, PSI_MAX = 0.4, 1.0
SEED_GEN_START = 400000   # semillas para el pool de entrenamiento
SEED_HARD_START = 900000  # semillas del test difícil (disjuntas)

SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}


# ══════════════════════════════════════════════════════════════════════════════
# Generación con StyleGAN3
# ══════════════════════════════════════════════════════════════════════════════

def load_generator(device):
    print("📦 Cargando generador StyleGAN3...")
    with open(STYLEGAN3_PKL, "rb") as f:
        G = pickle.load(f)["G_ema"].to(device)
    G.eval()
    return G


def generate_batch(G, seeds, psis, device, desc):
    """
    Genera rostros y los devuelve como arrays uint8 (C,H,W) a 224x224.

    Replica el pipeline del dataset de Kaggle: 1024 -> 256 (bicúbico) -> 224,
    para que las imágenes sintéticas y las del dataset compartan el mismo
    número de pasos de remuestreo.
    """
    arrays = []
    for seed, psi in tqdm(list(zip(seeds, psis)), desc=desc, unit="img"):
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        c = torch.zeros([1, G.c_dim], device=device)
        with torch.no_grad():
            img = G(z, c, truncation_psi=float(psi), noise_mode="const")

        arr = ((img.clamp(-1, 1) + 1) * (255 / 2))
        arr = arr.permute(0, 2, 3, 1).to(torch.uint8)[0].cpu().numpy()

        pil = Image.fromarray(arr, "RGB")
        pil = pil.resize((256, 256), Image.BICUBIC)     # igual que Kaggle
        pil = pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

        a = np.asarray(pil, dtype=np.uint8).transpose(2, 0, 1)
        arrays.append(np.ascontiguousarray(a))

    return arrays


# ══════════════════════════════════════════════════════════════════════════════
# Preprocesamiento paralelo de imágenes de Kaggle
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_kaggle(paths, labels, desc):
    """Procesa imágenes de disco en paralelo, devolviendo arrays uint8."""
    args_list = [(str(p), int(l), False) for p, l in zip(paths, labels)]
    out = [None] * len(args_list)

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(preprocess_image, a): i for i, a in enumerate(args_list)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="img"):
            i = futures[fut]
            arr, lbl = fut.result()
            if arr is not None:
                out[i] = (arr, lbl)

    return [o for o in out if o is not None]


# ══════════════════════════════════════════════════════════════════════════════
# Guardado en chunks
# ══════════════════════════════════════════════════════════════════════════════

def save_split(items, split_name):
    """items: lista de (array_uint8_CHW, label)."""
    split_dir = OUT_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    n = len(items)
    chunk_id = 0
    for start in range(0, n, CHUNK_SIZE):
        chunk = items[start:start + CHUNK_SIZE]
        images = np.stack([c[0] for c in chunk]).astype(np.uint8)
        labels = np.array([c[1] for c in chunk], dtype=np.int8)
        np.save(split_dir / f"images_chunk{chunk_id:03d}.npy", images)
        np.save(split_dir / f"labels_chunk{chunk_id:03d}.npy", labels)
        chunk_id += 1

    n_real = sum(1 for _, l in items if l == 1)
    print(f"   {split_name:10s}: {n:>6,} imgs "
          f"(real={n_real:,}, fake={n - n_real:,}) → {chunk_id} chunks")
    return {"split": split_name, "n": n, "n_real": n_real,
            "n_fake": n - n_real, "n_chunks": chunk_id}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    rng = np.random.default_rng(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("CONSTRUCCIÓN DEL DATASET v2 — CORRECCIÓN DEL SESGO DE TRUNCATION")
    print("=" * 70)
    print(f"  Device: {device}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Inventario de imágenes de Kaggle ────────────────────────────────
    print("\n📂 Escaneando dataset de Kaggle...")
    real_files = sorted((RAW_DIR / "Real faces").glob("*.png")) + \
                 sorted((RAW_DIR / "Real faces").glob("*.jpg"))
    fake_files = sorted((RAW_DIR / "Fake faces").glob("*.png")) + \
                 sorted((RAW_DIR / "Fake faces").glob("*.jpg"))
    print(f"   Reales disponibles : {len(real_files):,}")
    print(f"   Falsas disponibles : {len(fake_files):,}")

    assert len(real_files) >= N_REAL_MAIN + N_REAL_HARD, "Faltan imágenes reales"
    assert len(fake_files) >= N_FAKE_KAGGLE, "Faltan imágenes falsas de Kaggle"

    real_idx = rng.permutation(len(real_files))
    real_main = [real_files[i] for i in real_idx[:N_REAL_MAIN]]
    real_hard = [real_files[i] for i in real_idx[N_REAL_MAIN:N_REAL_MAIN + N_REAL_HARD]]

    fake_idx = rng.permutation(len(fake_files))
    fake_kaggle = [fake_files[i] for i in fake_idx[:N_FAKE_KAGGLE]]

    # ── 2. Generación sintética con truncation diverso ─────────────────────
    G = load_generator(device)

    gen_seeds = list(range(SEED_GEN_START, SEED_GEN_START + N_FAKE_GEN))
    gen_psis = rng.uniform(PSI_MIN, PSI_MAX, size=N_FAKE_GEN)

    print(f"\n🎨 Generando {N_FAKE_GEN:,} fakes con psi ~ U({PSI_MIN}, {PSI_MAX})")
    t0 = time.time()
    gen_arrays = generate_batch(G, gen_seeds, gen_psis, device, "   train-pool")
    t_gen_main = time.time() - t0

    hard_seeds = list(range(SEED_HARD_START, SEED_HARD_START + N_HARD_GEN))
    hard_psis = np.full(N_HARD_GEN, 1.0)

    print(f"\n🎯 Generando {N_HARD_GEN:,} fakes del TEST DIFÍCIL (psi=1.0, semillas nuevas)")
    t0 = time.time()
    hard_arrays = generate_batch(G, hard_seeds, hard_psis, device, "   test-hard")
    t_gen_hard = time.time() - t0

    del G
    torch.cuda.empty_cache()

    # ── 3. Preprocesamiento de imágenes de Kaggle ──────────────────────────
    print(f"\n⚙️  Preprocesando imágenes de Kaggle ({N_WORKERS} workers)...")
    t0 = time.time()
    real_items = preprocess_kaggle(real_main, [1] * len(real_main), "   reales")
    fake_items = preprocess_kaggle(fake_kaggle, [0] * len(fake_kaggle), "   falsas")
    hard_real_items = preprocess_kaggle(real_hard, [1] * len(real_hard), "   test-hard reales")
    t_prep = time.time() - t0

    # ── 4. Ensamblar conjunto principal y dividir ──────────────────────────
    gen_items = [(a, 0) for a in gen_arrays]
    main_items = real_items + fake_items + gen_items
    rng.shuffle(main_items)

    n_total = len(main_items)
    n_train = int(n_total * SPLIT_RATIOS["train"])
    n_val = int(n_total * SPLIT_RATIOS["val"])

    splits = {
        "train": main_items[:n_train],
        "val":   main_items[n_train:n_train + n_val],
        "test":  main_items[n_train + n_val:],
    }

    hard_items = hard_real_items + [(a, 0) for a in hard_arrays]
    rng.shuffle(hard_items)
    splits["test_hard"] = hard_items

    # ── 5. Guardar ─────────────────────────────────────────────────────────
    print(f"\n💾 Guardando chunks en {OUT_DIR}...")
    stats = [save_split(items, name) for name, items in splits.items()]

    # ── 6. Manifiesto de semillas (reproducibilidad) ───────────────────────
    manifest = {
        "description": "Dataset v2 — corrige el sesgo de truncation detectado en v1",
        "random_seed": SEED,
        "composition": {
            "real_kaggle_main": N_REAL_MAIN,
            "fake_kaggle": N_FAKE_KAGGLE,
            "fake_generated": N_FAKE_GEN,
            "test_hard_real": N_REAL_HARD,
            "test_hard_generated": N_HARD_GEN,
        },
        "generation": {
            "generator": STYLEGAN3_PKL.name,
            "resize_pipeline": "1024 -> 256 (bicubic) -> 224 (bilinear)",
            "train_pool": {
                "seed_start": SEED_GEN_START,
                "seed_end": SEED_GEN_START + N_FAKE_GEN - 1,
                "psi_distribution": f"uniform({PSI_MIN}, {PSI_MAX})",
                "psi_values": [round(float(p), 4) for p in gen_psis],
            },
            "test_hard": {
                "seed_start": SEED_HARD_START,
                "seed_end": SEED_HARD_START + N_HARD_GEN - 1,
                "psi": 1.0,
            },
        },
        "timings_s": {
            "generation_main": round(t_gen_main, 2),
            "generation_hard": round(t_gen_hard, 2),
            "preprocessing_kaggle": round(t_prep, 2),
        },
        "splits": stats,
    }

    manifest_path = META_DIR / "seeds_manifest_v2.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 70)
    print("DATASET v2 COMPLETADO")
    print("=" * 70)
    print(f"  Generación (pool)  : {t_gen_main:.1f}s  ({N_FAKE_GEN / t_gen_main:.1f} img/s)")
    print(f"  Generación (hard)  : {t_gen_hard:.1f}s")
    print(f"  Preproc. Kaggle    : {t_prep:.1f}s")
    print(f"  Manifiesto         : {manifest_path}")
    print(f"\n  El test_hard es la prueba de fuego: 1,000 reales nunca vistos +")
    print(f"  1,000 sintéticos a psi=1.0 con semillas disjuntas del entrenamiento.")


if __name__ == "__main__":
    main()
