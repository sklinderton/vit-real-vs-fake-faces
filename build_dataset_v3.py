"""
build_dataset_v3.py
--------------------
Tercera versión del conjunto de entrenamiento. Corrige el desajuste que
todavía quedaba entre la distribución de entrenamiento y la de evaluación.

DIAGNÓSTICO
-----------
El conjunto v2 generaba sus rostros sintéticos con psi ~ U(0.4, 1.0). El
conjunto de evaluación difícil, en cambio, es íntegramente psi = 1.0. Con
esa mezcla uniforme, apenas una sexta parte del entrenamiento caía en el
régimen que realmente cuesta, y el modelo llegaba a la prueba difícil
habiendo visto pocos ejemplos parecidos.

Resultado de v2/v3 de entrenamiento sobre el test difícil:
    aciertos 89.8 %   ·   AUC 0.9534   ·   detecta sintéticas 87.0 %

CORRECCIÓN
----------
Se estratifica psi para concentrar la mitad de las muestras sintéticas en
[0.85, 1.0], manteniendo cobertura del resto del rango para no perder la
capacidad de detectar rostros suaves:

    25 %  psi ~ U(0.40, 0.70)   régimen fácil, mantiene cobertura
    25 %  psi ~ U(0.70, 0.85)   régimen intermedio
    50 %  psi ~ U(0.85, 1.00)   régimen difícil, el que mide la evaluación

Además se amplía el total de sintéticas generadas de 4 500 a 9 000 y se
reduce la proporción de sintéticas del conjunto público de Kaggle (que
están sesgadas hacia truncation bajo) de 4 500 a 3 000.

COMPOSICIÓN RESULTANTE
----------------------
  Principal (21 000, split 80/10/10):
      9 000 auténticas (Kaggle)
      3 000 sintéticas del conjunto público
      9 000 sintéticas generadas con psi estratificado

  val_hard  (1 000): 500 auténticas + 500 sintéticas psi = 1.0
  test_hard (1 000): 500 auténticas + 500 sintéticas psi = 1.0
      Semillas disjuntas entre sí y del entrenamiento.

Uso:
    python build_dataset_v3.py
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
OUT_DIR = PROJECT_ROOT / "data" / "processed_v3"
META_DIR = PROJECT_ROOT / "data" / "metadata"

sys.path.insert(0, str(STYLEGAN3_REPO))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from preprocessing_worker import preprocess_image  # noqa: E402

IMG_SIZE = 224
N_WORKERS = 16
CHUNK_SIZE = 2000
SEED = 42

N_REAL_MAIN = 9000
N_FAKE_KAGGLE = 3000
N_FAKE_GEN = 9000

N_REAL_VALHARD = 500
N_REAL_TESTHARD = 500
N_GEN_VALHARD = 500
N_GEN_TESTHARD = 500

# Estratificación de psi: mitad del peso en el régimen difícil
PSI_STRATA = [
    (0.25, 0.40, 0.70),
    (0.25, 0.70, 0.85),
    (0.50, 0.85, 1.00),
]

SEED_GEN_START = 1_000_000       # pool de entrenamiento
SEED_VALHARD_START = 1_500_000   # validación difícil
SEED_TESTHARD_START = 1_600_000  # prueba difícil

SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}


def sample_psi(n, rng):
    """Muestrea psi según los estratos definidos."""
    psis = []
    for weight, lo, hi in PSI_STRATA:
        k = int(round(n * weight))
        psis.extend(rng.uniform(lo, hi, size=k).tolist())
    # Ajustar por redondeo
    while len(psis) < n:
        psis.append(float(rng.uniform(0.85, 1.0)))
    psis = psis[:n]
    rng.shuffle(psis)
    return np.array(psis)


def load_generator(device):
    print("📦 Cargando generador StyleGAN3...")
    with open(STYLEGAN3_PKL, "rb") as f:
        G = pickle.load(f)["G_ema"].to(device)
    G.eval()
    return G


def generate(G, seeds, psis, device, desc):
    """Genera rostros como arrays uint8 (C,H,W) de 224×224."""
    out = []
    for seed, psi in tqdm(list(zip(seeds, psis)), desc=desc, unit="img"):
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        c = torch.zeros([1, G.c_dim], device=device)
        with torch.no_grad():
            img = G(z, c, truncation_psi=float(psi), noise_mode="const")
        arr = ((img.clamp(-1, 1) + 1) * (255 / 2))
        arr = arr.permute(0, 2, 3, 1).to(torch.uint8)[0].cpu().numpy()
        pil = Image.fromarray(arr, "RGB")
        pil = pil.resize((256, 256), Image.BICUBIC)      # igual que Kaggle
        pil = pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        out.append(np.ascontiguousarray(
            np.asarray(pil, dtype=np.uint8).transpose(2, 0, 1)))
    return out


def preprocess_kaggle(paths, label, desc):
    args = [(str(p), label, False) for p in paths]
    res = [None] * len(args)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(preprocess_image, a): i for i, a in enumerate(args)}
        for f in tqdm(as_completed(futs), total=len(futs), desc=desc, unit="img"):
            i = futs[f]
            arr, lbl = f.result()
            if arr is not None:
                res[i] = (arr, lbl)
    return [r for r in res if r is not None]


def save_split(items, name):
    d = OUT_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.npy"):
        old.unlink()

    n = len(items)
    cid = 0
    for s in range(0, n, CHUNK_SIZE):
        chunk = items[s:s + CHUNK_SIZE]
        np.save(d / f"images_chunk{cid:03d}.npy",
                np.stack([c[0] for c in chunk]).astype(np.uint8))
        np.save(d / f"labels_chunk{cid:03d}.npy",
                np.array([c[1] for c in chunk], dtype=np.int8))
        cid += 1

    n_real = sum(1 for _, l in items if l == 1)
    print(f"   {name:16s}: {n:>6,} imgs (real={n_real:,}, fake={n - n_real:,}) → {cid} chunks")
    return {"split": name, "n": n, "n_real": n_real, "n_fake": n - n_real, "n_chunks": cid}


def main():
    rng = np.random.default_rng(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("DATASET v3 — PSI ESTRATIFICADO HACIA EL RÉGIMEN DIFÍCIL")
    print("=" * 72)
    print(f"  Device: {device}")
    for w, lo, hi in PSI_STRATA:
        print(f"    {w:>5.0%} de las sintéticas con psi ∈ [{lo}, {hi}]")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    # ── Inventario ─────────────────────────────────────────────────────────
    print("\n📂 Escaneando el conjunto público...")
    real_files = sorted((RAW_DIR / "Real faces").glob("*.png")) + \
                 sorted((RAW_DIR / "Real faces").glob("*.jpg"))
    fake_files = sorted((RAW_DIR / "Fake faces").glob("*.png")) + \
                 sorted((RAW_DIR / "Fake faces").glob("*.jpg"))
    print(f"   auténticas: {len(real_files):,}   sintéticas: {len(fake_files):,}")

    need_real = N_REAL_MAIN + N_REAL_VALHARD + N_REAL_TESTHARD
    assert len(real_files) >= need_real, f"Se necesitan {need_real} auténticas"
    assert len(fake_files) >= N_FAKE_KAGGLE

    ridx = rng.permutation(len(real_files))
    real_main = [real_files[i] for i in ridx[:N_REAL_MAIN]]
    real_vh = [real_files[i] for i in ridx[N_REAL_MAIN:N_REAL_MAIN + N_REAL_VALHARD]]
    real_th = [real_files[i] for i in ridx[N_REAL_MAIN + N_REAL_VALHARD:need_real]]

    fidx = rng.permutation(len(fake_files))
    fake_kaggle = [fake_files[i] for i in fidx[:N_FAKE_KAGGLE]]

    # ── Generación ─────────────────────────────────────────────────────────
    G = load_generator(device)

    gen_seeds = list(range(SEED_GEN_START, SEED_GEN_START + N_FAKE_GEN))
    gen_psis = sample_psi(N_FAKE_GEN, rng)
    print(f"\n🎨 Generando {N_FAKE_GEN:,} sintéticas con psi estratificado")
    print(f"   psi medio: {gen_psis.mean():.3f}   ·   "
          f"fracción con psi ≥ 0.85: {(gen_psis >= 0.85).mean():.1%}")
    t0 = time.time()
    gen_arrays = generate(G, gen_seeds, gen_psis, device, "   entrenamiento")
    t_gen = time.time() - t0

    vh_seeds = list(range(SEED_VALHARD_START, SEED_VALHARD_START + N_GEN_VALHARD))
    th_seeds = list(range(SEED_TESTHARD_START, SEED_TESTHARD_START + N_GEN_TESTHARD))
    print(f"\n🎯 Generando conjuntos difíciles (psi = 1.0, semillas disjuntas)")
    vh_arrays = generate(G, vh_seeds, np.ones(N_GEN_VALHARD), device, "   val_hard")
    th_arrays = generate(G, th_seeds, np.ones(N_GEN_TESTHARD), device, "   test_hard")

    del G
    torch.cuda.empty_cache()

    # ── Preprocesamiento del conjunto público ─────────────────────────────
    print(f"\n⚙️  Preprocesando imágenes de disco ({N_WORKERS} procesos)...")
    t0 = time.time()
    real_items = preprocess_kaggle(real_main, 1, "   auténticas")
    fake_items = preprocess_kaggle(fake_kaggle, 0, "   sintéticas públicas")
    vh_real = preprocess_kaggle(real_vh, 1, "   val_hard auténticas")
    th_real = preprocess_kaggle(real_th, 1, "   test_hard auténticas")
    t_prep = time.time() - t0

    # ── Ensamblar ──────────────────────────────────────────────────────────
    main_items = real_items + fake_items + [(a, 0) for a in gen_arrays]
    rng.shuffle(main_items)

    n = len(main_items)
    n_tr = int(n * SPLIT_RATIOS["train"])
    n_va = int(n * SPLIT_RATIOS["val"])

    splits = {
        "train": main_items[:n_tr],
        "val": main_items[n_tr:n_tr + n_va],
        "test": main_items[n_tr + n_va:],
    }

    vh = vh_real + [(a, 0) for a in vh_arrays]
    th = th_real + [(a, 0) for a in th_arrays]
    rng.shuffle(vh)
    rng.shuffle(th)
    splits["val_hard"] = vh
    splits["test_hard"] = th

    print(f"\n💾 Guardando en {OUT_DIR}...")
    stats = [save_split(v, k) for k, v in splits.items()]

    manifest = {
        "description": "Dataset v3 — psi estratificado hacia el régimen difícil",
        "random_seed": SEED,
        "psi_strata": [{"weight": w, "low": lo, "high": hi} for w, lo, hi in PSI_STRATA],
        "psi_mean": float(gen_psis.mean()),
        "psi_frac_hard": float((gen_psis >= 0.85).mean()),
        "composition": {
            "real_kaggle": N_REAL_MAIN,
            "fake_kaggle": N_FAKE_KAGGLE,
            "fake_generated": N_FAKE_GEN,
        },
        "seeds": {
            "train_pool": [SEED_GEN_START, SEED_GEN_START + N_FAKE_GEN - 1],
            "val_hard": [SEED_VALHARD_START, SEED_VALHARD_START + N_GEN_VALHARD - 1],
            "test_hard": [SEED_TESTHARD_START, SEED_TESTHARD_START + N_GEN_TESTHARD - 1],
        },
        "timings_s": {"generation": round(t_gen, 1), "preprocessing": round(t_prep, 1)},
        "splits": stats,
    }
    with open(META_DIR / "seeds_manifest_v3.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 72)
    print("DATASET v3 LISTO")
    print("=" * 72)
    print(f"  Generación    : {t_gen:.1f}s  ({N_FAKE_GEN / t_gen:.1f} img/s)")
    print(f"  Preprocesado  : {t_prep:.1f}s")
    print(f"  Manifiesto    : {META_DIR / 'seeds_manifest_v3.json'}")


if __name__ == "__main__":
    main()
