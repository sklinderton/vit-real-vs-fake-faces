"""
run_preprocessing.py
---------------------
Regenera el dataset preprocesado (train/val/test) desde las imágenes raw,
usando ProcessPoolExecutor para paralelizar la decodificación/normalización.

Uso:
    python run_preprocessing.py

Requiere: preprocessing_worker.py en el mismo directorio (o en src/, ajustado
vía sys.path más abajo).
"""

import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import polars as pl
from tqdm import tqdm

# ── Rutas del proyecto ─────────────────────────────────────────────────────
PROJECT_ROOT = Path("/data/ulead-04/proyecto_paralela")
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))
from preprocessing_worker import preprocess_image  # noqa: E402

# ── Parámetros ──────────────────────────────────────────────────────────────
N_WORKERS = 16          # coincide con --cpus-per-task del srun
CHUNK_SIZE = 2000        # imágenes por archivo .npy
SEED = 42
SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}


def build_file_list():
    """Escanea Real faces / Fake faces y construye la lista (path, label)."""
    real_dir = RAW_DIR / "Real faces"
    fake_dir = RAW_DIR / "Fake faces"

    real_files = sorted(real_dir.glob("*.jpg")) + sorted(real_dir.glob("*.png")) + sorted(real_dir.glob("*.jpeg"))
    fake_files = sorted(fake_dir.glob("*.jpg")) + sorted(fake_dir.glob("*.png")) + sorted(fake_dir.glob("*.jpeg"))

    print(f"  Reales encontradas: {len(real_files):,}")
    print(f"  Falsas encontradas: {len(fake_files):,}")

    records = [(str(p), 1) for p in real_files] + [(str(p), 0) for p in fake_files]
    return records


def split_dataset(records):
    """Split estratificado 80/10/10 usando Polars, reproducible con SEED."""
    df = pl.DataFrame(records, schema=["filepath", "label"], orient="row")

    # Shuffle reproducible por clase (estratificado)
    splits = {"train": [], "val": [], "test": []}
    for label in [0, 1]:
        sub = df.filter(pl.col("label") == label).sample(fraction=1.0, seed=SEED)
        n = sub.height
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])

        splits["train"].append(sub.slice(0, n_train))
        splits["val"].append(sub.slice(n_train, n_val))
        splits["test"].append(sub.slice(n_train + n_val, n - n_train - n_val))

    result = {}
    for split_name, parts in splits.items():
        merged = pl.concat(parts).sample(fraction=1.0, seed=SEED)  # shuffle final
        result[split_name] = list(zip(merged["filepath"].to_list(), merged["label"].to_list()))

    return result


def preprocess_split_parallel(records, split_name, is_train):
    """Procesa un split completo en paralelo y guarda chunks .npy."""
    out_dir = PROCESSED_DIR / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    args_list = [(fp, lbl, is_train) for fp, lbl in records]
    n_total = len(args_list)

    print(f"\n🔄 Procesando split '{split_name}' | {n_total:,} imgs | {N_WORKERS} workers")
    t0 = time.time()

    results = [None] * n_total
    n_errors = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(preprocess_image, a): i for i, a in enumerate(args_list)}
        pbar = tqdm(total=n_total, desc=f"  {split_name:5s}", unit="img")

        for future in as_completed(futures):
            idx = futures[future]
            arr, lbl = future.result()
            pbar.update(1)

            if arr is None:
                n_errors += 1
                continue
            results[idx] = (arr, lbl)

        pbar.close()

    elapsed = time.time() - t0

    # Filtrar errores y guardar en chunks
    valid = [r for r in results if r is not None]
    n_saved = len(valid)

    chunk_id = 0
    for start in range(0, n_saved, CHUNK_SIZE):
        chunk = valid[start:start + CHUNK_SIZE]
        images = np.stack([c[0] for c in chunk]).astype(np.uint8)   # crudo 0-255, 4x más liviano que float32
        labels = np.array([c[1] for c in chunk], dtype=np.int8)

        np.save(out_dir / f"images_chunk{chunk_id:03d}.npy", images)
        np.save(out_dir / f"labels_chunk{chunk_id:03d}.npy", labels)
        chunk_id += 1

    throughput = n_total / elapsed
    print(f"  ✅ {split_name}: {n_saved:,}/{n_total:,} guardadas ({n_errors} errores)")
    print(f"     Tiempo: {elapsed:.1f}s | Throughput: {throughput:.1f} img/s | Chunks: {chunk_id}")

    return {"split": split_name, "n_total": n_total, "n_saved": n_saved,
            "n_errors": n_errors, "time_s": round(elapsed, 2),
            "throughput": round(throughput, 2), "n_chunks": chunk_id}


def main():
    print("=" * 70)
    print("PREPROCESAMIENTO PARALELO — Real vs Fake Faces (StyleGAN3)")
    print("=" * 70)

    print("\n📂 Escaneando imágenes raw...")
    records = build_file_list()
    print(f"  Total: {len(records):,} imágenes")

    print("\n✂️  Generando split estratificado 80/10/10...")
    splits = split_dataset(records)
    for name, recs in splits.items():
        n_real = sum(1 for _, lbl in recs if lbl == 1)
        n_fake = sum(1 for _, lbl in recs if lbl == 0)
        print(f"  {name:5s}: {len(recs):,} imgs (real={n_real:,}, fake={n_fake:,})")

    all_stats = []
    all_stats.append(preprocess_split_parallel(splits["train"], "train", is_train=True))
    all_stats.append(preprocess_split_parallel(splits["val"], "val", is_train=False))
    all_stats.append(preprocess_split_parallel(splits["test"], "test", is_train=False))

    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    df_stats = pl.DataFrame(all_stats)
    print(df_stats)

    metadata_dir = PROJECT_ROOT / "data" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    df_stats.write_csv(metadata_dir / "preprocessing_stats_kabre_gpu.csv")
    print(f"\n✅ Estadísticas guardadas en {metadata_dir / 'preprocessing_stats_kabre_gpu.csv'}")


if __name__ == "__main__":
    main()
