"""
split_test_hard.py
-------------------
Divide el conjunto test_hard (2,000 imgs) en dos mitades estratificadas:

    val_hard  (1,000)  → guía el early stopping durante el entrenamiento v3
    test_hard (1,000)  → evaluación final, nunca vista durante el ajuste

MOTIVACIÓN
----------
En el entrenamiento v2 el early stopping se guiaba por el conjunto `val`,
cuya distribución de truncation es la misma que la del entrenamiento
(psi ~ U(0.4, 1.0)). Eso produce una señal diluida: el modelo puede parecer
estable en validación acertando los casos fáciles (psi bajo) mientras falla
en los difíciles (psi=1.0). El resultado fue que el mejor checkpoint quedó
en la época 1 y el entrenamiento se detuvo prematuramente.

Al usar val_hard (psi=1.0, el escenario adverso) como criterio de parada,
el modelo se optimiza explícitamente para generalizar al caso difícil.

No hay fuga de datos: ninguna de estas imágenes participó del entrenamiento,
y val_hard/test_hard quedan disjuntos entre sí.

Uso:
    python split_test_hard.py
"""

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/data/ulead-04/proyecto_paralela")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_v2"
META_DIR = PROJECT_ROOT / "data" / "metadata"

SRC_SPLIT = PROCESSED_DIR / "test_hard"
VAL_HARD = PROCESSED_DIR / "val_hard"
TEST_HARD = PROCESSED_DIR / "test_hard_final"

SEED = 123


def main():
    print("=" * 66)
    print("DIVISIÓN DE test_hard → val_hard + test_hard_final")
    print("=" * 66)

    img_files = sorted(SRC_SPLIT.glob("images_chunk*.npy"))
    lbl_files = sorted(SRC_SPLIT.glob("labels_chunk*.npy"))
    if not img_files:
        raise FileNotFoundError(f"No hay chunks en {SRC_SPLIT}")

    images = np.concatenate([np.load(f) for f in img_files], axis=0)
    labels = np.concatenate([np.load(f) for f in lbl_files], axis=0)
    print(f"\n  Cargado: {images.shape[0]:,} imágenes  shape={images.shape[1:]}")
    print(f"  Real={int((labels == 1).sum()):,}  Fake={int((labels == 0).sum()):,}")

    rng = np.random.default_rng(SEED)

    # ── División estratificada 50/50 por clase ─────────────────────────────
    val_idx, test_idx = [], []
    for cls in (0, 1):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        half = len(idx) // 2
        val_idx.extend(idx[:half].tolist())
        test_idx.extend(idx[half:].tolist())

    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    for name, out_dir, idx in (
        ("val_hard", VAL_HARD, val_idx),
        ("test_hard_final", TEST_HARD, test_idx),
    ):
        out_dir.mkdir(parents=True, exist_ok=True)
        # Limpiar chunks previos si se re-ejecuta
        for old in out_dir.glob("*.npy"):
            old.unlink()

        np.save(out_dir / "images_chunk000.npy", images[idx])
        np.save(out_dir / "labels_chunk000.npy", labels[idx])

        n_real = int((labels[idx] == 1).sum())
        n_fake = int((labels[idx] == 0).sum())
        print(f"  {name:16s}: {len(idx):>5,} imgs (real={n_real:,}, fake={n_fake:,})")

    manifest = {
        "source": str(SRC_SPLIT),
        "seed": SEED,
        "val_hard": {"n": int(len(val_idx)),
                      "n_real": int((labels[val_idx] == 1).sum()),
                      "n_fake": int((labels[val_idx] == 0).sum())},
        "test_hard_final": {"n": int(len(test_idx)),
                             "n_real": int((labels[test_idx] == 1).sum()),
                             "n_fake": int((labels[test_idx] == 0).sum())},
        "note": ("val_hard guía el early stopping del entrenamiento v3; "
                  "test_hard_final permanece intacto para la evaluación final."),
    }
    META_DIR.mkdir(parents=True, exist_ok=True)
    with open(META_DIR / "hard_split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  ✅ Manifiesto: {META_DIR / 'hard_split_manifest.json'}")
    print("\n  Nota: el directorio original 'test_hard' se conserva intacto;")
    print("  el entrenamiento v3 usará val_hard y test_hard_final.")


if __name__ == "__main__":
    main()
