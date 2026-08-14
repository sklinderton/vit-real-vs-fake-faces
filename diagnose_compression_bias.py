"""
diagnose_compression_bias.py
-----------------------------
Diagnostica si existe un sesgo de formato/compresión entre las clases
"real" y "fake" del dataset, que el clasificador pueda estar explotando
como atajo en lugar de aprender artefactos genuinos de la GAN.

Motivación: el ViT entrenado alcanza 82% de recall sobre fakes del dataset
de Kaggle, pero solo 21.5% sobre rostros StyleGAN3 generados en vivo. Esa
brecha sugiere que el modelo aprendió una característica del procesamiento
del dataset, no del generador.

Qué revisa:
  1. Formato de archivo real (vía PIL, no solo la extensión)
  2. Dimensiones y modo de color
  3. Distribución de tamaño de archivo (bytes por píxel = proxy de compresión)
  4. Tablas de cuantización JPEG — la "huella digital" del compresor usado
  5. Estadísticas de alta frecuencia (energía en el espectro FFT)

Uso:
    python diagnose_compression_bias.py --sample 300
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path("/data/ulead-04/proyecto_paralela")
RAW_DIR = PROJECT_ROOT / "data" / "raw"
GENERATED_DIR = PROJECT_ROOT / "data" / "stylegan3_generated"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=300,
                    help="Imágenes a muestrear por clase")
    return p.parse_args()


def collect_files(directory, limit):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
    files = []
    for ext in exts:
        files.extend(sorted(directory.glob(ext)))
    return files[:limit]


def analyze_image(path):
    """Extrae metadatos de formato y compresión de una sola imagen."""
    info = {"filename": path.name, "file_bytes": path.stat().st_size}

    with Image.open(path) as im:
        info["format"] = im.format
        info["mode"] = im.mode
        info["width"], info["height"] = im.size
        info["bytes_per_pixel"] = info["file_bytes"] / (im.width * im.height)

        # Tabla de cuantización JPEG: huella del compresor.
        # Dos imágenes comprimidas por el mismo software/calidad comparten tabla.
        qtable_sig = None
        if im.format == "JPEG" and hasattr(im, "quantization"):
            qt = im.quantization
            if qt:
                # Firma compacta: suma de la primera tabla (luma)
                first_table = qt.get(0)
                if first_table is not None:
                    qtable_sig = int(sum(first_table))
        info["jpeg_qtable_sum"] = qtable_sig

        # Energía en altas frecuencias (proxy de nitidez / artefactos)
        gray = np.asarray(im.convert("L").resize((256, 256)), dtype=np.float32)
        f = np.fft.fftshift(np.fft.fft2(gray))
        magnitude = np.abs(f)
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        y, x = np.indices((h, w))
        r = np.hypot(x - cx, y - cy)
        # Anillo de alta frecuencia: r > 40% del radio máximo
        high_freq_mask = r > (r.max() * 0.4)
        info["high_freq_energy"] = float(np.log1p(magnitude[high_freq_mask]).mean())

    return info


def summarize(records, label):
    if not records:
        return {"class": label, "n": 0}

    formats = Counter(r["format"] for r in records)
    modes = Counter(r["mode"] for r in records)
    sizes = Counter(f"{r['width']}x{r['height']}" for r in records)
    qtables = Counter(r["jpeg_qtable_sum"] for r in records if r["jpeg_qtable_sum"] is not None)

    bpp = np.array([r["bytes_per_pixel"] for r in records])
    hf = np.array([r["high_freq_energy"] for r in records])

    summary = {
        "class": label,
        "n": len(records),
        "formats": dict(formats),
        "modes": dict(modes),
        "dimensions": dict(sizes.most_common(3)),
        "bytes_per_pixel": {
            "mean": round(float(bpp.mean()), 4),
            "std": round(float(bpp.std()), 4),
            "min": round(float(bpp.min()), 4),
            "max": round(float(bpp.max()), 4),
        },
        "high_freq_energy": {
            "mean": round(float(hf.mean()), 4),
            "std": round(float(hf.std()), 4),
        },
        "n_distinct_jpeg_qtables": len(qtables),
        "top_jpeg_qtables": dict(qtables.most_common(5)),
    }
    return summary


def print_summary(s):
    print(f"\n{'─' * 66}")
    print(f"  CLASE: {s['class']}  (n={s['n']})")
    print(f"{'─' * 66}")
    if s["n"] == 0:
        print("  (sin imágenes encontradas)")
        return
    print(f"  Formatos          : {s['formats']}")
    print(f"  Modos de color    : {s['modes']}")
    print(f"  Dimensiones       : {s['dimensions']}")
    print(f"  Bytes por píxel   : media={s['bytes_per_pixel']['mean']:.4f}  "
          f"std={s['bytes_per_pixel']['std']:.4f}  "
          f"[{s['bytes_per_pixel']['min']:.4f} – {s['bytes_per_pixel']['max']:.4f}]")
    print(f"  Energía alta frec.: media={s['high_freq_energy']['mean']:.4f}  "
          f"std={s['high_freq_energy']['std']:.4f}")
    print(f"  Tablas JPEG únicas: {s['n_distinct_jpeg_qtables']}")
    if s["top_jpeg_qtables"]:
        print(f"  Tablas más comunes: {s['top_jpeg_qtables']}")


def main():
    args = parse_args()

    print("=" * 66)
    print("DIAGNÓSTICO DE SESGO DE FORMATO / COMPRESIÓN")
    print("=" * 66)

    sources = {
        "Real (Kaggle)": RAW_DIR / "Real faces",
        "Fake (Kaggle)": RAW_DIR / "Fake faces",
        "Fake (StyleGAN3 generado)": GENERATED_DIR,
    }

    all_summaries = []

    for label, directory in sources.items():
        if not directory.exists():
            print(f"\n⚠️  No existe: {directory}")
            continue
        files = collect_files(directory, args.sample)
        print(f"\nAnalizando {len(files)} imágenes de '{label}'...")
        records = [analyze_image(f) for f in files]
        s = summarize(records, label)
        all_summaries.append(s)
        print_summary(s)

    # ── Veredicto ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 66}")
    print("  INTERPRETACIÓN")
    print(f"{'=' * 66}")

    kaggle = {s["class"]: s for s in all_summaries if "Kaggle" in s["class"]}
    if len(kaggle) == 2:
        r = kaggle["Real (Kaggle)"]
        f = kaggle["Fake (Kaggle)"]

        fmt_differ = set(r["formats"]) != set(f["formats"])
        bpp_diff = abs(r["bytes_per_pixel"]["mean"] - f["bytes_per_pixel"]["mean"])
        bpp_pooled_std = (r["bytes_per_pixel"]["std"] + f["bytes_per_pixel"]["std"]) / 2
        bpp_effect = bpp_diff / bpp_pooled_std if bpp_pooled_std > 0 else 0
        qt_differ = set(r["top_jpeg_qtables"]) != set(f["top_jpeg_qtables"])

        print(f"\n  ¿Formatos distintos entre real y fake?  {'SÍ ⚠️' if fmt_differ else 'No'}")
        print(f"  ¿Tablas JPEG distintas?                 {'SÍ ⚠️' if qt_differ else 'No'}")
        print(f"  Diferencia en bytes/píxel (d de Cohen): {bpp_effect:.3f}", end="")
        if bpp_effect > 0.8:
            print("  ← EFECTO GRANDE ⚠️")
        elif bpp_effect > 0.5:
            print("  ← efecto moderado")
        else:
            print("  ← efecto pequeño")

        if fmt_differ or qt_differ or bpp_effect > 0.5:
            print("\n  ⚠️  SE DETECTA SESGO DE PROCESAMIENTO entre las clases del dataset.")
            print("      El clasificador puede estar usando esta diferencia como atajo,")
            print("      en lugar de aprender artefactos reales del generador.")
            print("      → Se requiere ecualización de compresión en el pipeline v2.")
        else:
            print("\n  ✓  No se detecta un sesgo obvio de formato/compresión.")
            print("     La brecha de generalización puede deberse a otros factores")
            print("     (p. ej. distribución de semillas o post-procesado del dataset).")

    out_path = RESULTS_DIR / "compression_bias_diagnosis.json"
    with open(out_path, "w") as fh:
        json.dump(all_summaries, fh, indent=2)
    print(f"\n  ✅ Guardado: {out_path}")


if __name__ == "__main__":
    main()
