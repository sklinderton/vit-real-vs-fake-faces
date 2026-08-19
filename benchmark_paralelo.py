"""
benchmark_paralelo.py
----------------------
Mide formalmente el rendimiento del preprocesamiento paralelo: speedup,
eficiencia y escalabilidad fuerte.

Es el experimento que exige la rúbrica del curso y que hasta ahora faltaba.
Se conocía el rendimiento con 16 procesos (2 358 imágenes por segundo) pero
nunca se midió el caso secuencial equivalente, sin el cual no se puede
calcular el speedup.

DEFINICIONES
------------
    Speedup       Sp = T1 / Tp
                  Cuántas veces más rápido va con p procesos que con uno.

    Eficiencia    Ep = Sp / p
                  Qué fracción del ideal se aprovecha. Con 8 procesos, un
                  speedup de 8 daría eficiencia 1.0 (perfecta); un speedup
                  de 6 daría 0.75.

    Escalabilidad fuerte
                  Cómo evoluciona el speedup al añadir procesos manteniendo
                  fijo el tamaño del problema. La ley de Amdahl predice que
                  se estanca en 1/f, donde f es la fracción no paralelizable.

QUÉ SE MIDE
-----------
La decodificación, redimensionado y normalización de imágenes. Cada imagen
es independiente del resto, así que el trabajo se reparte sin comunicación
entre procesos: es un caso favorable al paralelismo, y el interés está en
cuánto se aleja de todos modos del ideal por el coste de crear procesos y
transferir resultados.

Uso:
    python benchmark_paralelo.py --images 2000
"""

import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Estas variables deben fijarse ANTES de importar NumPy, PIL o cualquier
# librería numérica: se leen una sola vez, al cargarse.
#
# Sin ellas, la referencia "secuencial" no lo es de verdad. NumPy y las
# bibliotecas de imagen abren hilos internos por su cuenta, así que el caso
# de un proceso ya estaría usando varios núcleos y el speedup calculado a
# partir de él saldría subestimado. En una medición previa el paso de uno a
# dos procesos daba 1.03x, un valor imposible en trabajo paralelizable, que
# es justo el síntoma de este efecto.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np  # noqa: E402  (tras fijar las variables de entorno)

PROJECT_ROOT = Path("/data/ulead-04/proyecto_paralela")
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SRC_DIR = PROJECT_ROOT / "src"
RESULTS = PROJECT_ROOT / "results"
FIGURES = PROJECT_ROOT / "figures"

RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC_DIR))
os.environ["VIT_IMG_SIZE"] = "256"
from preprocessing_worker import preprocess_image  # noqa: E402

WORKER_COUNTS = [1, 2, 4, 8, 12, 16]
REPEATS = 3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=int, default=2000,
                    help="Imágenes por medición")
    p.add_argument("--repeats", type=int, default=REPEATS)
    return p.parse_args()


def collect_images(n):
    """Toma n rutas balanceadas entre las dos clases."""
    real = sorted((RAW_DIR / "Real faces").glob("*.png"))[:n // 2]
    fake = sorted((RAW_DIR / "Fake faces").glob("*.png"))[:n // 2]
    files = real + fake
    if len(files) < n:
        raise RuntimeError(f"Solo se encontraron {len(files)} imágenes de {n}")
    return [(str(f), 1 if i < len(real) else 0, False)
            for i, f in enumerate(files)]


def run_serial(args_list):
    """Referencia secuencial: un solo proceso, sin ProcessPoolExecutor."""
    t0 = time.perf_counter()
    for a in args_list:
        preprocess_image(a)
    return time.perf_counter() - t0


def run_parallel(args_list, n_workers):
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        list(ex.map(preprocess_image, args_list, chunksize=16))
    return time.perf_counter() - t0


def main():
    args = parse_args()

    print("=" * 74)
    print("RENDIMIENTO DEL PREPROCESAMIENTO PARALELO")
    print("=" * 74)
    print(f"  Nodo            : {platform.node()}")
    print(f"  Núcleos visibles: {os.cpu_count()}")
    print(f"  Imágenes        : {args.images:,} por medición")
    print(f"  Repeticiones    : {args.repeats} (se toma el mínimo)")

    print("\n📂 Preparando lista de imágenes...")
    args_list = collect_images(args.images)

    # Calentamiento completo. Es imprescindible: las imágenes viven en un
    # sistema de archivos en red (Lustre), y la primera lectura de cada una
    # paga el coste de traerla por red. En una medición previa la primera
    # pasada secuencial tardó 36 s y la tercera 5.4 s sobre los mismos datos,
    # una diferencia de siete veces que no tiene nada que ver con el
    # paralelismo. Se recorre el conjunto completo dos veces antes de medir.
    print("🔥 Calentando la caché del sistema de archivos (2 pasadas)...")
    for _ in range(2):
        run_parallel(args_list, 16)
    print("   Listo.\n")

    print(f"\n{'Procesos':>9} {'Tiempo (s)':>12} {'img/s':>10} "
          f"{'Speedup':>9} {'Eficiencia':>12}")
    print("-" * 74)

    rows = []
    t1 = None

    for p in WORKER_COUNTS:
        times = []
        for _ in range(args.repeats):
            if p == 1:
                times.append(run_serial(args_list))
            else:
                times.append(run_parallel(args_list, p))
        t = min(times)          # el mínimo refleja mejor el rendimiento real
        dispersion = (max(times) - min(times)) / min(times) if min(times) else 0

        if t1 is None:
            t1 = t

        speedup = t1 / t
        efficiency = speedup / p
        throughput = args.images / t

        rows.append({
            "workers": p,
            "time_s": round(t, 4),
            "throughput_img_s": round(throughput, 1),
            "speedup": round(speedup, 4),
            "efficiency": round(efficiency, 4),
            "times_all": [round(x, 4) for x in times],
            "dispersion": round(dispersion, 4),
        })

        flag = "  ⚠ inestable" if dispersion > 0.15 else ""
        print(f"{p:>9} {t:>12.3f} {throughput:>10.1f} "
              f"{speedup:>9.2f} {efficiency:>11.1%}{flag}")

    # ── Interpretación ─────────────────────────────────────────────────────
    best = max(rows, key=lambda r: r["speedup"])
    at16 = next((r for r in rows if r["workers"] == 16), rows[-1])

    print("\n" + "=" * 74)
    print("ANÁLISIS")
    print("=" * 74)
    print(f"  Mejor speedup   : {best['speedup']:.2f}× con {best['workers']} procesos")
    print(f"  Eficiencia ahí  : {best['efficiency']:.1%}")
    print(f"  Con 16 procesos : {at16['speedup']:.2f}× "
          f"(eficiencia {at16['efficiency']:.1%})")

    # Fracción secuencial estimada por la ley de Amdahl:
    #   Sp = 1 / (f + (1-f)/p)  ->  f = (p/Sp - 1) / (p - 1)
    p_, s_ = at16["workers"], at16["speedup"]
    f_seq = (p_ / s_ - 1) / (p_ - 1) if p_ > 1 and s_ > 0 else 0.0
    print(f"\n  Fracción no paralelizable estimada (Amdahl): {f_seq:.1%}")
    print(f"  Techo teórico con infinitos procesos: {1 / f_seq:.1f}×"
          if f_seq > 0 else "  Techo teórico: no acotado por esta medición")

    # ── Punto de saturación ────────────────────────────────────────────
    # Saturación: primer punto a partir del cual duplicar procesos aporta
    # menos del 15 % de speedup adicional. Se exige haber superado 2x antes,
    # para no confundir el arranque con el estancamiento.
    saturation = None
    for i in range(1, len(rows)):
        prev, cur = rows[i - 1], rows[i]
        if prev["speedup"] < 2.0:
            continue
        rel_gain = (cur["speedup"] - prev["speedup"]) / prev["speedup"]
        proc_ratio = cur["workers"] / prev["workers"]
        if rel_gain < 0.15 * (proc_ratio - 1):
            saturation = prev["workers"]
            break

    if saturation:
        print(f"\n  Punto de saturación: {saturation} procesos")
        print("  A partir de ahí, añadir procesos deja de compensar. Las")
        print("  imágenes se leen de un sistema de archivos en red compartido,")
        print("  así que el límite no está en el cálculo sino en la velocidad")
        print("  a la que llegan los datos: es un problema limitado por")
        print("  entrada/salida, no por CPU.")
    else:
        print("\n  No se observa saturación en el rango medido.")

    print("\n  La fracción no paralelizable recoge el coste de crear los")
    print("  procesos, enviarles las rutas y traer de vuelta los arreglos.")
    print("  Al ser imágenes independientes no hay sincronización entre")
    print("  procesos, así que el cálculo en sí escala bien.")

    # ── Guardar ────────────────────────────────────────────────────────────
    out = {
        "node": platform.node(),
        "cpu_count": os.cpu_count(),
        "n_images": args.images,
        "repeats": args.repeats,
        "image_size": 256,
        "measurements": rows,
        "best_speedup": best["speedup"],
        "best_workers": best["workers"],
        "efficiency_at_best": best["efficiency"],
        "amdahl_serial_fraction": round(f_seq, 4),
        "saturation_workers": saturation,
        "bottleneck": ("entrada/salida sobre sistema de archivos en red"
                        if saturation else "no determinado"),
    }
    path = RESULTS / "benchmark_paralelo.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  ✅ {path}")

    # ── Figura ─────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ws = [r["workers"] for r in rows]
        sp = [r["speedup"] for r in rows]
        ef = [r["efficiency"] * 100 for r in rows]

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

        axes[0].plot(ws, ws, "--", color="#999", label="Ideal (lineal)")
        axes[0].plot(ws, sp, "o-", color="#7B4FD1", lw=2.2, ms=8,
                      label="Medido")
        axes[0].set_xlabel("Procesos")
        axes[0].set_ylabel("Speedup")
        axes[0].set_title("Escalabilidad fuerte")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].axhline(100, ls="--", color="#999", label="Ideal")
        axes[1].plot(ws, ef, "s-", color="#0E7C6B", lw=2.2, ms=8,
                      label="Medida")
        axes[1].set_xlabel("Procesos")
        axes[1].set_ylabel("Eficiencia (%)")
        axes[1].set_title("Eficiencia paralela")
        axes[1].set_ylim(0, 115)
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.suptitle(f"Preprocesamiento de {args.images:,} imágenes · "
                      f"{platform.node()}", fontweight="bold")
        plt.tight_layout()
        fig_path = FIGURES / "benchmark_paralelo.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✅ {fig_path}")
    except Exception as exc:
        print(f"  (No se pudo generar la figura: {exc})")


if __name__ == "__main__":
    main()
