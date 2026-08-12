"""
eda_worker.py
─────────────
Funciones "worker" usadas por 02_eda_avanzado.ipynb con ProcessPoolExecutor.

Deben vivir en un módulo .py separado (no definidas dentro del notebook) para
que puedan serializarse correctamente con pickle. Esto es estrictamente
necesario en Windows/macOS (método 'spawn'), y también funciona sin problemas
en Linux/Kabré (método 'fork'), así que dejamos el mismo patrón en ambos casos
por consistencia con preprocessing_worker.py de la Etapa 1-2.

Coloca este archivo en la MISMA carpeta que 02_eda_avanzado.ipynb.
"""

import numpy as np
from PIL import Image
from scipy import ndimage

try:
    import imagehash
    _HAS_IMAGEHASH = True
except ImportError:
    _HAS_IMAGEHASH = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. Verificación de integridad
# ─────────────────────────────────────────────────────────────────────────────
def verify_image_integrity(args):
    """
    Verifica que una imagen se pueda abrir y decodificar completamente.

    Args:
        args: tuple (filepath, label, cls)

    Returns:
        dict con: filepath, label, class, ok (bool), error (str|None),
                  width, height (None si falló)
    """
    filepath, label, cls = args
    try:
        # verify() detecta archivos truncados/corruptos sin decodificar pixeles
        with Image.open(filepath) as im:
            im.verify()

        # Reabrir para leer dimensiones (verify() deja el objeto inutilizable)
        with Image.open(filepath) as im:
            im_rgb = im.convert("RGB")
            w, h = im_rgb.size

        return {
            "filepath": str(filepath), "label": label, "class": cls,
            "ok": True, "error": None, "width": w, "height": h,
        }
    except Exception as exc:
        return {
            "filepath": str(filepath), "label": label, "class": cls,
            "ok": False, "error": str(exc), "width": None, "height": None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Extracción de features visuales interpretables (+ phash)
# ─────────────────────────────────────────────────────────────────────────────
def compute_visual_features(args):
    """
    Extrae un vector de features visuales interpretables de una imagen.

    Args:
        args: tuple (filepath, label, cls)

    Returns:
        dict con filepath, label, class, width, height, aspect_ratio,
        brightness, contrast, saturation, sharpness_laplacian, edge_density,
        entropy, unique_colors_ratio, phash
    """
    filepath, label, cls = args
    try:
        with Image.open(filepath) as im:
            im_rgb = im.convert("RGB")
            w, h = im_rgb.size

            gray = np.asarray(im_rgb.convert("L"), dtype=np.float32)  # (H, W)

            # ── Brillo y contraste (escala de grises) ──────────────────────
            brightness = float(gray.mean())
            contrast = float(gray.std())

            # ── Saturación (canal S de HSV) ─────────────────────────────────
            hsv = np.asarray(im_rgb.convert("HSV"), dtype=np.float32)
            saturation = float(hsv[:, :, 1].mean())

            # ── Nitidez: varianza del Laplaciano ─────────────────────────────
            laplacian = ndimage.laplace(gray)
            sharpness_laplacian = float(laplacian.var())

            # ── Densidad de bordes: magnitud de Sobel ────────────────────────
            sx = ndimage.sobel(gray, axis=0)
            sy = ndimage.sobel(gray, axis=1)
            sobel_mag = np.hypot(sx, sy)
            edge_density = float((sobel_mag > sobel_mag.mean()).mean())

            # ── Entropía de Shannon (histograma de grises, 256 bins) ────────
            hist, _ = np.histogram(gray, bins=256, range=(0, 255))
            p = hist.astype(np.float64)
            p = p[p > 0]
            p = p / p.sum()
            img_entropy = float(-np.sum(p * np.log2(p)))

            # ── Diversidad de color (sobre versión reducida por costo) ──────
            # Contar colores únicos en la imagen a resolución completa es
            # costoso en paralelo sobre miles de imágenes; se reduce a 256x256
            # como aproximación razonable para EDA (no afecta el orden relativo
            # real vs fake, que es lo que nos interesa comparar aquí).
            small = im_rgb.resize((256, 256))
            colors = small.getcolors(maxcolors=256 * 256)
            n_unique = len(colors) if colors is not None else 256 * 256
            unique_colors_ratio = float(n_unique / (256 * 256))

            # ── Perceptual hash (para detección de duplicados) ──────────────
            phash = str(imagehash.phash(im_rgb)) if _HAS_IMAGEHASH else None

            return {
                "filepath": str(filepath), "label": label, "class": cls,
                "width": w, "height": h, "aspect_ratio": round(w / h, 4),
                "brightness": round(brightness, 4),
                "contrast": round(contrast, 4),
                "saturation": round(saturation, 4),
                "sharpness_laplacian": round(sharpness_laplacian, 4),
                "edge_density": round(edge_density, 4),
                "entropy": round(img_entropy, 4),
                "unique_colors_ratio": round(unique_colors_ratio, 6),
                "phash": phash,
            }
    except Exception as exc:
        return {
            "filepath": str(filepath), "label": label, "class": cls,
            "width": None, "height": None, "error": str(exc),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Espectro de potencia radial (FFT) — huellas de GAN
# ─────────────────────────────────────────────────────────────────────────────
def compute_radial_power_spectrum(args, n_bins=64):
    """
    Calcula el espectro de potencia radial promedio (2D FFT) de una imagen.
    Las GANs suelen dejar artefactos periódicos (p. ej. por upsampling con
    transposed convolutions o PixelShuffle) visibles en este espectro.

    Args:
        args   : tuple (filepath, label, cls)
        n_bins : número de anillos radiales de frecuencia

    Returns:
        dict con filepath, label, class, radial_profile (lista de n_bins floats)
    """
    filepath, label, cls = args
    try:
        with Image.open(filepath) as im:
            gray = np.asarray(im.convert("L").resize((256, 256)), dtype=np.float32)

        # FFT 2D → centrar frecuencia cero → magnitud logarítmica
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        y, x = np.indices((h, w))
        r = np.hypot(x - cx, y - cy)
        r_max = r.max()

        bin_edges = np.linspace(0, r_max, n_bins + 1)
        radial_profile = np.zeros(n_bins, dtype=np.float32)
        for i in range(n_bins):
            mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
            radial_profile[i] = magnitude[mask].mean() if mask.any() else 0.0

        return {
            "filepath": str(filepath), "label": label, "class": cls,
            "radial_profile": radial_profile.tolist(),
        }
    except Exception as exc:
        return {
            "filepath": str(filepath), "label": label, "class": cls,
            "error": str(exc),
        }
