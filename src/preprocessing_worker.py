"""
preprocessing_worker.py
------------------------
Módulo externo requerido por los notebooks 01 (preprocesamiento) y 03
(modelado ViT). Debe vivir en la MISMA carpeta que los notebooks/scripts
que lo importan.

Por qué vive en un .py separado y no dentro de una celda:
ProcessPoolExecutor y el DataLoader de PyTorch (num_workers>0) usan
multiprocessing con 'spawn' en Windows/macOS, y las clases/funciones deben
ser importables (picklables) desde un módulo real en disco. En Linux/Kabré
esto no es estrictamente necesario ('fork' funciona con clases definidas
inline), pero se mantiene el mismo patrón por consistencia y portabilidad
entre los distintos entornos donde ha corrido este proyecto (Windows local,
macOS de un compañero, y ahora Kabré/nukwa-l40s con GPU).

Contiene:
- preprocess_image(args): worker que decodifica, redimensiona y normaliza
  una imagen individual. Se usa con ProcessPoolExecutor en la Etapa 2.
- ChunkedNpyDataset: Dataset de PyTorch (map-style) que lee chunks .npy
  con indexado perezoso y caché del último chunk accedido.
- ChunkedNpyIterableDataset: versión IterableDataset, evita releer el mismo
  chunk una vez por muestra; cada worker recibe un subconjunto disjunto de
  chunks completos. Recomendada para el entrenamiento del ViT (Etapa 3).
"""

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import IterableDataset

# ── Parámetros de preprocesamiento (deben coincidir con el notebook) ─────────
IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(args):
    """
    Worker de preprocesamiento para una sola imagen.

    IMPORTANTE — formato de salida: devuelve un array uint8 SIN normalizar
    (valores de píxel 0-255 crudos, solo redimensionado + flip). La
    normalización ImageNet (resta de media, división por std) se aplica
    en tiempo de carga (ChunkedNpyDataset/__getitem__), no aquí.

    Motivo: guardar en float32 ya normalizado pesa 4x más en disco
    (20,000 imgs × 224×224×3 × 4 bytes ≈ 12 GB) que guardar en uint8
    crudo (≈ 3 GB), y la cuota de /data en Kabré es de 20 GB. La
    normalización on-the-fly es computacionalmente trivial frente al
    costo de decodificar/redimensionar la imagen, así que no hay
    penalización real de rendimiento por hacerla en el Dataset.

    Parameters
    ----------
    args : tuple(filepath, label, is_train)
        filepath : str o Path — ruta a la imagen original.
        label    : int        — 1 = real, 0 = fake.
        is_train : bool       — si True, aplica flip horizontal aleatorio
                                 (augmentation barata, no destructiva).

    Returns
    -------
    (arr, label) : tuple(np.ndarray | None, int)
        arr en formato (C, H, W) uint8, SIN normalizar. None si la imagen
        no pudo procesarse.
    """
    filepath, label, is_train = args

    try:
        with Image.open(filepath) as img:
            img = img.convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

            if is_train and np.random.rand() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            arr = np.asarray(img, dtype=np.uint8)      # (H, W, C) valores 0-255
            arr = arr.transpose(2, 0, 1)                # (H,W,C) -> (C,H,W)
            arr = np.ascontiguousarray(arr, dtype=np.uint8)

        return arr, label

    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ Error procesando {filepath}: {exc}")
        return None, label


def _normalize_uint8_to_tensor(img_uint8):
    """
    Convierte un array uint8 (C,H,W) crudo en 0-255 a un tensor float32
    normalizado con media/std de ImageNet, listo para el modelo.
    Se aplica en tiempo de carga (no al preprocesar) para ahorrar 4x
    espacio en disco. Ver nota en preprocess_image().
    """
    import torch
    arr = img_uint8.astype(np.float32) / 255.0          # (C,H,W) en [0,1]
    arr = arr.transpose(1, 2, 0)                          # (C,H,W) -> (H,W,C)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD             # normalización ImageNet
    arr = arr.transpose(2, 0, 1)                          # (H,W,C) -> (C,H,W)
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))


class ChunkedNpyDataset:
    """
    Dataset de PyTorch (map-style) que lee imágenes uint8 y labels desde
    chunks .npy generados por preprocess_split_parallel, y las normaliza
    a tensor float32 (ImageNet) en __getitem__.

    Estructura esperada en `split_dir`:
        images_chunk000.npy  (N, 3, 224, 224) uint8   [0-255, SIN normalizar]
        labels_chunk000.npy  (N,)             int8
        images_chunk001.npy  ...
        labels_chunk001.npy  ...

    Los chunks se indexan de forma perezosa: el índice global se traduce
    al chunk correspondiente y se cachea el último chunk leído para evitar
    recargar el mismo archivo repetidamente en iteración secuencial.
    """

    def __init__(self, split_dir):
        self.split_dir = Path(split_dir)

        self.image_files = sorted(self.split_dir.glob("images_chunk*.npy"))
        self.label_files = sorted(self.split_dir.glob("labels_chunk*.npy"))

        if not self.image_files:
            raise FileNotFoundError(
                f"No se encontraron chunks 'images_chunk*.npy' en {self.split_dir}."
            )
        if len(self.image_files) != len(self.label_files):
            raise ValueError(
                f"Número de chunks de imágenes ({len(self.image_files)}) no coincide "
                f"con el de labels ({len(self.label_files)}) en {self.split_dir}."
            )

        self._chunk_sizes = []
        for f in self.image_files:
            arr = np.load(f, mmap_mode="r")
            self._chunk_sizes.append(arr.shape[0])
            del arr

        self._cum_sizes = np.cumsum(self._chunk_sizes)
        self._total = int(self._cum_sizes[-1]) if len(self._cum_sizes) else 0

        self._cached_chunk_idx = None
        self._cached_images = None
        self._cached_labels = None

    def __len__(self):
        return self._total

    def _locate(self, idx):
        chunk_idx = int(np.searchsorted(self._cum_sizes, idx, side="right"))
        prev_cum = self._cum_sizes[chunk_idx - 1] if chunk_idx > 0 else 0
        offset = idx - prev_cum
        return chunk_idx, int(offset)

    def _load_chunk(self, chunk_idx):
        if self._cached_chunk_idx != chunk_idx:
            self._cached_images = np.load(self.image_files[chunk_idx])
            self._cached_labels = np.load(self.label_files[chunk_idx])
            self._cached_chunk_idx = chunk_idx
        return self._cached_images, self._cached_labels

    def __getitem__(self, idx):
        import torch

        if idx < 0:
            idx += self._total
        if idx < 0 or idx >= self._total:
            raise IndexError(f"Índice {idx} fuera de rango (0..{self._total - 1})")

        chunk_idx, offset = self._locate(idx)
        images, labels = self._load_chunk(chunk_idx)

        img = _normalize_uint8_to_tensor(images[offset])
        lbl = torch.tensor(int(labels[offset]), dtype=torch.long)
        return img, lbl


class ChunkedNpyIterableDataset(IterableDataset):
    """
    Versión iterable de ChunkedNpyDataset, pensada para eliminar el cuello
    de botella de I/O: cada chunk .npy se carga UNA sola vez por época
    (no una vez por muestra), completo en RAM, y desde ahí se entregan
    las muestras barajadas y normalizadas. Con num_workers>1, cada worker
    recibe un subconjunto DISJUNTO de chunks completos.

    Los chunks se almacenan como uint8 (crudo, sin normalizar) para
    ahorrar 4x espacio en disco; la normalización ImageNet se aplica
    aquí, en tiempo de carga.

    Recomendada para el entrenamiento del ViT (Etapa 3) porque el dataset
    completo (train: 16,000 imágenes x 224x224x3 uint8 ≈ 2.4 GB) puede
    manejarse cómodamente por chunks sin cargar todo de una vez.
    """

    def __init__(self, split_dir, shuffle=True, seed=0):
        self.split_dir = Path(split_dir)
        self.image_files = sorted(self.split_dir.glob("images_chunk*.npy"))
        self.label_files = sorted(self.split_dir.glob("labels_chunk*.npy"))

        if not self.image_files:
            raise FileNotFoundError(f"No se encontraron chunks en {self.split_dir}")
        if len(self.image_files) != len(self.label_files):
            raise ValueError("Número de chunks de imágenes y labels no coincide.")

        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        self._chunk_sizes = []
        for f in self.image_files:
            arr = np.load(f, mmap_mode="r")
            self._chunk_sizes.append(arr.shape[0])
            del arr
        self._total = int(sum(self._chunk_sizes))

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self._total

    def __iter__(self):
        import torch
        worker_info = torch.utils.data.get_worker_info()
        n_chunks = len(self.image_files)
        chunk_order = list(range(n_chunks))

        rng = np.random.default_rng(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(chunk_order)

        if worker_info is not None:
            chunk_order = chunk_order[worker_info.id::worker_info.num_workers]

        for c in chunk_order:
            images = np.load(self.image_files[c])
            labels = np.load(self.label_files[c])
            idxs = np.arange(images.shape[0])
            if self.shuffle:
                rng.shuffle(idxs)
            for i in idxs:
                img = _normalize_uint8_to_tensor(images[i])
                lbl = torch.tensor(int(labels[i]), dtype=torch.long)
                yield img, lbl
            del images, labels
