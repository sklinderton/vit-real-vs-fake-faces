# Detección de Rostros Sintéticos Generados por StyleGAN3 mediante Vision Transformers

**Curso:** Computación Paralela y Distribuida | LEAD University
**Profesor:** Johansell Villalobos Cubillo
**Equipo:** Jason Barrantes Sánchez · Melany Ramírez Anchía · Walter Bowyer Carpenter · Mauro Espinoza Hernández
**Plataforma:** Kabré Supercomputer — CeNAT, Costa Rica (partición `nukwa-l40s`, NVIDIA L40S)

---

## Estado del proyecto — Entrega 2 (Implementación Inicial)

| Etapa | Estado |
|---|---|
| 1. Adquisición y gestión de datos (Polars, Parquet) | ✅ Completo |
| 2. Preprocesamiento paralelo (ProcessPoolExecutor, Dask) | ✅ Completo |
| 3. Análisis exploratorio y estadístico | ✅ Completo |
| 4. Modelado con Vision Transformer | ✅ Completo — entrenado en GPU real |
| 5. Postprocesamiento y análisis de resultados | ✅ Completo para esta entrega |
| 6. Dashboard interactivo | 🔜 Próxima entrega |
| Generación StyleGAN3 (rostros nuevos para test de generalización) | 🔜 En progreso |

---

## Descripción del Problema

Clasificación binaria de rostros humanos: dado I ∈ R^(H×W×3), predecir si corresponde a
una fotografía real (y=1) o a un rostro sintético generado por StyleGAN3 (y=0). El proyecto
además cuantifica el beneficio de las herramientas de cómputo paralelo y distribuido usadas
en cada etapa del pipeline.

## Dataset

- **Fuente:** [10000 Real vs Fake Faces (StyleGAN3)](https://www.kaggle.com/datasets/troykueh/real-vs-fake-faces-stylegan3) — Kaggle
- **Total:** 20,000 imágenes (10,000 reales + 10,000 generadas por StyleGAN3)
- **Resolución original:** 1024×1024 px, 2.11 GB en disco
- **Split estratificado 80/10/10:** train=16,000 · val=2,000 · test=2,000

---

## Resultados Obtenidos

### Preprocesamiento Paralelo (ProcessPoolExecutor, 16 workers)

| Split | Imágenes | Tiempo | Throughput |
|---|---|---|---|
| Train | 16,000 | 6.8 s | 2,358 img/s |
| Val | 2,000 | 1.3 s | 1,550 img/s |
| Test | 2,000 | 1.4 s | 1,435 img/s |

Formato de almacenamiento: `uint8` crudo (sin normalizar), normalización ImageNet aplicada
en tiempo de carga — reduce el uso de disco 4× (12 GB → 2.9 GB) sin costo de rendimiento
relevante, crítico dado el límite de cuota de 20 GB en Kabré.

### Entrenamiento del Vision Transformer

- **Arquitectura:** ViT-B/16 (torchvision), preentrenado en ImageNet-1K, backbone completo sin congelar
- **Parámetros entrenables:** 85,800,194
- **Hardware:** NVIDIA L40S (46 GB VRAM), CUDA 12.1, cuDNN 9.1, precisión mixta (AMP fp16)
- **Optimizador:** AdamW, lr=1e-4, weight_decay=0.01, batch_size=32
- **Épocas:** 15 completadas en **10.75 minutos** (throughput promedio: 565 img/s)
- **Mejor modelo:** época 11 (checkpointing automático por AUC-ROC de validación, no por última época)

### Métricas en Test Set (2,000 imágenes, nunca vistas en entrenamiento)

| Métrica | Valor |
|---|---|
| Accuracy | 0.8700 |
| Precision | 0.8376 |
| Recall | 0.9180 |
| F1-Score | 0.8760 |
| **AUC-ROC** | **0.9438** |

| Clase | Precision | Recall | F1 |
|---|---|---|---|
| Fake | 0.91 | 0.82 | 0.86 |
| Real | 0.84 | 0.92 | 0.88 |

**Análisis de errores:** el modelo detecta rostros reales (recall 92%) mejor que rostros
generados (recall 82%), consistente con el objetivo de diseño de StyleGAN3 de minimizar
artefactos detectables. El modelo tiende a clasificar rostros fake difíciles como reales
más que al revés.

### Interpretabilidad y Validación Cualitativa

- Mapas de atención del token `[CLS]` extraídos interceptando `nn.MultiheadAttention`
- Función de clasificación de fotos arbitrarias: probada con una foto personal fuera del
  dataset, clasificada correctamente como **REAL con 99.99% de confianza**

---

## Gestión de Problemas (obstáculos encontrados y solución aplicada)

| # | Obstáculo | Solución |
|---|---|---|
| 1 | SSH directo bloqueado por firewall institucional | Acceso vía Open OnDemand Shell (SSH funcionalmente equivalente) |
| 2 | Home de Kabré limitado a 10 GB, se llenó dos veces | Migración completa del proyecto a `/data` (Lustre, cuota 20 GB) |
| 3 | Particiones CPU estándar (`kura`) sin GPU | Descubrimiento y verificación de particiones `nukwa-v100`/`nukwa-l40s` vía `nvidia-smi` |
| 4 | Sintaxis `--gres=gpu:1` no soportada en Kabré | GPU se asigna automáticamente por partición, sin bandera `--gres` |
| 5 | Escritura de disco interrumpida a mitad de un chunk (cuota excedida) | Rediseño de almacenamiento a `uint8` (4× más liviano) con normalización on-the-fly |
| 6 | Límite de 4 horas por trabajo SLURM en GPU | `train_vit.py` con checkpointing por época y bandera `--resume` |
| 7 | Sobreajuste detectado desde época ~10-12 | Checkpointing por mejor AUC de validación (no última época) selecciona automáticamente el modelo con mejor generalización |
| 8 | Intento previo del equipo en CPU (Mac) inviable — 85.8M parámetros | Migración a GPU real en Kabré resolvió el cuello de botella computacional |

---

## Requisitos de Software

```
torch, torchvision (CUDA 12.1)
polars, dask[dataframe], pyarrow
Pillow, numpy, pandas, scikit-learn
matplotlib, seaborn
jupyter, jupyterlab, ipykernel
imagehash, scipy (usados en EDA avanzado)
```

Ver `requirements.txt` para versiones exactas.

## Instrucciones de Instalación

```bash
# En Kabré, dentro de una sesión con GPU (nukwa-l40s o nukwa-v100)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

uv venv /data/<usuario>/envs/vit_faces --python 3.11
source /data/<usuario>/envs/vit_faces/bin/activate

uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install polars "dask[dataframe]" pyarrow Pillow tqdm pandas numpy \
    matplotlib seaborn scikit-learn ipykernel jupyter jupyterlab nbconvert \
    kaggle imagehash scipy

python -m ipykernel install --user --name=vit_faces --display-name "Python (vit_faces)"
```

## Instrucciones de Ejecución

```bash
# 1. Preprocesamiento paralelo (regenera data/processed/ desde data/raw/)
python run_preprocessing.py

# 2. Entrenamiento del ViT (como trabajo SLURM, corre desatendido)
sbatch submit_train_vit.sh
squeue -u $USER                              # verificar estado
tail -f logs/vit_train_<JOBID>.log           # seguir progreso

# 3. Evaluación (notebook interactivo, requiere checkpoints/vit_best.pt)
jupyter lab
# Abrir notebooks/03_evaluacion_vit.ipynb, kernel "Python (vit_faces)"
```

> Los checkpoints entrenados (`checkpoints/*.pt`) **no están incluidos en este
> repositorio** por exceder el límite de tamaño de archivo de GitHub (~1GB+ cada
> uno, dado que ViT-B/16 tiene 85.8M parámetros). Ejecuta `sbatch submit_train_vit.sh`
> para regenerarlos — el entrenamiento completo toma ~11 minutos en una GPU L40S.

## Instrucciones para Obtener el Dataset

```bash
mkdir -p ~/.kaggle
echo "TU_TOKEN_KAGGLE" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token

kaggle datasets download -d troykueh/real-vs-fake-faces-stylegan3
unzip real-vs-fake-faces-stylegan3.zip -d data/raw/
```

---

## Estructura del Repositorio

```
proyecto_paralela/
├── data/
│   ├── raw/                          # Dataset original (no incluido, ver instrucciones)
│   ├── processed/                    # Tensores uint8 preprocesados (no incluido, regenerable)
│   ├── metadata/                     # Inventario Parquet + estadísticas
│   ├── plots/                        # Gráficas del EDA
│   └── uploads/                      # Carpeta para fotos propias a clasificar
├── notebooks/
│   ├── 01_adquisicion_preprocesamiento.ipynb
│   ├── 02_eda_avanzado.ipynb
│   └── 03_evaluacion_vit.ipynb
├── src/
│   └── preprocessing_worker.py       # Worker paralelo + Dataset classes (uint8 + norm on-the-fly)
├── run_preprocessing.py              # Script de preprocesamiento paralelo standalone
├── train_vit.py                      # Entrenamiento ViT con checkpointing
├── submit_train_vit.sh               # Job SLURM (nukwa-l40s, 4h límite)
├── checkpoints/                      # vit_best.pt / vit_last.pt (excluidos de git)
├── figures/                          # Matriz de confusión, ROC, atención, curvas
├── results/                          # training_log.csv, evaluation_summary.json, etc.
├── logs/                             # Logs de los trabajos SLURM
├── requirements.txt
└── README.md
```

## Entorno de Cómputo

| Recurso | Detalle |
|---|---|
| Plataforma | Kabré Supercomputer — CeNAT |
| Partición | `nukwa-l40s` |
| GPU | NVIDIA L40S, 46 GB VRAM |
| CPUs por trabajo | 16 |
| RAM por trabajo | 64 GB |
| CUDA / cuDNN | 12.1 / 9.1 |
| Límite por trabajo | 4 horas |
| Cuota de almacenamiento | 20 GB (`/data`, filesystem Lustre) |

---

*Jason Barrantes Sánchez · sklinderton · LEAD University · 2026*
