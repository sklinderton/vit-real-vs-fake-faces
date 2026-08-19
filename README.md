# Detección de Rostros Sintéticos Generados por StyleGAN3

**Curso:** Computación Paralela y Distribuida · Universidad LEAD
**Profesor:** Johansell Villalobos Cubillo
**Equipo:** Jason Barrantes Sánchez · Melany Ramírez Anchía · Walter Bowyer Carpenter · Mauro Espinoza Hernández
**Infraestructura:** Clúster Kabré (CeNAT), partición `nukwa-l40s` con NVIDIA L40S

---

## Resumen

Sistema de clasificación binaria que distingue fotografías auténticas de rostros
generados por StyleGAN3. Alcanza **99.7 % de aciertos** y **AUC-ROC de 0.9999**
sobre rostros creados con semillas que el modelo nunca vio, al máximo de
diversidad del generador.

El aporte principal no es el clasificador sino el diagnóstico que lo precedió.
La primera versión obtenía 82 % sobre el conjunto público de referencia pero
solo **23 %** sobre rostros que el mismo generador producía en el momento. La
causa resultó ser un sesgo en la construcción del conjunto público, identificado
tras descartar experimentalmente dos hipótesis alternativas.

---

## Resultados

### Evolución sobre el conjunto difícil

Rostros con truncamiento ψ = 1.0 y semillas disjuntas del entrenamiento.

| Versión | Aciertos | AUC-ROC | Detecta sintéticas |
|---|---|---|---|
| v1 — línea base | — | — | 23.0 % |
| v2 — datos corregidos | — | — | 60.9 % |
| v3 — ajuste de ritmo de aprendizaje | 89.8 % | 0.9534 | 87.0 % |
| v4 — augmentación y umbral | 92.0 % | 0.9679 | 92.8 % |
| v5 — tres ViT promediados | 94.0 % | 0.9832 | 96.2 % |
| **ConvNeXt-Tiny + Swin-T** | **99.7 %** | **0.9999** | **99.8 %** |

### Comparación de arquitecturas

Resultado contrario a la expectativa inicial del proyecto: la red convolucional
supera al Vision Transformer con un tercio de los parámetros.

| Arquitectura | Parámetros | Aciertos | AUC-ROC |
|---|---|---|---|
| ViT-B/16 | 85.8 M | 93.3 % | 0.9778 |
| Swin-T | 27.5 M | 97.3 % | 0.9995 |
| **ConvNeXt-Tiny** | **27.8 M** | **99.6 %** | **0.9999** |

Los rastros que deja StyleGAN3 —textura de piel, transición del cabello, patrones
del iris— operan a una escala menor que los parches de 16×16 píxeles del ViT, que
los promedia al proyectarlos. Las convoluciones jerárquicas los preservan.

### Rendimiento del preprocesamiento paralelo

| Procesos | Tiempo (s) | Imágenes/s |
|---|---|---|
| 1 | 3.913 | 511 |
| 4 | 2.470 | 810 |
| 8 | 1.354 | 1 477 |
| 12 | 1.061 | 1 884 |
| 16 | 0.910 | 2 198 |

> **Limitación documentada.** La referencia secuencial resultó inestable pese a
> calentar la caché, fijar las bibliotecas numéricas a un solo hilo y promediar
> repeticiones. La causa probable es que SLURM restringe el trabajo a 16 de los 20
> núcleos físicos mediante cgroups, y el reparto interno entre proceso padre e hijos
> queda a criterio del planificador del sistema. Se reporta el rendimiento absoluto,
> que sí es consistente, y se omite el cálculo formal de speedup. Ver sección VI-A-1
> del informe.

---

## El diagnóstico

Tres hipótesis falsables, evaluadas experimentalmente:

| Hipótesis | Veredicto | Evidencia |
|---|---|---|
| Compresión distinta entre clases | Descartada | Todo el conjunto es PNG sin pérdida. Diferencia en bytes por píxel: d = 0.093 |
| Filtro de redimensionado distinto | Descartada | Seis filtros probados; detección entre 18.5 % y 21.5 % (3 puntos de rango) |
| **Truncamiento del generador** | **Confirmada** | De 100 % con ψ = 0.40 a 23 % con ψ = 1.00 (77 puntos de rango) |

El análisis espectral confirmó que el generador era el mismo: la distancia entre
los perfiles de frecuencia de las sintéticas del conjunto y las generadas
(0.00733) resultó menor que entre las dos clases del propio conjunto (0.01055).

El conjunto público se había generado con truncamiento bajo, produciendo rostros
suavizados. El modelo aprendió a reconocer esa suavidad en lugar de los artefactos
del generador.

---

## Requisitos

```
torch, torchvision (CUDA 12.1)
polars, dask[dataframe], pyarrow
Pillow, numpy, pandas, scikit-learn
matplotlib, seaborn, plotly
streamlit
jupyter, jupyterlab, ipykernel
kaggle, imagehash, scipy, ninja, setuptools<81
```

`setuptools<81` es necesario porque StyleGAN3 depende de `pkg_resources`, removido
en versiones posteriores.

## Instalación

```bash
# En un nodo con GPU del clúster
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

uv venv /data/<usuario>/envs/vit_faces --python 3.11
source /data/<usuario>/envs/vit_faces/bin/activate

uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install polars "dask[dataframe]" pyarrow Pillow tqdm pandas numpy \
    matplotlib seaborn scikit-learn ipykernel jupyter jupyterlab nbconvert \
    kaggle imagehash scipy ninja "setuptools<81"

python -m ipykernel install --user --name=vit_faces --display-name "Python (vit_faces)"
```

### Dependencias externas

```bash
git clone https://github.com/NVlabs/stylegan3.git /data/<usuario>/stylegan3

mkdir -p /data/<usuario>/stylegan3_models
cd /data/<usuario>/stylegan3_models
wget https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl
```

### Conjunto de datos

```bash
mkdir -p ~/.kaggle
echo "TU_TOKEN" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token

cd data/raw
kaggle datasets download -d troykueh/real-vs-fake-faces-stylegan3
unzip real-vs-fake-faces-stylegan3.zip
```

---

## Reproducción completa

```bash
# 1. Verificar que el modelo se construye a 256 px (1 min)
python test_256_setup.py

# 2. Construir el conjunto: 9 000 rostros generados con ψ estratificado (6 min)
python build_dataset_v4.py

# 3. Entrenar los dos miembros del ensamble (21 min)
python train_hetero.py --arch convnext --tag cnx
python train_hetero.py --arch swin     --tag swin

# 4. Evaluar el ensamble y fijar el umbral (2 min)
python evaluate_ensemble.py --tags cnx swin

# 5. Medir el rendimiento paralelo (6 min)
python benchmark_paralelo.py --images 2000
```

Los rostros sintéticos **no se almacenan**: se registran sus semillas y valores de
ψ en `data/metadata/seeds_manifest_v4.json`, lo que permite regenerarlos de forma
determinista.

Esta propiedad se verificó de forma involuntaria: una limpieza de disco eliminó
los modelos y el conjunto procesado. La reconstrucción completa tomó 35 minutos y
devolvió resultados **idénticos hasta el cuarto decimal**.

---

## Dashboard

Aplicación Streamlit que corre en máquina local, sin GPU.

```bash
cd dashboard
uv venv .venv --python 3.11
source .venv/bin/activate      # en Windows: .\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
streamlit run app.py
```

Requiere copiar a `dashboard/assets/`:

| Origen | Destino |
|---|---|
| `checkpoints/vit_v5_cnx.pt` | `assets/` |
| `checkpoints/vit_v5_swin.pt` | `assets/` |
| `results/evaluation_summary_ensemble.json` | `assets/` |
| `results/training_log_cnx.csv` | `assets/` |
| `dashboard/assets/stylegan3_pool/` | ya incluido |

Tres modos de entrada: subir un archivo, tomar una foto con la cámara, o pedir un
rostro generado por StyleGAN3. Incluye secciones explicativas del modelo, del
diagnóstico y de los resultados, con gráficas interactivas.

---

## Estructura

```
proyecto_paralela/
├── data/
│   ├── raw/                       # Conjunto público (no versionado)
│   ├── processed_v4/              # Tensores uint8 a 256 px (no versionado)
│   ├── metadata/                  # Inventarios y manifiestos de semillas
│   ├── eda_avanzado/              # Resultados del análisis exploratorio
│   └── plots/
├── notebooks/
│   ├── 01_adquisicion_preprocesamiento.ipynb
│   ├── 02_eda_avanzado.ipynb
│   └── 03_evaluacion_vit.ipynb
├── src/
│   ├── preprocessing_worker.py    # Worker paralelo, Datasets y augmentación
│   └── eda_worker.py
├── dashboard/                     # Aplicación Streamlit
│   ├── app.py  theme.py  engine.py  charts.py
│   └── .streamlit/config.toml
├── build_dataset_v4.py            # Construcción del conjunto a 256 px
├── train_hetero.py                # Entrenamiento de ConvNeXt y Swin
├── train_vit_v5.py                # Entrenamiento del ViT (comparación)
├── evaluate_ensemble.py           # Ensamble y selección de umbral
├── benchmark_paralelo.py          # Medición de rendimiento paralelo
├── generate_stylegan3.py          # Generación y prueba de generalización
├── diagnose_compression_bias.py   # Hipótesis 1
├── test_resampling_effect.py      # Hipótesis 2
├── compare_generators.py          # Hipótesis 3 y análisis espectral
├── figures/  results/  logs/  report/
└── requirements.txt
```

---

## Alcance y limitaciones

El sistema distingue **rostros de StyleGAN3-R frente a fotografías de FFHQ**. No
es un detector universal de contenido generado por inteligencia artificial: ante
modelos de difusión como Stable Diffusion o Midjourney, o incluso versiones
anteriores de StyleGAN, cabe esperar un desempeño sustancialmente menor.

Esa expectativa se apoya en el propio hallazgo del trabajo: si un cambio en el
parámetro de truncamiento del mismo generador bastó para llevar la detección del
82 % al 23 %, un cambio de arquitectura generativa producirá un desplazamiento
mayor.

---

## Licencias

| Componente | Licencia |
|---|---|
| PyTorch, torchvision | BSD-3-Clause |
| StyleGAN3 (NVIDIA) | Uso no comercial |
| Polars | MIT |
| Streamlit | Apache 2.0 |
| Plotly | MIT |
| Conjunto de datos (Kaggle) | CC-BY-NC-SA-4.0 |

Trabajo académico sin fin comercial.
