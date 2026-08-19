"""
engine.py — Carga del modelo, inferencia e instrumentos de lectura
==================================================================
Aísla todo lo que toca PyTorch para que app.py se ocupe solo de la interfaz.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import vit_b_16

import theme

IMG_SIZE = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════
# Ensamble
# ══════════════════════════════════════════════════════════════════════════

# Umbral de decisión fijado sobre el conjunto de validación difícil, nunca
# sobre la prueba final. Queda por debajo de 0.5 porque el entrenamiento
# tenía más rostros sintéticos que auténticos (57 % frente a 43 %), lo que
# desplaza el punto de equilibrio del modelo hacia la sospecha.
DECISION_THRESHOLD = 0.285

MEMBERS = [
    ("vit_v5_cnx.pt", "convnext", "ConvNeXt-Tiny"),
    ("vit_v5_swin.pt", "swin", "Swin-T"),
]


def _build(arch):
    """Construye la arquitectura vacía a la que se cargarán los pesos."""
    import torchvision.models as tvm

    if arch == "convnext":
        m = tvm.convnext_tiny(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, 2)
    elif arch == "swin":
        m = tvm.swin_t(weights=None)
        m.head = nn.Linear(m.head.in_features, 2)
    elif arch == "vit":
        from torchvision.models import vit_b_16
        m = vit_b_16(weights=None, image_size=IMG_SIZE)
        m.heads.head = nn.Linear(m.heads.head.in_features, 2)
    else:
        raise ValueError(f"Arquitectura desconocida: {arch}")
    return m


def load_ensemble(assets_dir):
    """
    Carga los miembros del ensamble disponibles en la carpeta de recursos.

    Devuelve (modelos, device, meta). Si no encuentra ninguno, modelos es una
    lista vacía y la aplicación muestra instrucciones en lugar del análisis.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assets_dir = Path(assets_dir)
    models, names, total_params = [], [], 0

    for filename, arch, label in MEMBERS:
        path = assets_dir / filename
        if not path.exists():
            continue
        m = _build(arch)
        ck = torch.load(path, map_location=device, weights_only=False)
        m.load_state_dict(ck["model_state"])
        m.to(device).eval()
        models.append((m, label))
        names.append(label)
        total_params += sum(p.numel() for p in m.parameters())

    if not models:
        return [], device, None

    meta = {"names": names, "params": total_params, "n": len(models),
            "threshold": DECISION_THRESHOLD}
    return models, device, meta


def preprocess(pil_img):
    """Redimensiona a 224 y normaliza con las estadísticas de ImageNet."""
    shown = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(shown, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)), shown


def analyze(pil_img, models, device):
    """
    Clasifica una imagen promediando las predicciones de todos los miembros
    del ensamble, cada uno evaluado sobre la imagen y su espejo horizontal.

    El espejo no cambia si un rostro es auténtico o generado, así que
    promediar ambas vistas reduce la varianza de la predicción sin sesgarla.

    El mapa de atención se toma del primer miembro que lo permita. ConvNeXt
    no tiene atención —es convolucional— así que en su caso se usa el mapa de
    activación de la última etapa, que cumple el mismo papel explicativo:
    señalar qué zonas pesaron en la decisión.
    """
    tensor, shown = preprocess(pil_img)
    batch = tensor.unsqueeze(0).to(device)
    flipped = torch.flip(batch, dims=[3])

    probs_fake = []
    for model, _ in models:
        with torch.no_grad():
            p1 = torch.softmax(model(batch), dim=1)[0]
            p2 = torch.softmax(model(flipped), dim=1)[0]
        probs_fake.append(((p1 + p2) / 2).cpu().numpy())

    avg = np.mean(probs_fake, axis=0)
    heat = _activation_map(models[0][0], batch) if models else None

    return {"p_fake": float(avg[0]), "p_real": float(avg[1]),
            "image": shown, "attention": heat,
            "per_model": [float(p[0]) for p in probs_fake]}


def _activation_map(model, batch):
    """
    Mapa de la última capa convolucional o de atención, según la arquitectura.

    Se engancha un hook temporal a la última etapa de características y se
    promedia su salida sobre los canales. El resultado indica qué regiones
    activaron más fuertemente al modelo.
    """
    feats = {}

    def hook(_m, _inp, out):
        feats["out"] = out.detach()

    # Localizar la última etapa de características según la arquitectura
    target = None
    if hasattr(model, "features"):          # ConvNeXt
        target = model.features[-1]
    elif hasattr(model, "layers"):          # Swin
        target = model.layers[-1]
    elif hasattr(model, "encoder"):         # ViT
        target = model.encoder.layers[-1]

    if target is None:
        return None

    h = target.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(batch)
    finally:
        h.remove()

    if "out" not in feats:
        return None

    t = feats["out"][0]
    # ConvNeXt entrega (C,H,W); Swin entrega (H,W,C)
    if t.ndim == 3:
        if t.shape[0] > t.shape[-1]:
            t = t.permute(2, 0, 1)          # (H,W,C) -> (C,H,W)
        m = t.mean(dim=0)
    elif t.ndim == 2:                        # (tokens, dim) del ViT
        n = int(round((t.shape[0] - 1) ** 0.5))
        if n * n != t.shape[0] - 1:
            return None
        m = t[1:].mean(dim=1).reshape(n, n)
    else:
        return None

    return m.float().cpu().numpy()


def attention_overlay(pil_img, attn, alpha=0.5):
    """Superpone el mapa de atención sobre la imagen."""
    if attn is None:
        return None
    import matplotlib.cm as cm

    a = attn - attn.min()
    a = a / (a.max() + 1e-8)
    heat = Image.fromarray((cm.magma(a) * 255).astype(np.uint8)[:, :, :3], "RGB")
    heat = heat.resize(pil_img.size, Image.BICUBIC)
    return Image.blend(pil_img.convert("RGB"), heat, alpha=alpha)


# ══════════════════════════════════════════════════════════════════════════
# Dial de veredicto — elemento distintivo de la lectura
# ══════════════════════════════════════════════════════════════════════════

def verdict_dial(p_fake: float) -> str:
    """
    Dial de doble arco. Dos semiarcos opuestos —teal para auténtica, violeta
    para sintética— se encuentran en el vértice superior, que marca el umbral
    de decisión. La aguja indica la probabilidad medida.

    Se prefiere a una barra de progreso porque una barra sugiere "avance
    completado", mientras que un dial comunica medición sobre una escala con
    un punto de corte, que es lo que realmente ocurre aquí.

    La animación usa elementos SMIL (<animateTransform>) en lugar de un bloque
    <style> interno: Streamlit descarta el SVG completo si detecta una etiqueta
    <style> dentro, lo que hacía que el dial se renderizara como texto suelto.
    """
    W, H = 400, 232
    cx, cy, r = W / 2, 186.0, 138.0
    stroke = 15

    def polar(frac, radius):
        ang = np.pi * (1 - frac)
        return cx + radius * np.cos(ang), cy - radius * np.sin(ang)

    x0, y0 = polar(0.0, r)
    xm, ym = polar(0.5, r)
    x1, y1 = polar(1.0, r)
    nx, ny = polar(p_fake, r - 30)

    is_fake = p_fake > 0.5
    marker = theme.SYNTHETIC if is_fake else theme.AUTHENTIC

    # Grado inicial de la aguja para el barrido (arranca desde el extremo opuesto)
    sweep_from = -180 * (1 - p_fake) + 90

    ticks = ""
    for i in range(11):
        f = i / 10
        major = i % 5 == 0
        r_out = r + stroke / 2 + 4
        r_in = r_out + (10 if major else 5)
        ax, ay = polar(f, r_out)
        bx, by = polar(f, r_in)
        ticks += (f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" '
                  f'stroke="{theme.RULE}" stroke-width="{1.6 if major else 1}"/>')
        if major:
            lx, ly = polar(f, r_in + 13)
            ticks += (f'<text x="{lx:.1f}" y="{ly + 4:.1f}" fill="{theme.TEXT_FAINT}" '
                      f'font-family="JetBrains Mono, monospace" font-size="10" '
                      f'text-anchor="middle">{f:.1f}</text>')

    op_auth = 0.95 if not is_fake else 0.16
    op_synt = 0.95 if is_fake else 0.16

    return f"""
<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px; display:block"
     xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Probabilidad de que la imagen sea sintética: {p_fake:.3f}">
  <path d="M {x0:.1f} {y0:.1f} A {r} {r} 0 0 1 {xm:.1f} {ym:.1f}"
        fill="none" stroke="{theme.AUTHENTIC}" stroke-width="{stroke}"
        stroke-linecap="butt" opacity="{op_auth}"
        stroke-dasharray="460" stroke-dashoffset="460">
    <animate attributeName="stroke-dashoffset" from="460" to="0"
             dur="0.85s" fill="freeze" calcMode="spline"
             keySplines="0.25 0.8 0.3 1" keyTimes="0;1"/>
  </path>
  <path d="M {xm:.1f} {ym:.1f} A {r} {r} 0 0 1 {x1:.1f} {y1:.1f}"
        fill="none" stroke="{theme.SYNTHETIC}" stroke-width="{stroke}"
        stroke-linecap="butt" opacity="{op_synt}"
        stroke-dasharray="460" stroke-dashoffset="460">
    <animate attributeName="stroke-dashoffset" from="460" to="0"
             dur="0.85s" begin="0.1s" fill="freeze" calcMode="spline"
             keySplines="0.25 0.8 0.3 1" keyTimes="0;1"/>
  </path>
  <line x1="{xm:.1f}" y1="{ym - stroke:.1f}" x2="{xm:.1f}" y2="{ym + stroke:.1f}"
        stroke="{theme.TEXT_DIM}" stroke-width="2"/>
  {ticks}
  <g opacity="0">
    <animate attributeName="opacity" from="0" to="1" dur="0.3s"
             begin="0.35s" fill="freeze"/>
    <animateTransform attributeType="XML" attributeName="transform" type="rotate"
                      from="{sweep_from:.1f} {cx} {cy}" to="0 {cx} {cy}"
                      dur="0.75s" begin="0.35s" fill="freeze"
                      calcMode="spline" keySplines="0.3 1.2 0.4 1" keyTimes="0;1"/>
    <line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}"
          stroke="{theme.TEXT}" stroke-width="2.6" stroke-linecap="round"/>
    <circle cx="{nx:.1f}" cy="{ny:.1f}" r="7.5" fill="{marker}"
            stroke="{theme.BG}" stroke-width="2"/>
  </g>
  <circle cx="{cx}" cy="{cy}" r="6" fill="{theme.TEXT}"/>
  <text x="{x0 - 4:.1f}" y="{cy + 24:.1f}" fill="{theme.AUTHENTIC}"
        font-family="JetBrains Mono, monospace" font-size="10"
        letter-spacing="1.5" font-weight="600">AUTÉNTICA</text>
  <text x="{x1 + 4:.1f}" y="{cy + 24:.1f}" fill="{theme.SYNTHETIC}"
        font-family="JetBrains Mono, monospace" font-size="10"
        letter-spacing="1.5" font-weight="600" text-anchor="end">SINTÉTICA</text>
</svg>
"""


def attention_stats(attn):
    """
    Resume el mapa de atención: cuán concentrada está la mirada del modelo.
    Un valor alto de concentración indica que se fijó en pocas zonas; uno bajo,
    que repartió la atención por toda la imagen.
    """
    if attn is None:
        return None
    a = attn.flatten()
    a = a / (a.sum() + 1e-9)
    # Entropía normalizada: 1 = atención repartida, 0 = concentrada en un punto
    ent = -(a * np.log(a + 1e-12)).sum() / np.log(len(a))
    top10 = np.sort(a)[::-1][:int(len(a) * 0.1)].sum()
    return {"dispersion": float(ent), "top10_share": float(top10)}
