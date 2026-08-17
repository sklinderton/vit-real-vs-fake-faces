"""
test_256_setup.py
------------------
Comprobación rápida (menos de un minuto) de que el modelo se puede construir
a 256 píxeles antes de lanzar un entrenamiento completo.

Verifica tres cosas:
  1. Que torchvision expone interpolate_embeddings con la firma esperada.
  2. Que las codificaciones posicionales se reescalan de 14x14 a 16x16.
  3. Que una pasada hacia adelante con un lote de 256 px no falla ni agota
     la memoria de la GPU.

Uso:
    python test_256_setup.py
"""

import sys
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

IMG = 256
BATCH = 32

print("=" * 60)
print("COMPROBACIÓN DEL MONTAJE A 256 PX")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

# ── 1. API disponible ──────────────────────────────────────────────────
try:
    from torchvision.models.vision_transformer import interpolate_embeddings
    print("  [1/3] interpolate_embeddings disponible ✓")
except ImportError as e:
    print(f"  [1/3] ✗ No disponible: {e}")
    print("        Actualiza torchvision o avísame para usar la alternativa manual.")
    sys.exit(1)

# ── 2. Interpolación de posicionales ───────────────────────────────────
weights = ViT_B_16_Weights.IMAGENET1K_V1
state = weights.get_state_dict(progress=True)
before = state["encoder.pos_embedding"].shape

state = interpolate_embeddings(image_size=IMG, patch_size=16, model_state=state)
after = state["encoder.pos_embedding"].shape
print(f"  [2/3] pos_embedding {tuple(before)} → {tuple(after)} ✓")
assert after[1] == (IMG // 16) ** 2 + 1, "El número de parches no cuadra"

# ── 3. Pasada hacia adelante ───────────────────────────────────────────
model = vit_b_16(weights=None, image_size=IMG)
model.load_state_dict(state)
model.heads.head = nn.Linear(model.heads.head.in_features, 2)
model.to(device).eval()

x = torch.randn(BATCH, 3, IMG, IMG, device=device)
with torch.no_grad():
    with torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
        y = model(x)
print(f"  [3/3] forward con lote de {BATCH} → salida {tuple(y.shape)} ✓")

if torch.cuda.is_available():
    used = torch.cuda.max_memory_allocated() / (1024 ** 3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"\n  Memoria GPU en inferencia: {used:.2f} GB de {total:.0f} GB")
    print(f"  El entrenamiento usará más (gradientes y estados del optimizador),")
    print(f"  pero con {total:.0f} GB hay margen de sobra con lote de 32.")

print("\n  ✅ Todo listo. Puedes lanzar build_dataset_v4.py y train_vit_v5.py")
