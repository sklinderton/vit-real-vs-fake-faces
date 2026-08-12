#!/bin/bash
#SBATCH --job-name=vit_train
#SBATCH --partition=nukwa-l40s
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:50:00
#SBATCH --output=/data/ulead-04/proyecto_paralela/logs/vit_train_%j.log
#SBATCH --error=/data/ulead-04/proyecto_paralela/logs/vit_train_%j.err

# ══════════════════════════════════════════════════════════════════════════════
# submit_train_vit.sh — Lanzar entrenamiento del ViT como trabajo SBATCH
# Corre desatendido, no depende de que la sesión interactiva siga abierta.
#
# Uso:
#   sbatch submit_train_vit.sh                  # entrenamiento nuevo
#   sbatch submit_train_vit.sh --resume          # continuar el último checkpoint
#
# Verificar estado:
#   squeue -u ulead-04
#   tail -f /data/ulead-04/proyecto_paralela/logs/vit_train_<JOBID>.log
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

PROJECT_ROOT="/data/ulead-04/proyecto_paralela"
mkdir -p "$PROJECT_ROOT/logs"

echo "=============================================="
echo "  Entrenamiento ViT — Job SLURM"
echo "  Fecha: $(date)"
echo "  Nodo:  $(hostname)"
echo "  JobID: $SLURM_JOB_ID"
echo "=============================================="

# ── Activar entorno ────────────────────────────────────────────────────────
source /data/ulead-04/envs/vit_faces/bin/activate
echo "[OK] Entorno activado: $(which python)"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

cd "$PROJECT_ROOT"

# ── Ejecutar entrenamiento ────────────────────────────────────────────────
# --max-minutes=220 deja margen de seguridad bajo el límite de 230 min (3:50h)
python train_vit.py \
    --epochs 15 \
    --batch-size 32 \
    --lr 1e-4 \
    --num-workers 16 \
    --max-minutes 220 \
    "$@"

echo ""
echo "=============================================="
echo "  Job finalizado — $(date)"
echo "=============================================="

# ── Commit automático de resultados ──────────────────────────────────────
git -C "$PROJECT_ROOT" add results/ checkpoints/vit_best.pt logs/ 2>/dev/null || true
git -C "$PROJECT_ROOT" commit -m "results: entrenamiento ViT job $SLURM_JOB_ID - $(date +%Y-%m-%d)" 2>/dev/null \
    && echo "[OK] Commit realizado" \
    || echo "[INFO] Sin cambios para commitear o repo no inicializado aún"
