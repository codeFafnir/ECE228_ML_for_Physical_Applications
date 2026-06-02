#!/usr/bin/env bash
# End-to-end PG-KD pipeline.
# All commands run from experiment2_PhysicsInformedKnowledgeDistillation/.
#
# Training order (least → most compressed):
#   light (8-10x, ~36M) → moderate (15-20x, ~18M) → extreme (35-40x, ~9M)
#
# Edit DATA_* and CKPT variables for your local paths.
# Pass --device cpu to force CPU on all steps if MPS is unstable.

set -euo pipefail
cd "$(dirname "$0")"

# ── Paths ─────────────────────────────────────────────────────────────────────
CKPT="../simple_ls_0_val.pth"
SMOMP="../PINN_channel-estimation-main/initial_estimate_ls_snr0.npy"
ACCURATE="../PINN_channel-estimation-main/3D_channel_15GHz_2x2_Pt50.npy"
POSITIONS="../PINN_channel-estimation-main/ue_positions_noisy.txt"
RSS_IMG="../PINN_channel-estimation-main/Dataset/50_15GHz.jpg"
CACHE_DIR="teacher_cache"
CKPT_DIR="checkpoints"
DEVICE="${DEVICE:-auto}"   # override: DEVICE=cpu ./run.sh

# ── Step 1: Precompute teacher cache (run once) ───────────────────────────────
echo "=== Step 1: Precomputing teacher cache ==="
python precompute_teacher.py \
    --checkpoint   "$CKPT"      \
    --smomp_file   "$SMOMP"     \
    --accurate_file "$ACCURATE" \
    --user_positions "$POSITIONS" \
    --rss_image    "$RSS_IMG"   \
    --cache_dir    "$CACHE_DIR" \
    --device       "$DEVICE"    \
    --batch_size   8            \
    --splits train val

# ── Step 2: Train light student (8-10x) ───────────────────────────────────────
echo ""
echo "=== Step 2: Training light student (8-10x, ~36M params) ==="
python train_kd.py \
    --preset       light        \
    --smomp_file   "$SMOMP"     \
    --accurate_file "$ACCURATE" \
    --user_positions "$POSITIONS" \
    --rss_image    "$RSS_IMG"   \
    --cache_dir    "$CACHE_DIR" \
    --save_dir     "$CKPT_DIR"  \
    --device       "$DEVICE"    \
    --epochs       40           \
    --lr           3e-4         \
    --batch_size   16           \
    --kd_mode      mse          \
    --alpha        1.0          \
    --beta         0.01         \
    --gamma        0.1          \
    --T            4.0

# ── Step 3: Train moderate student (15-20x) ───────────────────────────────────
echo ""
echo "=== Step 3: Training moderate student (15-20x, ~18M params) ==="
python train_kd.py \
    --preset       moderate     \
    --smomp_file   "$SMOMP"     \
    --accurate_file "$ACCURATE" \
    --user_positions "$POSITIONS" \
    --rss_image    "$RSS_IMG"   \
    --cache_dir    "$CACHE_DIR" \
    --save_dir     "$CKPT_DIR"  \
    --device       "$DEVICE"    \
    --epochs       40           \
    --lr           3e-4         \
    --batch_size   16           \
    --kd_mode      mse          \
    --alpha        1.0          \
    --beta         0.01         \
    --gamma        0.1          \
    --T            4.0

# ── Step 4: Train extreme student (35-40x) ────────────────────────────────────
echo ""
echo "=== Step 4: Training extreme student (35-40x, ~9M params) ==="
python train_kd.py \
    --preset       extreme      \
    --smomp_file   "$SMOMP"     \
    --accurate_file "$ACCURATE" \
    --user_positions "$POSITIONS" \
    --rss_image    "$RSS_IMG"   \
    --cache_dir    "$CACHE_DIR" \
    --save_dir     "$CKPT_DIR"  \
    --device       "$DEVICE"    \
    --epochs       40           \
    --lr           3e-4         \
    --batch_size   16           \
    --kd_mode      mse          \
    --alpha        1.0          \
    --beta         0.01         \
    --gamma        0.1          \
    --T            4.0

# ── Step 5: Final evaluation table ────────────────────────────────────────────
echo ""
echo "=== Step 5: Evaluation (Teacher FP32 vs all students) ==="
python eval_kd.py \
    --checkpoint   "$CKPT"      \
    --smomp_file   "$SMOMP"     \
    --accurate_file "$ACCURATE" \
    --user_positions "$POSITIONS" \
    --rss_image    "$RSS_IMG"   \
    --checkpoint_dir "$CKPT_DIR" \
    --device       "$DEVICE"    \
    --batch_size   16
