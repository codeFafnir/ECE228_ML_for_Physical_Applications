# ECE228_ML_for_Physical_Applications

We study compressing a large PINN for mmWave channel estimation that fuses
pilot-based initial estimates with an RSS map. The baseline (358.9M params,
1.4 GB) is unsuitable for edge deployment, so we explore three approaches:
quantization-aware mixed precision (QAT), post-training quantization with
Hadamard rotation (H-GPTQ), and physics-guided knowledge distillation (PG-KD).
On a Boston ray-tracing benchmark, these methods give large memory and speed
gains while retaining or improving NMSE performance.

## Steps to reproduce results

### Dataset Generation and Baseline PINN Training

First, navigate to the `PINN_channel-estimation-main/` folder.

```bash
cd PINN_channel-estimation-main
```

#### Step 1 — Build ground-truth channel tensors from the ray-tracing CSVs

`make_correct_channels.py` parses the Wireless Insite data to produce a `(num_snapshots, D, Nr, Nt)` complex tensor. Run the following command:

```bash
# Boston, 15 GHz, 400 MHz
python make_correct_channels.py \
    --csv Dataset/15GHz_concatenated_data.csv \
    --out 3D_channel_15GHz_2x2_Pt50.npy \
    --pt 50 --bw 4e8
```

#### Step 2 — Generate the initial LS-OFDM channel estimates

For a given `(SNR, Np)` operating point, `init_estimation.py` simulates
OFDM pilot transmission at `N_subcarriers / pilot_spacing` subcarriers,
performs LS interpolation, and saves a `.npy` with the same shape as Step 1. We limit ourselves to Np = 4 for this project, and run this command using 5 different snr values.

```bash
# SNR = 0 dB, Np = 256 (pilot_spacing = 4) -> run for other SNR as well
python init_estimation.py \
    --true-channels 3D_channel_15GHz_2x2_Pt50.npy \
    --output initial_estimate_ls_snr0.npy \
    --snr 0 \
    --n-subcarriers 1024 \
    --pilot-spacing 4
```

#### Step 3 — Train the PINN

Edit the `config` dict at the bottom of [`train.py`](./PINN_channel-estimation-main/train.py) so that
`smomp_file` points at the initial estimate from Step 2 and `accurate_file`
points at the ground-truth tensor from Step 1. Then:

```bash
python train.py
```

This trains for 500 epochs with the hyperparameters, saves the best-validation-NMSE checkpoint to `name_val`, and the
last-epoch checkpoint to `name_train`. The script prints the final test-set
NMSE in both linear and dB scale. Run this command for each different initial estimate file, keeping the same `accurate_file`.

---

## Experiment 1: GPTQ Post-Training Quantization

All scripts live in `experiment1_turboquant/`. Two PTQ variants are implemented:

| Method | Description |
|--------|-------------|
| **GPTQ** | Layer-by-layer second-order quantization using the OBQ formulation. One Hessian in memory at a time (MPS/CPU safe). |
| **Hadamard-GPTQ (H-GPTQ)** | Applies a Hadamard rotation $W \leftarrow WH^\top$ before quantization to redistribute outliers, then absorbs the rotation into the adjacent layer at save time — zero inference overhead. |

Both target **INT-8** and **INT-4** weight precision. Norm layers (GroupNorm, LayerNorm) are skipped. Conv2d inputs are unfolded so that all layer types are treated as linear maps for Hessian accumulation.

### Running GPTQ

```bash
cd experiment1_turboquant

# GPTQ at 8-bit
python quantize_pinn.py \
    --method gptq \
    --bits 8 \
    --checkpoint ../simple_ls_0_val.pth \
    --smomp_file  ../PINN_channel-estimation-main/initial_estimate_ls_snr0.npy \
    --accurate_file ../PINN_channel-estimation-main/3D_channel_15GHz_2x2_Pt50.npy \
    --user_positions ../PINN_channel-estimation-main/ue_positions_noisy.txt \
    --rss_image ../PINN_channel-estimation-main/Dataset/50_15GHz.jpg

# Hadamard-GPTQ at 4-bit
python quantize_pinn.py \
    --method hadamard_gptq \
    --bits 4 \
    --checkpoint ../simple_ls_0_val.pth \
    ...
```

### Evaluating GPTQ

```bash
python eval_all.py \
    --checkpoint ../simple_ls_0_val.pth \
    --smomp_file  ../PINN_channel-estimation-main/initial_estimate_ls_snr0.npy \
    --accurate_file ../PINN_channel-estimation-main/3D_channel_15GHz_2x2_Pt50.npy \
    --user_positions ../PINN_channel-estimation-main/ue_positions_noisy.txt \
    --rss_image ../PINN_channel-estimation-main/Dataset/50_15GHz.jpg \
    --device mps
```

Prints a comparison table: Teacher FP32 vs GPTQ vs H-GPTQ at each bit-width with NMSE (dB) and compression ratio.

---

## Experiment 2: Physics-Guided Knowledge Distillation (PG-KD)

All scripts live in `experiment2_PhysicsInformedKnowledgeDistillation/`. A compact student U-Net is trained to imitate the 358.9M-parameter teacher using a four-term loss:

$$\mathcal{L} = \mathcal{L}_\text{NMSE} + \alpha\,\mathcal{L}_\text{KD}^\text{soft} + \beta\,\mathcal{L}_\text{physics} + \gamma\,\mathcal{L}_\text{xattn}$$

| Term | Role |
|------|------|
| $\mathcal{L}_\text{NMSE}$ | Supervised NMSE vs ground-truth channel |
| $\mathcal{L}_\text{KD}^\text{soft}$ | Temperature-scaled MSE from teacher outputs |
| $\mathcal{L}_\text{physics}$ | RSS power-matching constraint (same as teacher training) |
| $\mathcal{L}_\text{xattn}$ | L2 alignment of student vs teacher cross-attention features |

**Student architecture** replaces every Conv2d with a depthwise-separable block and the 340M FC bottleneck + 5-layer Transformer with a single 8-head cross-attention layer.

Three presets:

| Preset | Params | Compression | Val NMSE |
|--------|--------|-------------|----------|
| light    | 36.0M |  10×  | −19.49 dB |
| moderate | 17.8M |  20×  | −19.48 dB |
| extreme  |  9.3M | 38.7× | −18.76 dB |

*(Evaluated on 790-sample true holdout split, never seen during training.)*

### Step 1 — Precompute teacher cache

Run the frozen teacher once and save outputs + cross-attention features as FP16 memmaps. This only needs to be done once.

```bash
cd experiment2_PhysicsInformedKnowledgeDistillation

python precompute_teacher.py \
    --checkpoint ../simple_ls_0_val.pth \
    --smomp_file  ../PINN_channel-estimation-main/initial_estimate_ls_snr0.npy \
    --accurate_file ../PINN_channel-estimation-main/3D_channel_15GHz_2x2_Pt50.npy \
    --user_positions ../PINN_channel-estimation-main/ue_positions_noisy.txt \
    --rss_image ../PINN_channel-estimation-main/Dataset/50_15GHz.jpg \
    --cache_dir teacher_cache
```

### Step 2 — Train student presets (in order)

```bash
# Light (~36M, 10×)
python train_kd.py --preset light \
    --smomp_file  ../PINN_channel-estimation-main/initial_estimate_ls_snr0.npy \
    --accurate_file ../PINN_channel-estimation-main/3D_channel_15GHz_2x2_Pt50.npy \
    --user_positions ../PINN_channel-estimation-main/ue_positions_noisy.txt \
    --rss_image ../PINN_channel-estimation-main/Dataset/50_15GHz.jpg \
    --cache_dir teacher_cache --epochs 40 --device mps

# Moderate (~18M, 20×)
python train_kd.py --preset moderate  [same flags]

# Extreme (~9.3M, 38.7×)
python train_kd.py --preset extreme   [same flags]
```

Checkpoints are saved to `checkpoints/student_{preset}.pth`. Training history is saved to `checkpoints/training_history_{preset}.json` for plotting.

### Step 3 — Evaluate on validation split

```bash
python eval_kd.py \
    --checkpoint ../simple_ls_0_val.pth \
    --smomp_file  ../PINN_channel-estimation-main/initial_estimate_ls_snr0.npy \
    --accurate_file ../PINN_channel-estimation-main/3D_channel_15GHz_2x2_Pt50.npy \
    --user_positions ../PINN_channel-estimation-main/ue_positions_noisy.txt \
    --rss_image ../PINN_channel-estimation-main/Dataset/50_15GHz.jpg \
    --device mps
```

### Step 4 — Create and evaluate on the true holdout set

```bash
# Create 790-sample holdout bundle (run once locally)
python create_holdout.py --output_dir holdout

# Evaluate on holdout
python eval_holdout.py \
    --holdout_dir holdout \
    --rss_image ../PINN_channel-estimation-main/Dataset/50_15GHz.jpg \
    --checkpoint ../simple_ls_0_val.pth \
    --checkpoint_dir checkpoints \
    --device mps
```

### Running on Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')

BASE = "/content/drive/MyDrive/ECE228"

# Precompute (once)
!python precompute_teacher.py \
    --checkpoint {BASE}/simple_ls_0_val.pth \
    --smomp_file  {BASE}/PINN_channel-estimation-main/initial_estimate_ls_snr0.npy \
    --accurate_file {BASE}/PINN_channel-estimation-main/3D_channel_15GHz_2x2_Pt50.npy \
    --user_positions {BASE}/PINN_channel-estimation-main/ue_positions_noisy.txt \
    --rss_image {BASE}/PINN_channel-estimation-main/Dataset/50_15GHz.jpg \
    --cache_dir {BASE}/teacher_cache

# Train
!python train_kd.py --preset light \
    --smomp_file  {BASE}/PINN_channel-estimation-main/initial_estimate_ls_snr0.npy \
    --accurate_file {BASE}/PINN_channel-estimation-main/3D_channel_15GHz_2x2_Pt50.npy \
    --user_positions {BASE}/PINN_channel-estimation-main/ue_positions_noisy.txt \
    --rss_image {BASE}/PINN_channel-estimation-main/Dataset/50_15GHz.jpg \
    --cache_dir {BASE}/teacher_cache \
    --save_dir {BASE}/checkpoints \
    --epochs 40 --device cuda
```
