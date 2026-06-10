# ECE228_ML_for_Physical_Applications

## Dataset Generation and Baseline PINN Training

First, navigate to the `PINN_channel-estimation-main/` folder.

### Step 1 — Build ground-truth channel tensors from the ray-tracing CSVs

`make_correct_channels.py` parses the Wireless Insite data to produce a `(num_snapshots, D, Nr, Nt)` complex tensor. Run the following command:

```bash
# Boston, 15 GHz, 400 MHz
python make_correct_channels.py \
    --csv Dataset/15GHz_concatenated_data.csv \
    --out 3D_channel_15GHz_2x2_Pt50.npy \
    --pt 50 --bw 4e8
```

### Step 2 — Generate the initial LS-OFDM channel estimates

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

### Step 3 — Train the PINN

Edit the `config` dict at the bottom of [`train.py`](./PINN_channel-estimation-main/train.py) so that
`smomp_file` points at the initial estimate from Step 2 and `accurate_file`
points at the ground-truth tensor from Step 1. Then:

```bash
python train.py
```

This trains for 500 epochs with the hyperparameters, saves the best-validation-NMSE checkpoint to `name_val`, and the
last-epoch checkpoint to `name_train`. The script prints the final test-set
NMSE in both linear and dB scale. Run this command for each different initial estimate file, keeping the same `accurate_file`.
