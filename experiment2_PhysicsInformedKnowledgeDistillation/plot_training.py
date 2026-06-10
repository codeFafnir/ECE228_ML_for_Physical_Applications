import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

epochs = list(range(1, 51))

train_loss = [
    7.5958, 2.5908, 0.9429, 0.8642, 1.1965,
    1.0311, 0.3602, 0.5680, 0.3122, 0.3049,
    0.1261, 0.2682, 1.9948, 0.3961, 0.1881,
    0.3179, 0.2913, 0.5678, 0.0809, 0.0853,
    0.1167, 0.2251, 0.2147, 0.1683, 1.0102,
    0.0657, 0.1833, 0.0679, 0.2266, 0.0522,
    0.0800, 0.0698, 0.0568, 0.0443, 0.0682,
    0.0506, 0.0512, 0.0988, 0.0398, 0.0420,
    0.0335, 0.1095, 0.0313, 0.0348, 0.0319,
    0.0341, 0.0304, 0.0296, 0.0313, 0.0288,
]

train_nmse = [
    7.5822, 2.5765, 0.9285, 0.8498, 1.1820,
    1.0166, 0.3458, 0.5535, 0.2977, 0.2905,
    0.1117, 0.2537, 1.9804, 0.3816, 0.1736,
    0.3034, 0.2768, 0.5533, 0.0664, 0.0709,
    0.1022, 0.2106, 0.2002, 0.1538, 0.9957,
    0.0512, 0.1688, 0.0534, 0.2121, 0.0377,
    0.0655, 0.0553, 0.0423, 0.0298, 0.0537,
    0.0361, 0.0367, 0.0843, 0.0252, 0.0275,
    0.0190, 0.0949, 0.0168, 0.0203, 0.0174,
    0.0196, 0.0158, 0.0151, 0.0168, 0.0143,
]

val_nmse_db = [
    -5.26, -9.31, -11.27, -12.82, -13.62,
    -14.17, -14.52, -12.73, -14.14, -14.91,
    -14.35, -15.36, -15.77, -16.09, -16.11,
    -16.29, -16.24, -15.37, -16.38, -16.40,
    -16.62, -17.02, -17.49, -17.30, -17.63,
    -17.83, -17.81, -17.55, -17.20, -18.00,
    -17.88, -18.38, -18.43, -18.61, -18.64,
    -18.77, -19.03, -19.15, -19.03, -19.24,
    -19.24, -19.35, -19.30, -19.41, -19.44,
    -19.45, -19.47, -19.48, -19.49, -19.49,
]

lr = [
    3.00e-4, 2.99e-4, 2.97e-4, 2.95e-4, 2.93e-4,
    2.90e-4, 2.86e-4, 2.82e-4, 2.77e-4, 2.72e-4,
    2.66e-4, 2.60e-4, 2.53e-4, 2.46e-4, 2.39e-4,
    2.31e-4, 2.23e-4, 2.15e-4, 2.06e-4, 1.97e-4,
    1.88e-4, 1.79e-4, 1.70e-4, 1.61e-4, 1.51e-4,
    1.42e-4, 1.33e-4, 1.24e-4, 1.15e-4, 1.06e-4,
    9.68e-5, 8.83e-5, 8.00e-5, 7.19e-5, 6.42e-5,
    5.68e-5, 4.98e-5, 4.32e-5, 3.71e-5, 3.14e-5,
    2.61e-5, 2.14e-5, 1.71e-5, 1.34e-5, 1.03e-5,
    7.67e-6, 5.63e-6, 4.17e-6, 3.29e-6, 3.00e-6,
]

best_epoch = val_nmse_db.index(min(val_nmse_db)) + 1
best_val = min(val_nmse_db)

fig = plt.figure(figsize=(12, 9))
fig.suptitle(
    "PG-KD Training  |  preset=light  |  35.97M params (10× compression)",
    fontsize=13, fontweight="bold", y=0.98,
)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

# ── Train Loss ──────────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(epochs, train_loss, color="#4C72B0", linewidth=1.5, label="Train Loss")
ax0.set_yscale("log")
ax0.set_xlabel("Epoch")
ax0.set_ylabel("Loss (log scale)")
ax0.set_title("Training Loss")
ax0.grid(True, which="both", linestyle="--", alpha=0.4)
ax0.legend(fontsize=9)

# ── Train NMSE ──────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(epochs, train_nmse, color="#DD8452", linewidth=1.5, label="Train NMSE")
ax1.set_yscale("log")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("NMSE (log scale)")
ax1.set_title("Training NMSE")
ax1.grid(True, which="both", linestyle="--", alpha=0.4)
ax1.legend(fontsize=9)

# ── Val NMSE (dB) ────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(epochs, val_nmse_db, color="#55A868", linewidth=1.8, label="Val NMSE")
ax2.scatter([best_epoch], [best_val], color="red", zorder=5, s=60,
            label=f"Best: {best_val:.2f} dB (ep {best_epoch})")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Val NMSE (dB)")
ax2.set_title("Validation NMSE")
ax2.grid(True, linestyle="--", alpha=0.4)
ax2.legend(fontsize=9)

# ── Learning Rate ────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(epochs, lr, color="#8172B2", linewidth=1.5, label="LR")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Learning Rate")
ax3.set_title("Learning Rate Schedule")
ax3.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax3.grid(True, linestyle="--", alpha=0.4)
ax3.legend(fontsize=9)

out_path = "training_curves_light.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")
