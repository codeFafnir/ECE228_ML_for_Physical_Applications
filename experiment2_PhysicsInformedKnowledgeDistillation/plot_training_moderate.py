import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

epochs = list(range(1, 50))  # epoch 50 was incomplete

train_loss = [
    6993.3514, 4.6123, 1.0798, 0.6114, 0.2028,
    0.7691, 0.5027, 0.3207, 0.1907, 0.8398,
    0.5969, 0.5106, 0.1764, 0.4143, 0.2956,
    0.2353, 0.4763, 0.9696, 0.1130, 0.1071,
    0.1426, 1.3937, 0.2817, 0.0923, 0.2076,
    0.4350, 0.2017, 0.1582, 0.1523, 0.0842,
    0.7890, 0.0947, 0.0940, 0.0683, 0.3600,
    0.0605, 0.0572, 0.0447, 0.0511, 0.0466,
    0.0506, 0.0782, 0.0694, 0.0375, 0.0316,
    0.0345, 0.0311, 0.0343, 0.0331,
]

train_nmse = [
    6993.3376, 4.5979, 1.0654, 0.5970, 0.1884,
    0.7546, 0.4883, 0.3062, 0.1762, 0.8253,
    0.5824, 0.4961, 0.1619, 0.3998, 0.2811,
    0.2208, 0.4618, 0.9551, 0.0985, 0.0926,
    0.1281, 1.3792, 0.2672, 0.0778, 0.1931,
    0.4205, 0.1872, 0.1438, 0.1378, 0.0697,
    0.7745, 0.0802, 0.0795, 0.0538, 0.3455,
    0.0460, 0.0427, 0.0302, 0.0365, 0.0321,
    0.0361, 0.0637, 0.0549, 0.0230, 0.0171,
    # approx for 46-49: loss - phys*beta (phys≈1.4514, beta=0.01)
    0.0200, 0.0166, 0.0198, 0.0186,
]

val_nmse_db = [
    -3.93, -8.66, -11.39, -12.53, -12.48,
    -13.27, -14.20, -14.73, -14.75, -14.49,
    -15.23, -14.78, -15.41, -15.89, -15.35,
    -16.08, -15.96, -16.45, -16.38, -16.18,
    -16.60, -16.58, -16.80, -16.52, -16.87,
    -17.23, -17.45, -17.50, -17.00, -17.57,
    -17.94, -17.98, -17.94, -18.13, -18.15,
    -18.20, -18.27, -18.43, -18.50, -18.50,
    -18.49, -18.56, -18.67, -18.71, -18.72,
    -18.74, -18.75, -18.76, -18.76,
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
    7.67e-6, 5.63e-6, 4.17e-6, 3.29e-6,
]

best_epoch = val_nmse_db.index(min(val_nmse_db)) + 1
best_val = min(val_nmse_db)

fig = plt.figure(figsize=(12, 9))
fig.suptitle(
    "PG-KD Training  |  preset=moderate  |  17.83M params (20.1× compression)",
    fontsize=13, fontweight="bold", y=0.98,
)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(epochs, train_loss, color="#4C72B0", linewidth=1.5, label="Train Loss")
ax0.set_yscale("log")
ax0.set_xlabel("Epoch")
ax0.set_ylabel("Loss (log scale)")
ax0.set_title("Training Loss")
ax0.grid(True, which="both", linestyle="--", alpha=0.4)
ax0.legend(fontsize=9)

ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(epochs, train_nmse, color="#DD8452", linewidth=1.5, label="Train NMSE")
ax1.set_yscale("log")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("NMSE (log scale)")
ax1.set_title("Training NMSE")
ax1.grid(True, which="both", linestyle="--", alpha=0.4)
ax1.legend(fontsize=9)

ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(epochs, val_nmse_db, color="#55A868", linewidth=1.8, label="Val NMSE")
ax2.scatter([best_epoch], [best_val], color="red", zorder=5, s=60,
            label=f"Best: {best_val:.2f} dB (ep {best_epoch})")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Val NMSE (dB)")
ax2.set_title("Validation NMSE")
ax2.grid(True, linestyle="--", alpha=0.4)
ax2.legend(fontsize=9)

ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(epochs, lr, color="#8172B2", linewidth=1.5, label="LR")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Learning Rate")
ax3.set_title("Learning Rate Schedule")
ax3.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax3.grid(True, linestyle="--", alpha=0.4)
ax3.legend(fontsize=9)

out_path = "training_curves_moderate.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")
