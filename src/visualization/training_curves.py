import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.style import polish_axes, set_thesis_style

# =========================================================
# LOAD LOG FILE
# =========================================================

log_path = "experiments/mobilenetv2/224/run_1/training_log.csv"

df = pd.read_csv(log_path)
set_thesis_style()
os.makedirs("plots/training_curves", exist_ok=True)

# =========================================================
# ACCURACY CURVE
# =========================================================

fig, ax = plt.subplots(figsize=(9, 5.5))

ax.plot(df["accuracy"], label="Training", color="#2F80ED", linewidth=2.4)

ax.plot(df["val_accuracy"], label="Validation", color="#EB5757", linewidth=2.4)

polish_axes(ax, "Epoch", "Accuracy", "Training and Validation Accuracy")
ax.set_ylim(0, 1.02)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("plots/training_curves/mobilenetv2_224_run1_accuracy.png", dpi=300, bbox_inches="tight")

plt.show()

# =========================================================
# LOSS CURVE
# =========================================================

fig, ax = plt.subplots(figsize=(9, 5.5))

ax.plot(df["loss"], label="Training", color="#2F80ED", linewidth=2.4)

ax.plot(df["val_loss"], label="Validation", color="#EB5757", linewidth=2.4)

polish_axes(ax, "Epoch", "Loss", "Training and Validation Loss")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("plots/training_curves/mobilenetv2_224_run1_loss.png", dpi=300, bbox_inches="tight")

plt.show()
