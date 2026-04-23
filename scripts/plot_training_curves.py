import json, sys
from pathlib import Path
import matplotlib.pyplot as plt

runs = Path("experiments/runs")
fig, (ax_loss, ax_rmspe) = plt.subplots(1, 2, figsize=(11, 4))

for model_dir in sorted(runs.iterdir()):
    hist_path = model_dir / "checkpoints" / "history.json"
    if not hist_path.exists():
        continue
    h = json.loads(hist_path.read_text())
    epochs = [r["epoch"] for r in h]
    ax_loss.plot(epochs, [r["train"] for r in h], label=f"{model_dir.name} train")
    ax_loss.plot(epochs, [r["val"]   for r in h], ls="--", label=f"{model_dir.name} val")
    ax_rmspe.plot(epochs, [r["val_rmspe_deg"] for r in h], label=model_dir.name)

ax_loss.set(xlabel="epoch", ylabel="loss", yscale="log"); ax_loss.legend(); ax_loss.grid(alpha=.3)
ax_rmspe.set(xlabel="epoch", ylabel="val RMSPE [deg]");   ax_rmspe.legend(); ax_rmspe.grid(alpha=.3)
fig.tight_layout()
fig.savefig("experiments/runs/training_curves.png", dpi=180)
print("wrote experiments/runs/training_curves.png")