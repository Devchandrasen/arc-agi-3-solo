"""Generate a thumbnail PNG for the paper track submission page."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(3, 2))
ax.text(0.5, 0.6, "OCA", fontsize=28, ha="center", va="center", fontweight="bold")
ax.text(0.5, 0.35, "Offline Compositional Agents", fontsize=10, ha="center", va="center")
ax.text(0.5, 0.2, "ARC-AGI-3 · 15.85% Phase 1", fontsize=8, ha="center", va="center", style="italic")
ax.axis("off")
fig.savefig("thumbnail.png", dpi=72, bbox_inches="tight")
print("Wrote thumbnail.png")
