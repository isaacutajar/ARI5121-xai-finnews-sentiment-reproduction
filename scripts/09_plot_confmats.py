# scripts/09_plot_confmats.py
"""
Plot global confusion matrices for models evaluated on gold annotations.

Reads:
  outputs/gold_eval/confmats/{MODEL}_confmat_global.json

Writes:
  outputs/figures/confmat_{MODEL}.png
Run: python scripts/09_plot_confmats.py
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from _config import OUTPUTS_DIR

CONF_DIR = OUTPUTS_DIR / "gold_eval" / "confmats"
FIG_DIR  = OUTPUTS_DIR / "figures"

def _plot_cm(cm, labels, title, out_path: Path):
    cm = np.array(cm)
    fig = plt.figure(figsize=(4.2, 3.6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    thresh = cm.max() / 2.0 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)

def main():
    if not CONF_DIR.exists():
        print(f"[ERROR] Missing directory: {CONF_DIR}")
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(CONF_DIR.glob("*_confmat_global.json"))
    if not files:
        print("[INFO] No confusion matrix files found.")
        return
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        labels = d.get("labels", [0,1,2])
        cm = d.get("matrix", [[0,0,0],[0,0,0],[0,0,0]])
        model = f.stem.replace("_confmat_global","")
        out = FIG_DIR / f"confmat_{model}.png"
        _plot_cm(cm, labels, f"Confusion Matrix — {model}", out)
        print(f"[OK] {out}")
    print("[DONE] Plotted confusion matrices.")

if __name__ == "__main__":
    main()
