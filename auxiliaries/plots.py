"""Plot every run in results/runs.jsonl -> results/plots/runs.png

    panel 1  float32 vs int8 test accuracy, per run
    panel 2  accuracy drop per run, against the 1% budget
    panel 3  training curves per run (solid = validation, dotted = train)

Run from the repo root:  python plot_runs.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

RUNS = Path("results") / "runs.jsonl"
OUT = Path("results") / "plots" / "runs.png"
BUDGET = 1.0  # percentage points

_FLOAT = "#257de0"
_INT8 = "#f47230"
_OK = "#2a9d4b"
_OVER = "#d1342f"
_INK = "#191919"
_MUTED = "#666666"
_GRID = "#e7e7e7"


def loadRuns(path=RUNS):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def style(ax, title, xlabel):
    ax.set_title(title, color=_INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=_MUTED, fontsize=9)
    ax.grid(axis="y", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c8c8c8")
    ax.tick_params(colors=_MUTED, labelsize=8)


def plotRuns(runs, outPath=OUT):
    labels = [r["timestamp"][5:16].replace("T", "\n") for r in runs]  # MM-DD / HH:MM
    floatAcc = np.array([r["float32"]["test_accuracy"] for r in runs]) * 100
    int8Acc = np.array([r["int8"]["test_accuracy"] for r in runs]) * 100
    drops = floatAcc - int8Acc
    x = np.arange(len(runs))

    fig, (axAcc, axDrop, axCurve) = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)

    # 1. float32 vs int8
    w = 0.38
    axAcc.bar(x - w / 2, floatAcc, w, color=_FLOAT, label="float32")
    axAcc.bar(x + w / 2, int8Acc, w, color=_INT8, label="int8")
    low = min(floatAcc.min(), int8Acc.min())
    axAcc.set_ylim(max(0, low - 3), max(floatAcc.max(), int8Acc.max()) + 1.5)
    axAcc.set_xticks(x, labels)
    axAcc.set_ylabel("Test accuracy (%)", color=_MUTED, fontsize=9)
    axAcc.legend(frameon=False, fontsize=9, loc="upper right")
    style(axAcc, "float32 vs int8", "Run")

    # 2. drop vs budget
    axDrop.bar(x, drops, 0.5, color=[_OK if d <= BUDGET else _OVER for d in drops])
    axDrop.axhline(BUDGET, color=_MUTED, linestyle="--", linewidth=1.2)
    axDrop.annotate(f"{BUDGET:g}% budget", (x[-1] + 0.3, BUDGET), xytext=(0, 4),
                    textcoords="offset points", ha="right", fontsize=8, color=_MUTED)
    for xi, d in zip(x, drops):
        axDrop.annotate(f"{d:.2f}", (xi, max(d, 0)), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8, color=_INK)
    axDrop.set_ylim(min(0, drops.min() - 0.1), max(BUDGET, drops.max()) * 1.25)
    axDrop.set_xticks(x, labels)
    axDrop.set_ylabel("Accuracy drop (pts)", color=_MUTED, fontsize=9)
    style(axDrop, "int8 accuracy drop", "Run")

    # 3. training curves
    for r, label in zip(runs, labels):
        h = r["history"]
        epochs = np.arange(1, len(h["val_accuracy"]) + 1)
        line, = axCurve.plot(epochs, np.array(h["val_accuracy"]) * 100, marker="o", linewidth=2,
                             label=label.replace("\n", " "))
        axCurve.plot(epochs, np.array(h["accuracy"]) * 100, linestyle=":", linewidth=1.5,
                     color=line.get_color())
    axCurve.xaxis.set_major_locator(MaxNLocator(integer=True))
    axCurve.set_ylabel("Accuracy (%)", color=_MUTED, fontsize=9)
    axCurve.legend(frameon=False, fontsize=8, title="solid = val, dotted = train", title_fontsize=8)
    style(axCurve, "Training curves (float32)", "Epoch")

    outPath = Path(outPath)
    outPath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outPath, dpi=200, facecolor="white")
    plt.close(fig)
    return outPath


if __name__ == "__main__":
    print(f"saved {plotRuns(loadRuns())}")
