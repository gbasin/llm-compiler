"""Plot eval curves and print a summary table for a directory of runs.

    uv run python scripts/plot.py runs/m1

Runs named <variant>_s<seed> are grouped by variant (mean, min-max band).
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(d: Path):
    recs = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines()]
    summ = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else None
    return recs, summ


def main(root):
    root = Path(root)
    groups = defaultdict(list)
    for d in sorted(root.iterdir()):
        if (d / "metrics.jsonl").exists():
            groups[re.sub(r"_s\d+$", "", d.name)].append(load(d))
    if not groups:
        sys.exit(f"no runs in {root}")
    splits = list(next(iter(groups.values()))[0][0][0]["eval"].keys())
    fig, axes = plt.subplots(2, len(splits), figsize=(5 * len(splits), 8), squeeze=False)
    for gi, (g, runs) in enumerate(sorted(groups.items())):
        color = f"C{gi}"
        n = min(len(r[0]) for r in runs)
        steps = [r["step"] for r in runs[0][0][:n]]
        for si, sp in enumerate(splits):
            for row, key in enumerate(["seq_acc", "loss"]):
                y = np.array([[r["eval"][sp][key] for r in rs[:n]] for rs, _ in runs])
                ax = axes[row, si]
                ax.plot(steps, y.mean(0), color=color, label=f"{g} (n={len(runs)})")
                ax.fill_between(steps, y.min(0), y.max(0), color=color, alpha=0.15)
                ax.set_title(f"{sp} {key}")
                if key == "loss":
                    ax.set_yscale("log")
    axes[0, 0].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(root / "curves.png", dpi=110)

    print(f"{'variant':24s} " + " ".join(f"{sp:>14s}" for sp in splits) + "   gate   ablated(" + splits[-1] + ")")
    for g, runs in sorted(groups.items()):
        fin = {sp: np.mean([s["final"][sp]["seq_acc"] for _, s in runs if s]) for sp in splits}
        gates = [i["gate"] for _, s in runs if s for i in s["injections"]]
        abl = [s["ablated"][splits[-1]]["seq_acc"] for _, s in runs if s and "ablated" in s]
        line = f"{g:24s} " + " ".join(f"{fin[sp]:14.3f}" for sp in splits)
        line += f"   {np.mean(gates):5.2f}" if gates else "      -"
        line += f"   {np.mean(abl):.3f}" if abl else ""
        print(line)
    print(f"plot: {root / 'curves.png'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs")
