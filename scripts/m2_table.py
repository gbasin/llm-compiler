"""Summarize milestone-2 runs: A retention, B learning, effective core size.

    uv run python scripts/m2_table.py runs/m2cal

Columns:
  A_id/A_ood  mean seq acc over tasks_A at the end (retention)
  B_id        mean seq acc over tasks_B at the end
  B_auc       mean B id acc over all phase-B evals (learning speed)
  core        effective neural core params at end (alive units + fixed)
  A_abl       A_ood with modules ablated (dependence on code)
Runs named <variant>_s<seed> are averaged over seeds.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def row(d: Path):
    cfg = json.loads((d / "config.json").read_text())
    recs = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines()]
    s = json.loads((d / "summary.json").read_text())
    A, B = cfg["tasks_A"].split(","), cfg["tasks_B"].split(",")
    fin = s["final"]
    mean = lambda ts, k, ev=fin: float(np.mean([ev[t][k] for t in ts]))
    phaseB = [r for r in recs if r["phase"] == "B"]
    r = dict(
        A_id=mean(A, "id"), A_ood=mean(A, "ood"), B_id=mean(B, "id"), B_ood=mean(B, "ood"),
        B_auc=float(np.mean([mean(B, "id", r["eval"]) for r in phaseB])) if phaseB else float("nan"),
        core=s["core"]["core_params"], full=s["core"]["full_params"], refits=s.get("refits", 0),
        A_abl=mean(A, "ood", s["ablated"]) if "ablated" in s else float("nan"),
    )
    return r


def main(root):
    groups = defaultdict(list)
    for d in sorted(Path(root).iterdir()):
        if (d / "summary.json").exists():
            groups[re.sub(r"_s\d+$", "", d.name)].append(row(d))
    cols = ["A_id", "A_ood", "B_id", "B_ood", "B_auc", "core", "full", "A_abl", "refits"]
    print(f"{'variant':28s} n " + " ".join(f"{c:>7s}" for c in cols))
    for g, rs in sorted(groups.items()):
        vals = [np.nanmean([r[c] for r in rs]) for c in cols]
        fmt = lambda c, v: f"{v:7.0f}" if c in ("core", "full", "refits") else f"{v:7.3f}"
        print(f"{g:28s} {len(rs)} " + " ".join(fmt(c, v) for c, v in zip(cols, vals)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/m2")
