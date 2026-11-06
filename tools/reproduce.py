"""Recompute the frozen evaluation and verify raw-record checksums (no model calls)."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from relay.results import write as write_results
from relay.runner import update_index
from relay.story import build


def reproduce(root, verify=False):
    root = Path(root).resolve()
    sums = root / "experiments" / "SHA256SUMS.json"
    if sums.exists():
        for rel, expected in json.loads(sums.read_text()).items():
            path = root / rel
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise SystemExit(f"Dataset checksum mismatch: {rel}")
    update_index(root / "runs")
    write_results(root / "runs")
    d = build(root / "runs")
    summary = {"release": "0.1.0", "core": d["core"], "counts": d["counts"],
               "scoreboard": [r for r in d["matrix"] if r["level"] == "medium"],
               "matrix": d["matrix"], "cost": d["cost"], "learning": d["learning"], "exclusions": d["exclusions"],
               "gaze": d["gaze"], "failures": d["failures"], "notes": d["notes"]}
    summary = json.loads(json.dumps(summary))  # normalize tuples exactly as stored JSON
    path = root / "experiments" / "evaluation.json"
    if verify:
        if json.loads(path.read_text()) != summary:
            raise SystemExit("Evaluation changed; review and regenerate experiments/evaluation.json")
    else:
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    for r in summary["scoreboard"]:
        print(f"{r['model']:8} {r['k']:2}/{r['n']} parked, seed runs {r['seeds']}")
    print(f"{d['counts']['studies']} valid studies; {d['counts']['excluded_studies']} excluded studies")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    reproduce(args.root, args.verify)
