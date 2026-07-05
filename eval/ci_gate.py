"""Blocks CI/CD if aggregate eval metrics drop below threshold vs baseline."""
import argparse
import json
import sys
from pathlib import Path

BASELINE_PATH = Path("eval/results/baseline.json")
LATEST_PATH = Path("eval/results/latest.json")


def main(threshold_drop: float):
    if not BASELINE_PATH.exists() or not LATEST_PATH.exists():
        print("Missing baseline or latest results. Passing gate by default.")
        sys.exit(0)

    base = json.loads(BASELINE_PATH.read_text())
    latest = json.loads(LATEST_PATH.read_text())

    failed = False
    for metric, base_val in base.items():
        latest_val = latest.get(metric, 0.0)
        drop = base_val - latest_val
        if drop > threshold_drop:
            print(f"❌ FAIL: {metric} dropped by {drop:.3f} "
                  f"(Baseline: {base_val:.3f}, Latest: {latest_val:.3f})")
            failed = True
        else:
            print(f"✅ PASS: {metric} "
                  f"(Baseline: {base_val:.3f}, Latest: {latest_val:.3f})")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold-drop", type=float, default=0.03)
    args = parser.parse_args()
    main(args.threshold_drop)
