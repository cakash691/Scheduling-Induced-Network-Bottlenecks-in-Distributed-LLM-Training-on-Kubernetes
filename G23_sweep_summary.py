# G23_sweep_summary.py
#
# Scans all "results_*" folders in the current directory and compiles
# a single master CSV comparing all latency conditions side by side.
#
# Produces: G23_sweep_summary.csv with columns:
#   latency_ms, config, num_trials, mean_sec, std_sec, min_sec, max_sec, slowdown_vs_affinity_pct
#
# Also prints a readable table to stdout.
#
# Usage: python G23_sweep_summary.py

import os
import re
import csv
import glob
import numpy as np


def read_trial_csv(filepath):
    totals = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        last_cum = 0.0
        for row in reader:
            try:
                last_cum = float(row["cumulative_sec"])
            except (ValueError, KeyError):
                continue
        totals.append(last_cum)
    return totals[0] if totals else None


def load_folder(folder):
    """Return (affinity_totals, antiaffinity_totals) lists for a given results folder."""
    aff, anti = [], []
    for f in sorted(glob.glob(os.path.join(folder, "G23_results_affinity_run*.csv"))):
        t = read_trial_csv(f)
        if t: aff.append(t)
    for f in sorted(glob.glob(os.path.join(folder, "G23_results_antiaffinity_run*.csv"))):
        t = read_trial_csv(f)
        if t: anti.append(t)
    return aff, anti


def extract_latency(folder_name):
    """Pull the latency number out of a folder name like 'results_25ms' → 25."""
    base = os.path.basename(folder_name.rstrip("/"))
    if base == "results":
        return 0
    m = re.search(r"results_(\d+)ms", base)
    if m:
        return int(m.group(1))
    return None


def main():
    folders = sorted(glob.glob("results_*ms")) + (["results"] if os.path.isdir("results") else [])

    # De-duplicate and sort by latency
    seen = set()
    entries = []
    for folder in folders:
        latency = extract_latency(folder)
        if latency is None or folder in seen:
            continue
        seen.add(folder)
        entries.append((latency, folder))
    entries.sort()

    if not entries:
        print("No results_*ms or results/ folders found. Run experiments first.")
        return

    rows = []
    for latency, folder in entries:
        aff, anti = load_folder(folder)
        if not aff or not anti:
            print(f"  Skipping {folder} (incomplete data)")
            continue

        aff_mean = float(np.mean(aff))
        anti_mean = float(np.mean(anti))
        slowdown = ((anti_mean - aff_mean) / aff_mean) * 100 if aff_mean > 0 else 0.0

        rows.append({
            "latency_ms": latency,
            "config": "affinity",
            "num_trials": len(aff),
            "mean_sec": round(aff_mean, 4),
            "std_sec": round(float(np.std(aff)), 4),
            "min_sec": round(float(np.min(aff)), 4),
            "max_sec": round(float(np.max(aff)), 4),
            "slowdown_vs_affinity_pct": 0.0,
        })
        rows.append({
            "latency_ms": latency,
            "config": "antiaffinity",
            "num_trials": len(anti),
            "mean_sec": round(anti_mean, 4),
            "std_sec": round(float(np.std(anti)), 4),
            "min_sec": round(float(np.min(anti)), 4),
            "max_sec": round(float(np.max(anti)), 4),
            "slowdown_vs_affinity_pct": round(slowdown, 2),
        })

    # Write master CSV
    outpath = "G23_sweep_summary.csv"
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "latency_ms", "config", "num_trials", "mean_sec", "std_sec",
            "min_sec", "max_sec", "slowdown_vs_affinity_pct"
        ])
        writer.writeheader()
        writer.writerows(rows)

    # Pretty-print table to stdout
    print(f"\n✓ Saved: {outpath}\n")
    print(f"{'Latency':>9} | {'Config':<14} | {'Mean (s)':>10} | {'Std (s)':>9} | {'Slowdown':>10}")
    print("─" * 66)
    for r in rows:
        latency_str = f"{r['latency_ms']}ms"
        slowdown_str = f"{r['slowdown_vs_affinity_pct']:+.1f}%" if r['config'] == "antiaffinity" else "—"
        print(f"{latency_str:>9} | {r['config']:<14} | {r['mean_sec']:>10.2f} | {r['std_sec']:>9.3f} | {slowdown_str:>10}")
    print()


if __name__ == "__main__":
    main()
