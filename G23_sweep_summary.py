# G23_sweep_summary.py
#
# Scans all "results_*" folders in the current directory and compiles
# a single master CSV comparing all latency conditions side by side.
#
# Produces: G23_sweep_summary.csv with per-phase aggregated statistics.
#
# Columns:
#   latency_ms, config, num_trials,
#   mean_total_sec, std_total_sec,
#   mean_forward_sec, mean_backward_sec, mean_optimizer_sec,
#   backward_pct, slowdown_vs_affinity_pct
#
# Usage: python G23_sweep_summary.py

import os
import re
import csv
import glob
import numpy as np


def read_trial_csv(filepath):
    """
    Returns dict with total_sec, forward_sum, backward_sum, optimizer_sum.
    Returns None if file is empty/malformed.
    """
    fwd_total = 0.0
    bwd_total = 0.0
    opt_total = 0.0
    last_cum  = 0.0
    row_count = 0

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                last_cum  = float(row["cumulative_sec"])
                fwd_total += float(row.get("forward_time_sec", 0) or 0)
                bwd_total += float(row.get("backward_time_sec", 0) or 0)
                opt_total += float(row.get("optimizer_time_sec", 0) or 0)
                row_count += 1
            except (ValueError, KeyError):
                continue

    if row_count == 0:
        return None

    return {
        "total_sec":     last_cum,
        "forward_sec":   fwd_total,
        "backward_sec":  bwd_total,
        "optimizer_sec": opt_total,
    }


def load_folder(folder):
    """Return (affinity_trials, antiaffinity_trials) lists of trial dicts."""
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


def aggregate(trials):
    """Compute mean/std across trials for each metric."""
    if not trials:
        return None
    totals  = np.array([t["total_sec"]     for t in trials])
    fwds    = np.array([t["forward_sec"]   for t in trials])
    bwds    = np.array([t["backward_sec"]  for t in trials])
    opts    = np.array([t["optimizer_sec"] for t in trials])
    return {
        "num_trials":        len(trials),
        "mean_total_sec":    float(totals.mean()),
        "std_total_sec":     float(totals.std()),
        "mean_forward_sec":  float(fwds.mean()),
        "mean_backward_sec": float(bwds.mean()),
        "mean_optimizer_sec": float(opts.mean()),
    }


def main():
    folders = sorted(glob.glob("results_*ms")) + (["results"] if os.path.isdir("results") else [])

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

        aff_stats  = aggregate(aff)
        anti_stats = aggregate(anti)

        aff_mean  = aff_stats["mean_total_sec"]
        anti_mean = anti_stats["mean_total_sec"]
        slowdown  = ((anti_mean - aff_mean) / aff_mean) * 100 if aff_mean > 0 else 0.0

        def bwd_pct(stats):
            return (stats["mean_backward_sec"] / stats["mean_total_sec"] * 100) if stats["mean_total_sec"] > 0 else 0.0

        rows.append({
            "latency_ms": latency,
            "config": "affinity",
            "num_trials":         aff_stats["num_trials"],
            "mean_total_sec":     round(aff_stats["mean_total_sec"], 4),
            "std_total_sec":      round(aff_stats["std_total_sec"], 4),
            "mean_forward_sec":   round(aff_stats["mean_forward_sec"], 4),
            "mean_backward_sec":  round(aff_stats["mean_backward_sec"], 4),
            "mean_optimizer_sec": round(aff_stats["mean_optimizer_sec"], 4),
            "backward_pct":       round(bwd_pct(aff_stats), 2),
            "slowdown_vs_affinity_pct": 0.0,
        })
        rows.append({
            "latency_ms": latency,
            "config": "antiaffinity",
            "num_trials":         anti_stats["num_trials"],
            "mean_total_sec":     round(anti_stats["mean_total_sec"], 4),
            "std_total_sec":      round(anti_stats["std_total_sec"], 4),
            "mean_forward_sec":   round(anti_stats["mean_forward_sec"], 4),
            "mean_backward_sec":  round(anti_stats["mean_backward_sec"], 4),
            "mean_optimizer_sec": round(anti_stats["mean_optimizer_sec"], 4),
            "backward_pct":       round(bwd_pct(anti_stats), 2),
            "slowdown_vs_affinity_pct": round(slowdown, 2),
        })

    # Write master CSV
    outpath = "G23_sweep_summary.csv"
    fieldnames = [
        "latency_ms", "config", "num_trials",
        "mean_total_sec", "std_total_sec",
        "mean_forward_sec", "mean_backward_sec", "mean_optimizer_sec",
        "backward_pct", "slowdown_vs_affinity_pct",
    ]
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Pretty-print table
    print(f"\n✓ Saved: {outpath}\n")
    print(f"{'Latency':>9} | {'Config':<14} | {'Total':>9} | {'Forward':>9} | {'Backward':>10} | {'Bwd %':>7} | {'Slowdown':>10}")
    print("─" * 88)
    for r in rows:
        latency_str  = f"{r['latency_ms']}ms"
        slowdown_str = f"{r['slowdown_vs_affinity_pct']:+.1f}%" if r['config'] == "antiaffinity" else "—"
        print(f"{latency_str:>9} | {r['config']:<14} | "
              f"{r['mean_total_sec']:>7.2f}s | "
              f"{r['mean_forward_sec']:>7.2f}s | "
              f"{r['mean_backward_sec']:>8.2f}s | "
              f"{r['backward_pct']:>6.1f}% | "
              f"{slowdown_str:>10}")
    print()


if __name__ == "__main__":
    main()
