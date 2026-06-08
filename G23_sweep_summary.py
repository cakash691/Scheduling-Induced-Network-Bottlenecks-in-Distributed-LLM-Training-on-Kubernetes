# G23_sweep_summary.py

import os, re, csv, glob
import numpy as np


def read_trial_csv(filepath):
    fwd = bwd = opt = last_cum = 0.0
    rx = tx = 0
    cpu_vals = []
    rows_seen = 0
    with open(filepath, "r") as f:
        for row in csv.DictReader(f):
            try:
                last_cum = float(row["cumulative_sec"])
                fwd += float(row.get("forward_time_sec", 0) or 0)
                bwd += float(row.get("backward_time_sec", 0) or 0)
                opt += float(row.get("optimizer_time_sec", 0) or 0)
                rx  += int(row.get("net_rx_bytes", 0) or 0)
                tx  += int(row.get("net_tx_bytes", 0) or 0)
                if row.get("cpu_pct"):
                    cpu_vals.append(float(row["cpu_pct"]))
                rows_seen += 1
            except (ValueError, KeyError):
                continue
    if rows_seen == 0:
        return None
    return {
        "total_sec": last_cum, "forward_sec": fwd,
        "backward_sec": bwd, "optimizer_sec": opt,
        "net_rx_mb": rx / 1024 / 1024, "net_tx_mb": tx / 1024 / 1024,
        "avg_cpu_pct": sum(cpu_vals) / len(cpu_vals) if cpu_vals else 0.0,
    }


def load_folder(folder):
    aff, anti = [], []
    for f in sorted(glob.glob(os.path.join(folder, "G23_results_affinity_run*.csv"))):
        t = read_trial_csv(f)
        if t: aff.append(t)
    for f in sorted(glob.glob(os.path.join(folder, "G23_results_antiaffinity_run*.csv"))):
        t = read_trial_csv(f)
        if t: anti.append(t)
    return aff, anti


def parse_folder(folder_name):
    """Returns (world_size, latency_ms) from folder name. Defaults world_size=2 for legacy folders."""
    base = os.path.basename(folder_name.rstrip("/"))
    # results_w4_25ms → (4, 25), results_25ms → (2, 25), results → (2, 0)
    m = re.match(r"results(?:_w(\d+))?(?:_(\d+)ms)?$", base)
    if not m:
        return None
    world_size = int(m.group(1)) if m.group(1) else 2
    latency    = int(m.group(2)) if m.group(2) else 0
    return world_size, latency


def aggregate(trials):
    if not trials: return None
    totals = np.array([t["total_sec"] for t in trials])
    return {
        "num_trials": len(trials),
        "mean_total_sec":     float(totals.mean()),
        "std_total_sec":      float(totals.std()),
        "mean_forward_sec":   float(np.mean([t["forward_sec"]   for t in trials])),
        "mean_backward_sec":  float(np.mean([t["backward_sec"]  for t in trials])),
        "mean_optimizer_sec": float(np.mean([t["optimizer_sec"] for t in trials])),
        "mean_net_tx_mb":     float(np.mean([t["net_tx_mb"]     for t in trials])),
        "mean_net_rx_mb":     float(np.mean([t["net_rx_mb"]     for t in trials])),
        "mean_cpu_pct":       float(np.mean([t["avg_cpu_pct"]   for t in trials])),
    }


def main():
    all_folders = sorted(glob.glob("results*"))
    entries = []
    seen = set()
    for folder in all_folders:
        if not os.path.isdir(folder) or folder in seen:
            continue
        parsed = parse_folder(folder)
        if parsed is None:
            continue
        seen.add(folder)
        ws, lat = parsed
        entries.append((ws, lat, folder))
    entries.sort()

    if not entries:
        print("No results folders found.")
        return

    rows = []
    for ws, lat, folder in entries:
        aff, anti = load_folder(folder)
        if not aff or not anti:
            print(f"  Skipping {folder} (incomplete data)")
            continue

        a, b = aggregate(aff), aggregate(anti)
        slowdown = ((b["mean_total_sec"] - a["mean_total_sec"]) / a["mean_total_sec"]) * 100 if a["mean_total_sec"] > 0 else 0.0

        for cfg, stats, sd in [("affinity", a, 0.0), ("antiaffinity", b, slowdown)]:
            bp = (stats["mean_backward_sec"] / stats["mean_total_sec"] * 100) if stats["mean_total_sec"] > 0 else 0.0
            rows.append({
                "world_size": ws, "latency_ms": lat, "config": cfg,
                "num_trials":         stats["num_trials"],
                "mean_total_sec":     round(stats["mean_total_sec"], 4),
                "std_total_sec":      round(stats["std_total_sec"], 4),
                "mean_forward_sec":   round(stats["mean_forward_sec"], 4),
                "mean_backward_sec":  round(stats["mean_backward_sec"], 4),
                "mean_optimizer_sec": round(stats["mean_optimizer_sec"], 4),
                "mean_net_tx_mb":     round(stats["mean_net_tx_mb"], 4),
                "mean_net_rx_mb":     round(stats["mean_net_rx_mb"], 4),
                "mean_cpu_pct":       round(stats["mean_cpu_pct"], 2),
                "backward_pct":       round(bp, 2),
                "slowdown_vs_affinity_pct": round(sd, 2),
            })

    fieldnames = ["world_size", "latency_ms", "config", "num_trials",
                  "mean_total_sec", "std_total_sec",
                  "mean_forward_sec", "mean_backward_sec", "mean_optimizer_sec",
                  "mean_net_tx_mb", "mean_net_rx_mb", "mean_cpu_pct",
                  "backward_pct", "slowdown_vs_affinity_pct"]
    with open("G23_sweep_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\n✓ Saved: G23_sweep_summary.csv\n")
    print(f"{'WS':>3} | {'Lat':>5} | {'Config':<14} | {'Total':>8} | {'Backward':>9} | {'Bwd%':>6} | {'TX MB':>7} | {'CPU%':>6} | {'Slowdown':>10}")
    print("─" * 96)
    for r in rows:
        sd = f"{r['slowdown_vs_affinity_pct']:+.1f}%" if r['config'] == "antiaffinity" else "—"
        print(f"{r['world_size']:>3} | {r['latency_ms']:>3}ms | {r['config']:<14} | "
              f"{r['mean_total_sec']:>6.2f}s | {r['mean_backward_sec']:>7.2f}s | "
              f"{r['backward_pct']:>5.1f}% | {r['mean_net_tx_mb']:>6.2f} | "
              f"{r['mean_cpu_pct']:>5.1f}% | {sd:>10}")
    print()


if __name__ == "__main__":
    main()
