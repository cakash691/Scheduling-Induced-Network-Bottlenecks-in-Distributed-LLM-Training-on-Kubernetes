# G23_sweep_plot.py
#
# Reads G23_sweep_summary.csv and generates a publication-quality line chart
# showing how execution time scales with injected network latency, for both
# affinity and anti-affinity configurations.
#
# Produces:
#   - G23_sweep_plot.png : main line chart with error bars
#
# Usage: python G23_sweep_plot.py

import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif']  = ['Times New Roman']

COLOR_AFFINITY     = '#4CAF50'
COLOR_ANTIAFFINITY = '#F44336'


def load_summary(path="G23_sweep_summary.csv"):
    """Returns dict: {latency_ms: {affinity: (mean, std), antiaffinity: (mean, std)}}"""
    data = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat = int(row["latency_ms"])
            cfg = row["config"]
            # Support both old schema (mean_sec/std_sec) and new schema (mean_total_sec/std_total_sec)
            mean = float(row.get("mean_total_sec") or row.get("mean_sec"))
            std  = float(row.get("std_total_sec")  or row.get("std_sec"))
            data.setdefault(lat, {})[cfg] = (mean, std)
    return data


def main():
    try:
        data = load_summary()
    except FileNotFoundError:
        print("ERROR: G23_sweep_summary.csv not found.")
        print("Run: make -f G23_Makefile sweep-summary")
        sys.exit(1)

    latencies = sorted(data.keys())

    aff_means  = [data[l]["affinity"][0]     for l in latencies]
    aff_stds   = [data[l]["affinity"][1]     for l in latencies]
    anti_means = [data[l]["antiaffinity"][0] for l in latencies]
    anti_stds  = [data[l]["antiaffinity"][1] for l in latencies]

    # ── Main figure ──
    fig, ax = plt.subplots(figsize=(10, 6))

    # Affinity line
    ax.errorbar(latencies, aff_means, yerr=aff_stds,
                color=COLOR_AFFINITY, linewidth=2.5, marker='o', markersize=9,
                markerfacecolor=COLOR_AFFINITY, markeredgecolor='#2E7D32',
                capsize=6, capthick=1.5,
                label='Affinity (Same Node)')

    # Anti-Affinity line
    ax.errorbar(latencies, anti_means, yerr=anti_stds,
                color=COLOR_ANTIAFFINITY, linewidth=2.5, marker='s', markersize=9,
                markerfacecolor=COLOR_ANTIAFFINITY, markeredgecolor='#B71C1C',
                capsize=6, capthick=1.5,
                label='Anti-Affinity (Different Nodes)')

    # Value annotations above anti-affinity points
    for x, y in zip(latencies, anti_means):
        ax.annotate(f"{y:.1f}s", xy=(x, y), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=10,
                    color='#B71C1C', fontweight='bold')

    # Value annotations below affinity points
    for x, y in zip(latencies, aff_means):
        ax.annotate(f"{y:.1f}s", xy=(x, y), xytext=(0, -16),
                    textcoords='offset points', ha='center', fontsize=10,
                    color='#2E7D32', fontweight='bold')

    ax.set_xlabel('Injected Network Latency (ms)', fontweight='bold', fontsize=13)
    ax.set_ylabel('Execution Time (Seconds)', fontweight='bold', fontsize=13)
    ax.set_title('G23: Distributed ELECTRA Training Time vs. Injected Network Latency\n'
                 '(5 trials per condition, error bars show ±1σ)',
                 fontweight='bold', fontsize=13)

    ax.legend(loc='upper left', fontsize=12, framealpha=0.95)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_xticks(latencies)

    # Small buffer on the y-axis
    ymax = max(anti_means) + max(anti_stds) + 8
    ax.set_ylim(0, ymax)

    fig.tight_layout()
    fig.savefig("G23_sweep_plot.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: G23_sweep_plot.png")
    plt.close(fig)

    # ── Print summary ──
    print(f"\n{'Latency':>10} | {'Affinity':>15} | {'Anti-Affinity':>16} | {'Slowdown':>10}")
    print("─" * 62)
    for l, a, aa in zip(latencies, aff_means, anti_means):
        slowdown = ((aa - a) / a) * 100 if a > 0 else 0.0
        print(f"{l:>7} ms | {a:>13.2f} s | {aa:>14.2f} s | {slowdown:>+9.1f}%")
    print()


if __name__ == "__main__":
    main()
