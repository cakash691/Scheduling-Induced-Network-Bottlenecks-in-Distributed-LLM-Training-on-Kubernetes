# G23_sweep_plot.py
import csv, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif']  = ['Times New Roman']

# Color/marker scheme: green=affinity, red=anti-affinity. Solid=2-pod, dashed=4-pod
STYLES = {
    (2, "affinity"):     {"color": "#4CAF50", "marker": "o", "linestyle": "-",  "label": "Affinity, 2 pods"},
    (2, "antiaffinity"): {"color": "#F44336", "marker": "s", "linestyle": "-",  "label": "Anti-Affinity, 2 pods"},
    (4, "affinity"):     {"color": "#2E7D32", "marker": "o", "linestyle": "--", "label": "Affinity, 4 pods"},
    (4, "antiaffinity"): {"color": "#B71C1C", "marker": "s", "linestyle": "--", "label": "Anti-Affinity, 4 pods"},
}


def load_summary(path="G23_sweep_summary.csv"):
    """Returns dict: {(world_size, config): {latency: (mean, std)}}"""
    data = {}
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            ws = int(row.get("world_size", 2))
            cfg = row["config"]
            lat = int(row["latency_ms"])
            mean = float(row.get("mean_total_sec") or row.get("mean_sec"))
            std  = float(row.get("std_total_sec")  or row.get("std_sec"))
            data.setdefault((ws, cfg), {})[lat] = (mean, std)
    return data


def main():
    try:
        data = load_summary()
    except FileNotFoundError:
        print("ERROR: G23_sweep_summary.csv not found. Run sweep-summary first.")
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(11, 6.5))

    all_means = []
    all_stds  = []

    for key, points in sorted(data.items()):
        if not points: continue
        style = STYLES.get(key, {"color": "gray", "marker": "x", "linestyle": "-", "label": str(key)})
        latencies = sorted(points.keys())
        means = [points[l][0] for l in latencies]
        stds  = [points[l][1] for l in latencies]
        all_means.extend(means)
        all_stds.extend(stds)

        ax.errorbar(latencies, means, yerr=stds,
                    color=style["color"], linewidth=2.2, marker=style["marker"], markersize=8,
                    linestyle=style["linestyle"], capsize=5, capthick=1.2,
                    label=style["label"])

        # Annotate the rightmost (highest-latency) point of each line
        if means:
            ax.annotate(f"{means[-1]:.1f}s", xy=(latencies[-1], means[-1]),
                        xytext=(8, 0), textcoords='offset points',
                        fontsize=9, fontweight='bold', color=style["color"], va='center')

    ax.set_xlabel('Injected Network Latency (ms)', fontweight='bold', fontsize=13)
    ax.set_ylabel('Execution Time (Seconds)', fontweight='bold', fontsize=13)

    has_w4 = any(ws == 4 for (ws, _) in data.keys())
    title = 'G23: Distributed ELECTRA Training Time vs. Injected Network Latency'
    if has_w4:
        title += '\n2-pod vs 4-pod DDP comparison (5 trials per condition, error bars = ±1σ)'
    else:
        title += '\n(5 trials per condition, error bars = ±1σ)'
    ax.set_title(title, fontweight='bold', fontsize=12)

    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    # X-axis: union of all latencies seen
    all_lats = sorted({l for points in data.values() for l in points.keys()})
    ax.set_xticks(all_lats)

    if all_means:
        ymax = max(all_means) + max(all_stds) + 12
        ax.set_ylim(0, ymax)

    fig.tight_layout()
    fig.savefig("G23_sweep_plot.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: G23_sweep_plot.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
