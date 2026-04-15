# G23_phase_plot.py
import csv, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif']  = ['Times New Roman']

COLOR_FWD = '#4CAF50'
COLOR_BWD = '#F44336'
COLOR_OPT = '#FFC107'


def load_summary(path="G23_sweep_summary.csv"):
    """Returns dict: {(world_size, latency_ms, config): {forward, backward, optimizer, total}}"""
    data = {}
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            ws  = int(row.get("world_size", 2))
            lat = int(row["latency_ms"])
            cfg = row["config"]
            data[(ws, lat, cfg)] = {
                "total":     float(row["mean_total_sec"]),
                "forward":   float(row["mean_forward_sec"]),
                "backward":  float(row["mean_backward_sec"]),
                "optimizer": float(row["mean_optimizer_sec"]),
            }
    return data


def main():
    try:
        data = load_summary()
    except FileNotFoundError:
        print("ERROR: G23_sweep_summary.csv not found.")
        sys.exit(1)

    world_sizes = sorted({k[0] for k in data.keys()})
    latencies   = sorted({k[1] for k in data.keys()})
    configs     = ["affinity", "antiaffinity"]

    # Build bar layout: for each latency, show all (ws, cfg) bars side-by-side
    fwd_vals, bwd_vals, opt_vals, x_positions, tick_labels = [], [], [], [], []
    group_centers = []
    pos = 0
    for lat in latencies:
        lat_positions = []
        for ws in world_sizes:
            for cfg in configs:
                key = (ws, lat, cfg)
                if key not in data:
                    continue
                d = data[key]
                fwd_vals.append(d["forward"])
                bwd_vals.append(d["backward"])
                opt_vals.append(d["optimizer"])
                x_positions.append(pos)
                lat_positions.append(pos)
                # Bar label: "Aff·W2", "Anti·W2", "Aff·W4", "Anti·W4"
                short_cfg = "Aff" if cfg == "affinity" else "Anti"
                tick_labels.append(f"{short_cfg}·W{ws}")
                pos += 1
        if lat_positions:
            group_centers.append(np.mean(lat_positions))
        pos += 1.2  # gap between latency groups

    fwd_vals, bwd_vals, opt_vals = map(np.array, (fwd_vals, bwd_vals, opt_vals))
    x_positions = np.array(x_positions)

    # Wider figure if 4-pod data is present
    has_w4 = 4 in world_sizes
    fig_width = 16 if has_w4 else 12
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    bar_width = 0.75
    ax.bar(x_positions, fwd_vals, bar_width,
           color=COLOR_FWD, label='Forward', edgecolor='#333', linewidth=0.6)
    ax.bar(x_positions, bwd_vals, bar_width, bottom=fwd_vals,
           color=COLOR_BWD, label='Backward (DDP sync)', edgecolor='#333', linewidth=0.6)
    ax.bar(x_positions, opt_vals, bar_width, bottom=fwd_vals + bwd_vals,
           color=COLOR_OPT, label='Optimizer', edgecolor='#333', linewidth=0.6)

    # Total time labels
    totals = fwd_vals + bwd_vals + opt_vals
    for x, total in zip(x_positions, totals):
        ax.text(x, total + max(totals) * 0.012, f"{total:.1f}s",
                ha='center', va='bottom', fontsize=8, fontweight='bold', color='#333')

    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_labels, fontsize=8, rotation=0)

    # Latency group headers below the bars
    ymax = max(totals) * 1.18
    for lat, center in zip(latencies, group_centers):
        ax.text(center, -0.10 * ymax, f"{lat} ms",
                ha='center', va='top', fontsize=11, fontweight='bold',
                color='#1F4E79')

    # Vertical separators between latency groups
    for i in range(len(group_centers) - 1):
        xmid = (group_centers[i] + group_centers[i+1]) / 2
        ax.axvline(xmid, color='#CCCCCC', linestyle=':', linewidth=1, alpha=0.7)

    ax.set_ylabel('Cumulative Time Across All Steps (Seconds)', fontweight='bold', fontsize=12)
    title = 'G23: Phase Breakdown of Training Time by Latency, World Size & Pod Placement'
    subtitle = 'Backward pass (red) contains the DDP gradient sync — expands with latency and pod count'
    ax.set_title(f"{title}\n{subtitle}", fontweight='bold', fontsize=12)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0, ymax)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    fig.savefig("G23_phase_plot.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: G23_phase_plot.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
