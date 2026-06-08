# G23_electra_train.py

import os, csv, time, torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup():
    dist.init_process_group(backend="gloo")

def cleanup():
    dist.destroy_process_group()


class DummyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1024, 4096), nn.ReLU(), nn.Linear(4096, 1024))
    def forward(self, x): return self.net(x)

class DummyDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1024, 4096), nn.ReLU(), nn.Linear(4096, 1))
    def forward(self, x): return self.net(x)


# ── /proc helpers for lightweight system monitoring ──
def read_net_bytes(iface="eth0"):
    """Returns (rx_bytes, tx_bytes) from /proc/net/dev for the given interface."""
    try:
        with open("/proc/net/dev") as f:
            for line in f:
                if iface + ":" in line:
                    parts = line.split()
                    return int(parts[1]), int(parts[9])
    except Exception:
        pass
    return 0, 0

def read_cpu_total():
    """Returns total CPU time (user + nice + system + idle + iowait + irq + softirq)."""
    try:
        with open("/proc/stat") as f:
            parts = f.readline().split()
            return sum(int(x) for x in parts[1:8]), int(parts[4])  # total, idle
    except Exception:
        return 0, 0


NUM_STEPS = 50
CSV_FIELDS = [
    "step", "step_time_sec",
    "forward_time_sec", "backward_time_sec", "optimizer_time_sec",
    "cumulative_sec", "loss",
    "cpu_pct", "net_rx_bytes", "net_tx_bytes",
]


def train():
    setup()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    output_csv = os.environ.get("OUTPUT_CSV", "/app/G23_results.csv")
    print(f"[Pod {rank}] Starting ELECTRA sim. World Size: {world_size}")

    generator, discriminator = DummyGenerator(), DummyDiscriminator()
    ddp_generator = DDP(generator)
    ddp_discriminator = DDP(discriminator)
    optimizer = torch.optim.SGD(
        list(ddp_generator.parameters()) + list(ddp_discriminator.parameters()), lr=0.01)
    loss_fn = nn.MSELoss()

    data = torch.randn(64, 1024)
    labels = torch.randn(64, 1)

    step_records = []
    print(f"[Pod {rank}] Beginning iterations...")
    total_start = time.time()

    for step in range(NUM_STEPS):
        # Snapshot system state before step
        rx0, tx0 = read_net_bytes()
        cpu_total0, cpu_idle0 = read_cpu_total()

        step_start = time.time()
        optimizer.zero_grad()

        fwd_start = time.time()
        gen_out = ddp_generator(data)
        disc_out = ddp_discriminator(gen_out)
        loss = loss_fn(disc_out, labels)
        fwd_end = time.time()

        bwd_start = time.time()
        loss.backward()
        bwd_end = time.time()

        opt_start = time.time()
        optimizer.step()
        opt_end = time.time()

        step_end = time.time()

        # Snapshot after step
        rx1, tx1 = read_net_bytes()
        cpu_total1, cpu_idle1 = read_cpu_total()

        if rank == 0:
            # CPU % = (1 - idle_delta / total_delta) * 100
            total_delta = max(cpu_total1 - cpu_total0, 1)
            idle_delta  = cpu_idle1 - cpu_idle0
            cpu_pct = round((1 - idle_delta / total_delta) * 100, 2)

            step_records.append({
                "step": step,
                "step_time_sec":      round(step_end - step_start, 6),
                "forward_time_sec":   round(fwd_end - fwd_start, 6),
                "backward_time_sec":  round(bwd_end - bwd_start, 6),
                "optimizer_time_sec": round(opt_end - opt_start, 6),
                "cumulative_sec":     round(step_end - total_start, 6),
                "loss":               round(loss.item(), 6),
                "cpu_pct":            cpu_pct,
                "net_rx_bytes":       rx1 - rx0,
                "net_tx_bytes":       tx1 - tx0,
            })
            if step % 10 == 0:
                r = step_records[-1]
                print(f"  Step {step}/{NUM_STEPS} | bwd={r['backward_time_sec']:.3f}s | "
                      f"cpu={r['cpu_pct']}% | tx={r['net_tx_bytes']/1024:.0f}KB")

    total_end = time.time()

    if rank == 0:
        exec_time = total_end - total_start
        total_fwd = sum(r["forward_time_sec"]   for r in step_records)
        total_bwd = sum(r["backward_time_sec"]  for r in step_records)
        total_opt = sum(r["optimizer_time_sec"] for r in step_records)
        total_tx  = sum(r["net_tx_bytes"] for r in step_records)
        total_rx  = sum(r["net_rx_bytes"] for r in step_records)
        avg_cpu   = sum(r["cpu_pct"] for r in step_records) / len(step_records)

        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(step_records)

        print(f"\n========================================")
        print(f"  Simulation Complete!")
        print(f"  Total Execution Time : {exec_time:.2f}s")
        print(f"  ├─ Forward  : {total_fwd:.2f}s ({100*total_fwd/exec_time:.1f}%)")
        print(f"  ├─ Backward : {total_bwd:.2f}s ({100*total_bwd/exec_time:.1f}%)  [DDP sync]")
        print(f"  └─ Optimizer: {total_opt:.2f}s ({100*total_opt/exec_time:.1f}%)")
        print(f"  Avg CPU  : {avg_cpu:.1f}%")
        print(f"  Net TX   : {total_tx/1024/1024:.2f} MB")
        print(f"  Net RX   : {total_rx/1024/1024:.2f} MB")
        print(f"========================================\n")

        print("===CSV_START===")
        print(",".join(CSV_FIELDS))
        for rec in step_records:
            print(",".join(str(rec[f]) for f in CSV_FIELDS))
        print("===CSV_END===")

    cleanup()


if __name__ == "__main__":
    train()
