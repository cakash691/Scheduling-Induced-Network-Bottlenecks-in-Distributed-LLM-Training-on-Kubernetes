# G23_electra_train.py
import os
import csv
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# ──────────────────────────────────────────────
# 1. Distributed Environment Setup / Teardown
# ──────────────────────────────────────────────
def setup():
    # 'gloo' is the standard backend for CPU-based distributed training
    dist.init_process_group(backend="gloo")

def cleanup():
    dist.destroy_process_group()


# ──────────────────────────────────────────────
# 2. Simulate the ELECTRA Architecture (Dual-Network)
# ──────────────────────────────────────────────
class DummyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # Creating a deliberately large linear layer to simulate heavy gradient synchronization
        self.net = nn.Sequential(nn.Linear(1024, 4096), nn.ReLU(), nn.Linear(4096, 1024))

    def forward(self, x):
        return self.net(x)

class DummyDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1024, 4096), nn.ReLU(), nn.Linear(4096, 1))

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────
# 3. Training Loop with Phase-Level Instrumentation
# ──────────────────────────────────────────────
NUM_STEPS = 50

# CSV schema — keep this in one place so downstream scripts stay in sync
CSV_FIELDS = [
    "step",
    "step_time_sec",       # total wall-clock for this step
    "forward_time_sec",    # time spent in forward pass
    "backward_time_sec",   # time spent in backward pass (includes DDP gradient sync)
    "optimizer_time_sec",  # time spent in optimizer.step()
    "cumulative_sec",      # total elapsed time from start of training
    "loss",
]


def train():
    setup()
    rank       = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])

    # OUTPUT_CSV can be overridden by the Makefile / YAML env vars.
    output_csv = os.environ.get("OUTPUT_CSV", "/app/G23_results.csv")

    print(f"[Pod {rank}] Starting ELECTRA distributed simulation. World Size: {world_size}")

    # Initialize models
    generator     = DummyGenerator()
    discriminator = DummyDiscriminator()

    # Wrap models in DDP to trigger network synchronization during backward passes
    ddp_generator     = DDP(generator)
    ddp_discriminator = DDP(discriminator)

    optimizer = torch.optim.SGD(
        list(ddp_generator.parameters()) + list(ddp_discriminator.parameters()),
        lr=0.01
    )
    loss_fn = nn.MSELoss()

    # Create dummy data
    data   = torch.randn(64, 1024)
    labels = torch.randn(64, 1)

    # ── Per-step timing storage ──
    step_records = []

    print(f"[Pod {rank}] Beginning synchronization iterations...")
    total_start = time.time()

    # Simulate training steps
    for step in range(NUM_STEPS):
        step_start = time.time()

        optimizer.zero_grad()

        # ── PHASE 1: Forward pass ──
        fwd_start = time.time()
        gen_out  = ddp_generator(data)
        disc_out = ddp_discriminator(gen_out)
        loss = loss_fn(disc_out, labels)
        fwd_end = time.time()

        # ── PHASE 2: Backward pass (DDP gradient sync happens here!) ──
        bwd_start = time.time()
        loss.backward()
        bwd_end = time.time()

        # ── PHASE 3: Optimizer step ──
        opt_start = time.time()
        optimizer.step()
        opt_end = time.time()

        step_end = time.time()

        # Only rank 0 records metrics (workers don't write CSVs)
        if rank == 0:
            step_time   = step_end - step_start
            fwd_time    = fwd_end - fwd_start
            bwd_time    = bwd_end - bwd_start
            opt_time    = opt_end - opt_start
            cumulative  = step_end - total_start

            step_records.append({
                "step":                step,
                "step_time_sec":       round(step_time, 6),
                "forward_time_sec":    round(fwd_time, 6),
                "backward_time_sec":   round(bwd_time, 6),
                "optimizer_time_sec":  round(opt_time, 6),
                "cumulative_sec":      round(cumulative, 6),
                "loss":                round(loss.item(), 6),
            })

            if step % 10 == 0:
                print(f"  Step {step}/{NUM_STEPS}  |  fwd={fwd_time:.4f}s  bwd={bwd_time:.4f}s  opt={opt_time:.4f}s  total={step_time:.4f}s")

    total_end = time.time()

    # ── Write results (rank 0 only) ──
    if rank == 0:
        execution_time = total_end - total_start

        # Aggregate phase totals for the final summary
        total_fwd = sum(r["forward_time_sec"]   for r in step_records)
        total_bwd = sum(r["backward_time_sec"]  for r in step_records)
        total_opt = sum(r["optimizer_time_sec"] for r in step_records)

        # Write per-step CSV
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(step_records)

        print(f"\n========================================")
        print(f"  Simulation Complete!")
        print(f"  Total Execution Time : {execution_time:.2f} seconds")
        print(f"  ├─ Forward total     : {total_fwd:.2f} seconds ({100*total_fwd/execution_time:.1f}%)")
        print(f"  ├─ Backward total    : {total_bwd:.2f} seconds ({100*total_bwd/execution_time:.1f}%)  [includes DDP sync]")
        print(f"  └─ Optimizer total   : {total_opt:.2f} seconds ({100*total_opt/execution_time:.1f}%)")
        print(f"  Per-step CSV saved to: {output_csv}")
        print(f"  Steps recorded       : {len(step_records)}")
        print(f"========================================\n")

        # Dump CSV to stdout between markers so Makefile can extract it from logs
        # (kubectl cp doesn't work on completed pods)
        print("===CSV_START===")
        print(",".join(CSV_FIELDS))
        for rec in step_records:
            print(",".join(str(rec[f]) for f in CSV_FIELDS))
        print("===CSV_END===")

    cleanup()


if __name__ == "__main__":
    train()
