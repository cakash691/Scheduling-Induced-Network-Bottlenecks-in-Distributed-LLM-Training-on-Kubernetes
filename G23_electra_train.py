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
# 3. Training Loop with Per-Step Instrumentation
# ──────────────────────────────────────────────
NUM_STEPS = 50

def train():
    setup()
    rank       = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])

    # OUTPUT_CSV can be overridden by the Makefile / YAML env vars.
    # Default: write to /app/G23_results.csv inside the container.
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
    step_records = []           # list of dicts to write to CSV

    print(f"[Pod {rank}] Beginning synchronization iterations...")
    total_start = time.time()

    # Simulate training steps
    for step in range(NUM_STEPS):
        step_start = time.time()

        optimizer.zero_grad()

        # Forward pass
        gen_out  = ddp_generator(data)
        disc_out = ddp_discriminator(gen_out)

        # Loss and backward pass (this is where DDP gradient sync happens!)
        loss = loss_fn(disc_out, labels)
        loss.backward()

        optimizer.step()

        step_end = time.time()

        # Only rank 0 records metrics (workers don't write CSVs)
        if rank == 0:
            step_time    = step_end - step_start
            cumulative   = step_end - total_start

            step_records.append({
                "step":            step,
                "step_time_sec":   round(step_time, 6),
                "cumulative_sec":  round(cumulative, 6),
                "loss":            round(loss.item(), 6),
            })

            if step % 10 == 0:
                print(f"  Step {step}/{NUM_STEPS}  |  step_time={step_time:.4f}s  |  cumulative={cumulative:.4f}s")

    total_end = time.time()

    # ── Write results (rank 0 only) ──
    if rank == 0:
        execution_time = total_end - total_start

        # Write per-step CSV
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "step_time_sec", "cumulative_sec", "loss"])
            writer.writeheader()
            writer.writerows(step_records)

        print(f"\n========================================")
        print(f"  Simulation Complete!")
        print(f"  Total Execution Time : {execution_time:.2f} seconds")
        print(f"  Per-step CSV saved to: {output_csv}")
        print(f"  Steps recorded       : {len(step_records)}")
        print(f"========================================\n")

        # Dump CSV to stdout between markers so Makefile can extract it from logs
        # (kubectl cp doesn't work on completed pods)
        print("===CSV_START===")
        print("step,step_time_sec,cumulative_sec,loss")
        for rec in step_records:
            print(f"{rec['step']},{rec['step_time_sec']},{rec['cumulative_sec']},{rec['loss']}")
        print("===CSV_END===")

    cleanup()


if __name__ == "__main__":
    train()