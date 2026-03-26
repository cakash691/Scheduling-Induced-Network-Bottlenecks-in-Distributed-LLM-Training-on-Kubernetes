# G23_electra_train.py
import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 1. Setup Distributed Environment
def setup():
    # 'gloo' is the standard backend for CPU-based distributed training
    dist.init_process_group(backend="gloo")

def cleanup():
    dist.destroy_process_group()

# 2. Simulate the ELECTRA Architecture (Dual-Network)
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

def train():
    setup()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])
    
    print(f"[Pod {rank}] Starting ELECTRA distributed simulation. World Size: {world_size}")

    # Initialize models
    generator = DummyGenerator()
    discriminator = DummyDiscriminator()

    # Wrap models in DDP to trigger network synchronization during backward passes
    ddp_generator = DDP(generator)
    ddp_discriminator = DDP(discriminator)

    optimizer = torch.optim.SGD(list(ddp_generator.parameters()) + list(ddp_discriminator.parameters()), lr=0.01)
    loss_fn = nn.MSELoss()

    # Create dummy data
    data = torch.randn(64, 1024)
    labels = torch.randn(64, 1)

    print(f"[Pod {rank}] Beginning synchronization iterations...")
    start_time = time.time()

    # Simulate 50 training steps
    for step in range(50):
        optimizer.zero_grad()
        
        # Forward pass
        gen_out = ddp_generator(data)
        disc_out = ddp_discriminator(gen_out)
        
        # Loss and backward pass (This is where the network synchronization actually happens!)
        loss = loss_fn(disc_out, labels)
        loss.backward()
        
        optimizer.step()
        
        if step % 10 == 0 and rank == 0:
            print(f"Step {step}/50 completed.")

    end_time = time.time()
    
    if rank == 0:
        execution_time = end_time - start_time
        print(f"\n========================================")
        print(f"Simulation Complete! Total Execution Time: {execution_time:.2f} seconds")
        print(f"========================================\n")

    cleanup()

if __name__ == "__main__":
    train()
