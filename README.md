# Kubernetes-Aware LLM Training Simulation

**Course:** Graduate Systems (CSE638) - IIIT Delhi  
**Team Members:** Akash Chakraborty

## Overview

This repository contains the Phase 1 (Part A) implementation for simulating and analyzing the system-level performance of Distributed Large Language Model (LLM) training under varying Kubernetes scheduling policies.

Specifically, we simulate the distributed pre-training of the ELECTRA architecture. ELECTRA's dual-network setup (Generator and Discriminator) creates heavy inter-container synchronization traffic. By wrapping the workload in PyTorch Distributed Data Parallel (DDP) and utilizing a local `kind` cluster, we measure the network latency and straggler effects caused by scheduling communication-heavy pods across different physical nodes versus co-locating them on the same node.

## Repository Structure

Strict file naming conventions (`G23_`) are enforced per assignment guidelines.

- `G23_Dockerfile`: Lightweight Python 3.10 container definition tailored for CPU-based PyTorch DDP. (Includes `numpy` for clean PyTorch execution).
- `G23_Makefile`: Automation script for building the environment, executing experiments (single-run and multi-trial), and cleaning up cluster resources.
- `G23_electra_train.py`: The distributed PyTorch workload simulating 50 synchronization steps of the ELECTRA generator/discriminator layers. Includes per-step CSV logging with stdout dump for data extraction.
- `G23_k8s_affinity.yaml`: Kubernetes manifest forcing the Master and Worker pods onto the **same** node using `podAffinity` (simulating a fast, local network).
- `G23_k8s_antiaffinity.yaml`: Kubernetes manifest forcing the Master and Worker pods onto **different** nodes using `podAntiAffinity` (simulating a slow, cross-network topology).
- `G23_plot.py`: Matplotlib script that reads per-step CSVs from `results/` directory, computes mean ± stddev across trials, and generates bar charts with error bars and per-step latency line charts with confidence bands.
- `G23_plot_bar.png`: Generated bar chart comparing total execution times.
- `G23_plot_perstep.png`: Generated per-step latency comparison line chart.
- `G23_results_summary.csv`: Aggregated statistics (mean, stddev, min, max) for both configurations.
- `G23_report.pdf`: The detailed project report.

## Prerequisites

To run this simulation locally, you need:

1. **Docker** (or OrbStack for Apple Silicon)
2. **kind** (Kubernetes IN Docker) to simulate a multi-node cluster — install via `brew install kind`
3. **kubectl** (Kubernetes command-line tool)
4. **Make** (Build automation tool)
5. **Python 3** with `numpy` and `matplotlib` (for generating plots)

Verify all prerequisites:

```bash
docker --version
kind --version
kubectl version --client
make --version
```

If any of these fail, install the missing one before proceeding.

## How to Run the Experiments

### 1. Create the Kind Cluster

Create a `kind-config.yaml` file and spin up the cluster:

```bash
cat <<EOF > kind-config.yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
  - role: worker
  - role: worker
EOF

kind create cluster --config kind-config.yaml --name llm-cluster
```

Verify all 3 nodes are ready:

```bash
kubectl get nodes
```

You should see one control-plane and two worker nodes, all in `Ready` status.

### 2. Build and Load the Container

Build the PyTorch environment and push it into the simulated cluster:

```bash
make -f G23_Makefile build
```

Wait until you see "Build and load complete!"

### 3. Run All Trials (Recommended)

Run 5 affinity + 5 anti-affinity experiments back to back:

```bash
make -f G23_Makefile run-all-trials
```

This takes roughly 5-10 minutes depending on your machine. Each trial deploys pods, waits for training to complete, extracts the CSV from logs, cleans up, then repeats. All CSVs are saved to the `results/` directory.

### 4. Quick Single Runs (Optional)

For quick testing, you can run one experiment at a time:

```bash
make -f G23_Makefile run-affinity
make -f G23_Makefile run-antiaffinity
```

### 5. Generate the Plots

After running experiments, generate all charts and the summary CSV:

```bash
make -f G23_Makefile plot
```

This produces `G23_plot_bar.png`, `G23_plot_perstep.png`, and `G23_results_summary.csv`.

## Cleanup

### Emergency Pod Cleanup

If pods get stuck or you need to force-wipe the experiment deployments:

```bash
make -f G23_Makefile clean
```

### Full Teardown

To completely remove everything:

```bash
# 1. Delete any running/completed pods
make -f G23_Makefile clean

# 2. Delete the entire Kind cluster (removes all nodes, pods, services)
kind delete cluster --name llm-cluster

# 3. Remove the Docker image you built
docker rmi electra-sim:v1

# 4. (Optional) Prune all unused Docker data — dangling images, build cache, etc.
docker system prune -a
```
