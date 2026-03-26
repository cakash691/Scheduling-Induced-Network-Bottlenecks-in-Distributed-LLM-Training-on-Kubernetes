# Project Part A: Kubernetes-Aware LLM Training Simulation

**Course:** Graduate Systems (CSE638) - IIIT Delhi  
**Group ID:** 23  
**Team Members:** Debarshi Roy, Akash Chakraborty, Shubham Gaur  

## Overview
This repository contains the Phase 1 (Part A) implementation for simulating and analyzing the system-level performance of Distributed Large Language Model (LLM) training under varying Kubernetes scheduling policies. 

Specifically, we simulate the distributed pre-training of the ELECTRA architecture. ELECTRA's dual-network setup (Generator and Discriminator) creates heavy inter-container synchronization traffic. By wrapping the workload in PyTorch Distributed Data Parallel (DDP) and utilizing a local `kind` cluster, we measure the network latency and straggler effects caused by scheduling communication-heavy pods across different physical nodes versus co-locating them on the same node.

## Repository Structure
Strict file naming conventions (`G23_`) are enforced per assignment guidelines.

* `G23_Dockerfile`: Lightweight Python 3.10 container definition tailored for CPU-based PyTorch DDP. (Includes `numpy` for clean PyTorch execution).
* `G23_Makefile`: Automation script for building the environment, executing experiments, and cleaning up cluster resources.
* `G23_electra_train.py`: The distributed PyTorch workload simulating 50 synchronization steps of the ELECTRA generator/discriminator layers.
* `G23_k8s_affinity.yaml`: Kubernetes manifest forcing the Master and Worker pods onto the **same** node using `podAffinity` (simulating a fast, local network).
* `G23_k8s_antiaffinity.yaml`: Kubernetes manifest forcing the Master and Worker pods onto **different** nodes using `podAntiAffinity` (simulating a slow, cross-network topology).
* `G23_plot.py`: Hardcoded Matplotlib script to visualize the performance bottleneck.
* `G23_results.csv`: Raw execution time data.
* `G23_plot.png`: The generated performance comparison bar chart.
* `G23_report.pdf`: The detailed project report.

## Prerequisites
To run this simulation locally, you need:
1. **Docker** (or OrbStack for Apple Silicon)
2. **kind** (Kubernetes IN Docker) to simulate a multi-node cluster
3. **kubectl** (Kubernetes command-line tool)
4. **Make** (Build automation tool)

## How to Run the Experiments (Automated Workflow)

### 1. Initialize the Cluster
Create a 3-node local cluster (1 Control Plane, 2 Workers):
```bash
# Create a kind-config.yaml file with 1 control-plane and 2 worker nodes, then run:
kind create cluster --config kind-config.yaml --name llm-cluster

## 2. Build and Load the Container

Build the PyTorch environment and push it into the simulated cluster using the Makefile:

make -f G23_Makefile build

## 3. Run Experiment 1: Same-Node (Affinity)

Deploy the pods, fetch the logs, and automatically clean up the cluster afterward:

make -f G23_Makefile run-affinity

Note the total execution time printed at the end of the simulation.

## 4. Run Experiment 2: Cross-Node (Anti-Affinity)

Deploy the cross-node pods, fetch the logs, and automatically clean up:

make -f G23_Makefile run-antiaffinity

Note the total execution time printed at the end of the simulation.

## 5. Generate the Plot

Update G23_plot.py and G23_results.csv with your newly observed execution times, then generate the updated Matplotlib chart:

make -f G23_Makefile plot

## 6. Emergency Cleanup

If pods get stuck or you need to force-wipe the experiment deployments from the cluster:

make -f G23_Makefile clean

