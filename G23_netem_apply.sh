#!/bin/bash
# G23_netem_apply.sh
#
# Applies artificial network latency to both Kind worker nodes using tc netem.
# Usage: ./G23_netem_apply.sh <delay_ms>
# Example: ./G23_netem_apply.sh 10

set -e

DELAY_MS=${1:-10}
INTERFACE="eth0"
WORKERS=("llm-cluster-worker" "llm-cluster-worker2")

echo "══ Applying ${DELAY_MS}ms latency to worker nodes on ${INTERFACE}..."

for worker in "${WORKERS[@]}"; do
    echo "── ${worker}"

    # Remove any existing qdisc first (ignore errors if none exists)
    docker exec "${worker}" tc qdisc del dev "${INTERFACE}" root 2>/dev/null || true

    # Apply new netem rule with specified delay
    docker exec "${worker}" tc qdisc add dev "${INTERFACE}" root netem delay "${DELAY_MS}ms"

    echo "   ✓ Applied ${DELAY_MS}ms delay"
done

echo "══ Done. Verify with: ./G23_netem_status.sh"
