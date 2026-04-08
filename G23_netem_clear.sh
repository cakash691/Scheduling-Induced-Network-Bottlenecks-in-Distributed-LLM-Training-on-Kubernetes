#!/bin/bash
# G23_netem_clear.sh
#
# Removes all tc netem rules from both Kind worker nodes, restoring normal networking.
# Usage: ./G23_netem_clear.sh

set -e

INTERFACE="eth0"
WORKERS=("llm-cluster-worker" "llm-cluster-worker2")

echo "══ Clearing tc rules from worker nodes..."

for worker in "${WORKERS[@]}"; do
    echo "── ${worker}"
    docker exec "${worker}" tc qdisc del dev "${INTERFACE}" root 2>/dev/null || echo "   (no rules to clear)"
    echo "   ✓ Cleared"
done

echo "══ Done. Networking restored to normal."
