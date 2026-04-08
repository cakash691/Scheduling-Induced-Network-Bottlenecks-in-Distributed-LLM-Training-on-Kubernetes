#!/bin/bash
# G23_netem_status.sh
#
# Shows the current tc qdisc configuration on both Kind worker nodes.
# Usage: ./G23_netem_status.sh

INTERFACE="eth0"
WORKERS=("llm-cluster-worker" "llm-cluster-worker2")

echo "══ Current tc rules on worker nodes:"

for worker in "${WORKERS[@]}"; do
    echo ""
    echo "── ${worker}"
    docker exec "${worker}" tc qdisc show dev "${INTERFACE}"
done
