#!/bin/bash
# G23_netem_apply.sh
set -e
DELAY_MS=${1:-10}
INTERFACE="eth0"
WORKERS=$(docker ps --format '{{.Names}}' | grep '^llm-cluster-worker' | sort)

echo "══ Applying ${DELAY_MS}ms latency to worker nodes..."
for worker in ${WORKERS}; do
    echo "── ${worker}"
    docker exec "${worker}" tc qdisc del dev "${INTERFACE}" root 2>/dev/null || true
    docker exec "${worker}" tc qdisc add dev "${INTERFACE}" root netem delay "${DELAY_MS}ms"
    echo "   ✓ Applied ${DELAY_MS}ms"
done
echo "══ Done."
