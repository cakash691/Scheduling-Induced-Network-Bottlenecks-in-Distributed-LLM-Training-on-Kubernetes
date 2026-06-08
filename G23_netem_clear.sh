#!/bin/bash
# G23_netem_clear.sh

set -e
INTERFACE="eth0"
WORKERS=$(docker ps --format '{{.Names}}' | grep '^llm-cluster-worker' | sort)

echo "══ Clearing tc rules from worker nodes..."
for worker in ${WORKERS}; do
    echo "── ${worker}"
    docker exec "${worker}" tc qdisc del dev "${INTERFACE}" root 2>/dev/null || echo "   (no rules)"
    echo "   ✓ Cleared"
done
echo "══ Done."
