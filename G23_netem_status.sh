#!/bin/bash
# G23_netem_status.sh

INTERFACE="eth0"
WORKERS=$(docker ps --format '{{.Names}}' | grep '^llm-cluster-worker' | sort)

echo "══ Current tc rules on worker nodes:"
for worker in ${WORKERS}; do
    echo ""
    echo "── ${worker}"
    docker exec "${worker}" tc qdisc show dev "${INTERFACE}"
done
