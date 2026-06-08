#!/bin/bash
# G23_sweep_run_w4.sh

# 4-pod version of the latency sweep. Same structure as G23_sweep_run.sh
# but targets the _w4 YAMLs and saves to results_w4_<delay>ms/

set -e
LATENCIES=${LATENCIES:-"0 1 5 10 25"}
NUM_TRIALS=${NUM_TRIALS:-5}

echo "══════════════════════════════════════════"
echo "  G23 Latency Sweep (4-worker DDP)"
echo "  Latencies: ${LATENCIES} ms"
echo "  Trials   : ${NUM_TRIALS} per config per latency"
echo "══════════════════════════════════════════"

SWEEP_START=$(date +%s)

for delay in ${LATENCIES}; do
    echo ""
    echo "══ LATENCY = ${delay}ms ══"
    RESULTS_DIR="results_w4_${delay}ms"
    mkdir -p "${RESULTS_DIR}"

    if [ "${delay}" -eq 0 ]; then
        bash G23_netem_clear.sh
    else
        bash G23_netem_apply.sh "${delay}"
    fi

    # Affinity trials (all 4 pods same node)
    for i in $(seq 1 "${NUM_TRIALS}"); do
        echo "── [${delay}ms W4] Affinity Trial ${i}/${NUM_TRIALS}"
        kubectl apply -f G23_k8s_affinity_w4.yaml
        kubectl wait --for=jsonpath='{.status.phase}'=Succeeded pod/electra-master --timeout=900s
        kubectl logs electra-master | sed -n '/===CSV_START===/,/===CSV_END===/p' | grep -v '===CSV' > "${RESULTS_DIR}/G23_results_affinity_run${i}.csv"
        kubectl delete -f G23_k8s_affinity_w4.yaml
        TOTAL=$(tail -1 "${RESULTS_DIR}/G23_results_affinity_run${i}.csv" | cut -d',' -f6)
        echo "   ✓ Total: ${TOTAL}s"
        sleep 2
    done

    # Anti-affinity trials (all 4 pods different nodes)
    for i in $(seq 1 "${NUM_TRIALS}"); do
        echo "── [${delay}ms W4] Anti-Affinity Trial ${i}/${NUM_TRIALS}"
        kubectl apply -f G23_k8s_antiaffinity_w4.yaml
        kubectl wait --for=jsonpath='{.status.phase}'=Succeeded pod/electra-master --timeout=900s
        kubectl logs electra-master | sed -n '/===CSV_START===/,/===CSV_END===/p' | grep -v '===CSV' > "${RESULTS_DIR}/G23_results_antiaffinity_run${i}.csv"
        kubectl delete -f G23_k8s_antiaffinity_w4.yaml
        TOTAL=$(tail -1 "${RESULTS_DIR}/G23_results_antiaffinity_run${i}.csv" | cut -d',' -f6)
        echo "   ✓ Total: ${TOTAL}s"
        sleep 2
    done
done

bash G23_netem_clear.sh

ELAPSED=$(($(date +%s) - SWEEP_START))
echo ""
echo "══ Sweep complete in $((ELAPSED / 60))m $((ELAPSED % 60))s ══"
echo "  Results: results_w4_*ms/"
