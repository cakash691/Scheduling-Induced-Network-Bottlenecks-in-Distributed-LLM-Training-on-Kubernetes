#!/bin/bash
# G23_sweep_run.sh
#
# Runs the complete latency sweep:
#   For each latency value in LATENCIES:
#     1. Apply tc netem delay to worker nodes
#     2. Run NUM_TRIALS affinity + NUM_TRIALS anti-affinity experiments
#     3. Save all CSVs to results_<delay>ms/
#   Finally, clear tc rules and print a summary.
#
# Usage: ./G23_sweep_run.sh
#
# Override latencies: LATENCIES="0 5 10" ./G23_sweep_run.sh
# Override trial count: NUM_TRIALS=3 ./G23_sweep_run.sh

set -e

LATENCIES=${LATENCIES:-"0 1 5 10 25"}
NUM_TRIALS=${NUM_TRIALS:-5}

echo "══════════════════════════════════════════"
echo "  G23 Latency Sweep"
echo "  Latencies : ${LATENCIES} ms"
echo "  Trials    : ${NUM_TRIALS} per config per latency"
echo "══════════════════════════════════════════"

SWEEP_START=$(date +%s)

for delay in ${LATENCIES}; do
    echo ""
    echo "══════════════════════════════════════════"
    echo "  LATENCY = ${delay}ms"
    echo "══════════════════════════════════════════"

    RESULTS_DIR="results_${delay}ms"
    mkdir -p "${RESULTS_DIR}"

    # Apply latency (or clear if 0)
    if [ "${delay}" -eq 0 ]; then
        echo "── Clearing any existing tc rules (baseline)..."
        bash G23_netem_clear.sh
    else
        echo "── Applying ${delay}ms latency..."
        bash G23_netem_apply.sh "${delay}"
    fi

    # Affinity trials
    for i in $(seq 1 "${NUM_TRIALS}"); do
        echo ""
        echo "── [${delay}ms] Affinity Trial ${i}/${NUM_TRIALS} ──"
        kubectl apply -f G23_k8s_affinity.yaml
        kubectl wait --for=jsonpath='{.status.phase}'=Succeeded pod/electra-master --timeout=600s
        kubectl logs electra-master | sed -n '/===CSV_START===/,/===CSV_END===/p' | grep -v '===CSV' > "${RESULTS_DIR}/G23_results_affinity_run${i}.csv"
        kubectl delete -f G23_k8s_affinity.yaml
        TOTAL=$(tail -1 "${RESULTS_DIR}/G23_results_affinity_run${i}.csv" | cut -d',' -f3)
        echo "   ✓ Total: ${TOTAL}s"
        sleep 2
    done

    # Anti-Affinity trials
    for i in $(seq 1 "${NUM_TRIALS}"); do
        echo ""
        echo "── [${delay}ms] Anti-Affinity Trial ${i}/${NUM_TRIALS} ──"
        kubectl apply -f G23_k8s_antiaffinity.yaml
        kubectl wait --for=jsonpath='{.status.phase}'=Succeeded pod/electra-master --timeout=600s
        kubectl logs electra-master | sed -n '/===CSV_START===/,/===CSV_END===/p' | grep -v '===CSV' > "${RESULTS_DIR}/G23_results_antiaffinity_run${i}.csv"
        kubectl delete -f G23_k8s_antiaffinity.yaml
        TOTAL=$(tail -1 "${RESULTS_DIR}/G23_results_antiaffinity_run${i}.csv" | cut -d',' -f3)
        echo "   ✓ Total: ${TOTAL}s"
        sleep 2
    done

    echo ""
    echo "── Completed ${delay}ms (${NUM_TRIALS} affinity + ${NUM_TRIALS} anti-affinity)"
done

# Always clear latency at the end
echo ""
echo "══ Clearing all tc rules..."
bash G23_netem_clear.sh

SWEEP_END=$(date +%s)
ELAPSED=$((SWEEP_END - SWEEP_START))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "══════════════════════════════════════════"
echo "  Sweep complete!"
echo "  Total time: ${MINUTES}m ${SECONDS}s"
echo ""
echo "  Results directories:"
for delay in ${LATENCIES}; do
    echo "    results_${delay}ms/"
done
echo ""
echo "  Next step:"
echo "    make -f G23_Makefile sweep-summary"
echo "══════════════════════════════════════════"
