#!/usr/bin/env bash
# ============================================================
# run_3.sh - Launch test run 3 (pre-built integrated scenario set).
#
# Run 3 re-runs the run-2 INDIVIDUAL-LAYER design on a pipeline that now also
# carries the additive empirical EXPOSURE-NETWORK layer (stages 01b / 02b / 04b,
# the network-exposure agents, prompts and the PolitiSky24 substrate), adapted
# onto this pipeline's contracts (full profile configuration, DISARM Plan /
# Prepare / Execute attack triplet, opinion parent cluster).
#
# Two toggles, both OFF by default:
#   --verbose   professional live progress monitor (per-stage call counts,
#               rate and ETA). Default off -> standard INFO logging only.
#   --network   also run the empirical exposure-network side branches
#               (01b / 02b / 04b) to produce the four-state backbone B/BN/P/PN.
#               Default off -> pure individual-layer run at exactly two LLM
#               calls per scenario (~200 calls for 100 scenarios).
#
# Usage:
#   bash scripts/tests/run_3.sh                 # individual layer, quiet
#   bash scripts/tests/run_3.sh --verbose       # individual layer, live progress
#   bash scripts/tests/run_3.sh --network       # + exposure-network layer
#   bash scripts/tests/run_3.sh --verbose --network
#
# Design rationale (inherited from run 2): stage 01 no longer samples from the
# ontology. It consumes the pre-built INTEGRATED scenario set directly, where
# every row already pairs:
#     1 full high-resolution profile configuration
#   + 1 DISARM-red Plan / Prepare / Execute attack triplet (external ontology)
#   + 1 opinion parent cluster (one issue domain + all its directional leaves)
#
# Stage 01 selects 100 scenarios stratified across the 7 issue domains (seed
# 120). Opinions are assessed CLUSTER-AT-ONCE (one baseline call + one
# post-attack call per scenario); stage 05 expands the per-leaf scores back into
# the standard long table. The opinion-coherence reviewer stays available but is
# switched OFF here to keep the test cheap.
#
# Outputs: evaluation/tests/run_3/ with the interactive dashboard at
# evaluation/tests/run_3/visuals/dashboard_results.html
# ============================================================

set -euo pipefail

VERBOSE=0
WITH_NETWORK=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --verbose)    VERBOSE=1 ;;
        --no-verbose) VERBOSE=0 ;;
        --network|--with-network-exposure)       WITH_NETWORK=1 ;;
        --no-network|--no-with-network-exposure) WITH_NETWORK=0 ;;
        -h|--help)
            grep '^# ' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "run_3.sh: unknown argument '$1' (try --verbose, --network, --help)" >&2
            exit 2
            ;;
    esac
    shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/.env"
    set +a
fi

PY="${PROJECT_ROOT}/.venv/bin/python"
[ -x "${PY}" ] || PY="python"

INTEGRATED_SCENARIOS="${PROJECT_ROOT}/src/backend/pipeline/separate/01_create_scenarios/samples/02_integrated/integrated_scenarios_10000.jsonl"
if [ ! -f "${INTEGRATED_SCENARIOS}" ]; then
    echo "Integrated scenario set not found: ${INTEGRATED_SCENARIOS}" >&2
    exit 1
fi

# Resolve the toggles into pipeline flags.
if [ "${VERBOSE}" = "1" ]; then VERBOSE_FLAG="--verbose"; else VERBOSE_FLAG="--no-verbose"; fi
NETWORK_FLAGS=("--no-with-network-exposure")
if [ "${WITH_NETWORK}" = "1" ]; then
    NETWORK_FLAGS=(
        "--with-network-exposure"
        "--exposure-network-root" "src/data/exposure_networks/politisky24_bluesky_v1"
        "--network-exposure-top-k" "8"
    )
fi

echo "============================================================"
echo " RUN 3  |  individual layer$( [ "${WITH_NETWORK}" = "1" ] && echo ' + exposure-network layer' )"
echo " verbose:          $( [ "${VERBOSE}" = "1" ] && echo on || echo off )"
echo " network exposure: $( [ "${WITH_NETWORK}" = "1" ] && echo on || echo off )"
echo " output:           evaluation/tests/run_3"
echo "============================================================"

# Pre-flight budget gate. Run 3 makes ~2 LLM calls per scenario (full profile +
# whole opinion cluster); the network layer adds two more measurement phases.
"${PY}" - <<'EOF'
import os, sys, httpx
key = os.environ.get("OPENROUTER_API_KEY", "")
if not key:
    sys.exit("OPENROUTER_API_KEY missing from environment / .env")
c = httpx.get("https://openrouter.ai/api/v1/credits",
              headers={"Authorization": f"Bearer {key}"}, timeout=30).json()["data"]
k = httpx.get("https://openrouter.ai/api/v1/auth/key",
              headers={"Authorization": f"Bearer {key}"}, timeout=30).json()["data"]
balance = float(c.get("total_credits", 0)) - float(c.get("total_usage", 0))
key_left = k.get("limit_remaining")
spendable = balance if key_left is None else min(balance, float(key_left))
print(f"OpenRouter spendable: {spendable:.2f} USD (account {balance:.2f}, key-month {key_left})")
if spendable < 1.00:
    sys.exit(
        f"Spendable {spendable:.2f} USD is below the 1.00 USD floor for this run. "
        "Top up or raise the key limit at https://openrouter.ai/settings/keys, then re-run."
    )
EOF

mkdir -p "evaluation/tests/run_3/logs"
LOG="evaluation/tests/run_3/logs/run_3_console.log"

# Shared pipeline arguments for both phases.
COMMON_ARGS=(
    --output-root           "evaluation/tests/run_3"
    --run-id                "run_3"
    --paper-title           "Inter-individual Differences in Susceptibility to Cyber-manipulation of Political Opinions: An Ontology-Constrained Multi-Agent Simulation Approach"
    --report-root           "research_report/report"
    --report-assets-root    "research_report/assets"
    --integrated-scenarios-path "${INTEGRATED_SCENARIOS}"
    --n-scenarios           100
    --seed                  120
    --attack-ratio          1.0
    --primary-moderator     "posthoc_profile_susceptibility_index"
    --bootstrap-samples     200
    --no-use-test-ontology
    --ontology-root         "src/backend/ontology/separate/production"
    --no-enforce-compatibility-rules
    --drop-direction-neutral-opinions
    --openrouter-model      "deepseek/deepseek-v4-flash"
    --temperature           0.15
    --max-repair-iter       1
    --profile-generation-mode deterministic
    --no-self-supervise-opinion-coherence
    --no-self-supervise-attack-realism
    --coherence-threshold   0.74
    "${NETWORK_FLAGS[@]}"
    "${VERBOSE_FLAG}"
    --generate-visuals
    --export-static-figures
    --no-build-report
    --save-raw-llm
    --timeout-sec           180
    --max-concurrency       8
    --log-level             INFO
)

# Phase 1: measurement + effect construction + SEM (stages 01..06, plus the
# network side-branches when --network is set).
"${PY}" src/backend/pipeline/full/run_full_pipeline.py "${COMMON_ARGS[@]}" \
    --resume-from-stage "01" --stop-after-stage "06" \
    2>&1 | tee "${LOG}"

# Ontology semantic embeddings for the dashboard's ontology explorer (computed
# before Stage 07 so the generated dashboard includes them).
"${PY}" - <<'PYEOF' 2>&1 | tee -a "${LOG}"
import sys
sys.path.insert(0, ".")
from src.backend.utils.embeddings.semantic_embedding import embed_ontology
for out in ("evaluation/tests/run_3/embeddings", "evaluation/tests/run_3/embeddings_production"):
    embed_ontology("src/backend/ontology/separate/production", out)
print("ontology semantic embeddings computed")
PYEOF

# Phase 2: research visuals + publication assets (stages 07..08).
"${PY}" src/backend/pipeline/full/run_full_pipeline.py "${COMMON_ARGS[@]}" \
    --resume-from-stage "07" --stop-after-stage "08" \
    2>&1 | tee -a "${LOG}"

# Group the run output into a small, clear top-level layout
# (analysis/ visuals/ publication/ alongside config/ logs/ provenance/ stage_outputs/).
# Inlined here so the run is reproducible from this one script with no helper.
RUN_DIR="evaluation/tests/run_3"
group_into() {
    local group="$1"; shift
    mkdir -p "${RUN_DIR}/${group}"
    for name in "$@"; do
        if [ -e "${RUN_DIR}/${name}" ]; then
            rm -rf "${RUN_DIR}/${group}/${name}"
            mv "${RUN_DIR}/${name}" "${RUN_DIR}/${group}/${name}"
            echo "  grouped ${name} -> ${group}/${name}"
        fi
    done
}
group_into analysis datasets sem report
group_into visuals embeddings embeddings_production network_exposure_analysis
group_into publication publication_assets paper
echo "finalized run layout at ${RUN_DIR}"

echo
echo "=== test run 3 complete ==="
echo "Interactive dashboard: evaluation/tests/run_3/visuals/dashboard_results.html"
echo "Network report:        evaluation/tests/run_3/visuals/network_exposure_analysis/reports/run_3_network_exposure_report.html"
