#!/usr/bin/env bash
# ============================================================
# run_4.sh - Launch test run 4 (concentrated 2-domain exposure-network run).
#
# Run 4 is the exposure-network reference. It re-runs the integrated production
# design BUT concentrates the scenario budget into TWO opinion parent clusters
# instead of spreading it across all seven. This is a deliberate methodological
# fix, not a shortcut: the empirical exposure network only carries signal for the
# network-position hypotheses when each opinion leaf is scored by MANY profiles,
# so that a target's same-leaf incoming-peer neighborhood is large. Spreading 100
# scenarios over 7 domains (run 3) left ~3 scored peers per measurement and 1-5
# profiles per opinion x Execute-tactic condition, i.e. the per-condition
# centrality-susceptibility alignment and outcome means were pure sampling noise
# (the H3 association came out near zero / wrong-signed). Concentrating 200
# scenarios into 2 domains lifts the same quantities to ~16-20 scored peers per
# measurement and ~16 profiles per condition, which is the regime in which the
# alignment-vs-network-effect association is actually estimable.
#
# Design (200 scenarios, 2 issue domains, exposure-network layer ON):
#   * scenarios:   200 from the 10,000-row integrated production set (seed 404),
#                  stratified across exactly two opinion parent clusters:
#                    - Information_Integrity_And_Platforms
#                    - Democratic_Resilience_And_Institutions
#   * each row:    1 full profile + 1 near-unique DISARM Plan/Prepare/Execute
#                  triplet + 1 opinion parent cluster (all its directional leaves)
#   * four-state backbone per opinion-cluster leaf: B / BN / P / PN
#   * ~820 LLM calls (200 B + 200 BN + 200 P + 200 PN + a few JSON repairs)
#
# Network-layer scenario governor: the exposure-network layer is capped at 500
# scenarios (--network-scenario-cap). Run 4 is well under the cap so nothing is
# dropped; for the production 10K the cap engages a media-keyword heuristic so the
# retained subset matches the social-media (Bluesky) exposure substrate.
#
# Toggles:
#   --verbose      professional live progress monitor (per-stage call counts).
#   --no-network   individual layer only (skips 01b/02b/04b/05b). Default: network ON.
#
# Usage:
#   bash scripts/tests/run_4.sh                 # 2-domain individual + exposure-network
#   bash scripts/tests/run_4.sh --verbose       # + live progress monitor
#   bash scripts/tests/run_4.sh --no-network    # individual layer only
#
# Outputs: evaluation/tests/run_4/ with the interactive dashboard at
# evaluation/tests/run_4/visuals/dashboard_results.html and the comprehensive
# exposure-network report at
# evaluation/tests/run_4/visuals/network_exposure_analysis/reports/.
# ============================================================

set -euo pipefail

VERBOSE=0
WITH_NETWORK=1
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
            echo "run_4.sh: unknown argument '$1' (try --verbose, --no-network, --help)" >&2
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

# The two issue domains this run concentrates into (densifies the exposure network).
FOCUS_DOMAINS="Information_Integrity_And_Platforms,Democratic_Resilience_And_Institutions"

# Resolve the toggles into pipeline flags.
if [ "${VERBOSE}" = "1" ]; then VERBOSE_FLAG="--verbose"; else VERBOSE_FLAG="--no-verbose"; fi
NETWORK_FLAGS=("--no-with-network-exposure")
if [ "${WITH_NETWORK}" = "1" ]; then
    NETWORK_FLAGS=(
        "--with-network-exposure"
        "--exposure-network-root" "src/data/exposure_networks/politisky24_bluesky_v1"
        "--network-exposure-top-k" "8"
        "--network-scenario-cap" "500"
    )
fi

echo "============================================================"
echo " RUN 4  |  2-domain concentrated individual$( [ "${WITH_NETWORK}" = "1" ] && echo ' + exposure-network' ) layer"
echo " focus domains:    ${FOCUS_DOMAINS}"
echo " scenarios:        200 (seed 404)"
echo " verbose:          $( [ "${VERBOSE}" = "1" ] && echo on || echo off )"
echo " network exposure: $( [ "${WITH_NETWORK}" = "1" ] && echo on || echo off )"
echo " output:           evaluation/tests/run_4"
echo "============================================================"

# Pre-flight budget gate. Run 4 makes ~4 LLM calls per scenario with the network
# layer on (B + BN + P + PN); ~820 calls for 200 scenarios.
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
if spendable < 1.50:
    sys.exit(
        f"Spendable {spendable:.2f} USD is below the 1.50 USD floor for this run. "
        "Top up or raise the key limit at https://openrouter.ai/settings/keys, then re-run."
    )
EOF

mkdir -p "evaluation/tests/run_4/logs"
LOG="evaluation/tests/run_4/logs/run_4_console.log"

# Shared pipeline arguments for both phases. --save-raw-llm keeps every raw LLM
# request/response under provenance/raw_llm so all data is retained for later
# analyses; stage_outputs retains every B/BN/P/PN assessment, context and delta.
COMMON_ARGS=(
    --output-root           "evaluation/tests/run_4"
    --run-id                "run_4"
    --paper-title           "Inter-individual Differences in Susceptibility to Cyber-manipulation of Political Opinions: An Ontology-Constrained Multi-Agent Simulation Approach"
    --report-root           "report/report"
    --report-assets-root    "report/assets"
    --integrated-scenarios-path "${INTEGRATED_SCENARIOS}"
    --n-scenarios           200
    --focus-opinion-domains "${FOCUS_DOMAINS}"
    --seed                  404
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
    --max-concurrency       20
    --log-level             INFO
)

# Phase 1: measurement + effect construction + SEM (stages 01..06, plus the
# network side-branches when network exposure is on).
"${PY}" src/backend/pipeline/full/run_full_pipeline.py "${COMMON_ARGS[@]}" \
    --resume-from-stage "01" --stop-after-stage "06" \
    2>&1 | tee "${LOG}"

# Ontology semantic embeddings for the dashboard's ontology explorer (computed
# before Stage 07 so the generated dashboard includes them).
"${PY}" - <<'PYEOF' 2>&1 | tee -a "${LOG}"
import sys
sys.path.insert(0, ".")
from src.backend.utils.embeddings.semantic_embedding import embed_ontology
for out in ("evaluation/tests/run_4/embeddings", "evaluation/tests/run_4/embeddings_production"):
    embed_ontology("src/backend/ontology/separate/production", out)
print("ontology semantic embeddings computed")
PYEOF

# Phase 2: research visuals + publication assets (stages 07..08).
"${PY}" src/backend/pipeline/full/run_full_pipeline.py "${COMMON_ARGS[@]}" \
    --resume-from-stage "07" --stop-after-stage "08" \
    2>&1 | tee -a "${LOG}"

# Group the run output into a small, clear top-level layout
# (analysis/ visuals/ publication/ alongside config/ logs/ provenance/ stage_outputs/).
RUN_DIR="evaluation/tests/run_4"
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
echo "=== test run 4 complete ==="
echo "Interactive dashboard: evaluation/tests/run_4/visuals/dashboard_results.html"
echo "Network report:        evaluation/tests/run_4/visuals/network_exposure_analysis/reports/run_4_network_exposure_report.html"
