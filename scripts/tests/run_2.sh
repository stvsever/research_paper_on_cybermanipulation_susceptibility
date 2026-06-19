#!/usr/bin/env bash
# ============================================================
# run_2.sh - Launch test run 2 (pre-built integrated scenario set).
#
# Design rationale: run 2 no longer samples from the ontology in stage 01.
# It consumes the pre-built INTEGRATED scenario set directly, where every row
# already pairs:
#     1 full high-resolution profile configuration
#   + 1 DISARM-red Plan / Prepare / Execute attack triplet (external ontology)
#   + 1 opinion parent cluster (one issue domain + all its directional leaves)
#
# Stage 01 selects 100 scenarios stratified across the 7 issue domains (random
# within strata, seed 120) so all 7 domains and all 106 directional issue
# positions are covered and every leaf gets enough observations for the
# conditional-susceptibility task models.
#
# Compute-saving change vs run 1: opinions are assessed CLUSTER-AT-ONCE. The
# baseline agent and the post-attack agent each make ONE call per scenario and
# return a score for every leaf of the issue domain (instead of one call per
# leaf). Stage 05 then expands those per-leaf scores back into the standard
# per-leaf long table, so the delta computation, the headline linear
# mixed-effects moderation model, the conditional-susceptibility estimation, the
# research visuals and the publication assets all run on the same structure.
#
# Cost control: this test runs at EXACTLY two LLM calls per scenario (one
# baseline elicitation + one post-attack elicitation) = ~200 calls for 100
# scenarios. The opinion-coherence reviewer / rewrite loop stays available in
# the pipeline but is switched OFF here (--no-self-supervise-opinion-coherence)
# to keep the test cheap; turn it back on for the production paper run.
#
# Methodological guarantees: the simulation agents receive the FULL profile
# configuration (all ~520 attributes) and the raw DISARM triplet (the three
# phase paths) and must reason how those combine into one operation; per-leaf
# adversarial directions are baked in and enforced (post score is clamped into
# the [baseline -> goal] interval). Each scenario keeps its own real attack
# identity; the conditional-susceptibility estimator pools over attacks because
# attacks are ~unique per scenario in this small test.
#
# Outputs: evaluation/tests/run_2/ with the interactive dashboard at
# evaluation/tests/run_2/visuals/dashboard_results.html
# ============================================================

set -euo pipefail

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

# Pre-flight budget gate. Run 2 makes far fewer calls than run 1 (~2 LLM calls
# per scenario instead of one per scenario x opinion-leaf), but each call is
# larger (full profile + whole opinion cluster), so we keep a safety floor.
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

mkdir -p "evaluation/tests/run_2/logs"

"${PY}" src/backend/pipeline/full/run_full_pipeline.py \
    --output-root           "evaluation/tests/run_2" \
    --run-id                "run_2" \
    --paper-title           "Inter-individual Differences in Susceptibility to Cyber-manipulation of Political Opinions: An Ontology-Constrained Multi-Agent Simulation Approach" \
    --report-root           "research_report/report" \
    --report-assets-root    "research_report/assets" \
    --integrated-scenarios-path "${INTEGRATED_SCENARIOS}" \
    --n-scenarios           100 \
    --seed                  120 \
    --attack-ratio          1.0 \
    --primary-moderator     "posthoc_profile_susceptibility_index" \
    --bootstrap-samples     200 \
    --no-use-test-ontology \
    --ontology-root         "src/backend/ontology/separate/production" \
    --no-enforce-compatibility-rules \
    --drop-direction-neutral-opinions \
    --openrouter-model      "deepseek/deepseek-v4-flash" \
    --temperature           0.15 \
    --max-repair-iter       1 \
    --profile-generation-mode deterministic \
    --no-self-supervise-opinion-coherence \
    --no-self-supervise-attack-realism \
    --coherence-threshold   0.74 \
    --generate-visuals \
    --export-static-figures \
    --no-build-report \
    --resume-from-stage     "01" \
    --stop-after-stage      "08" \
    --save-raw-llm \
    --timeout-sec           180 \
    --max-concurrency       8 \
    --log-level             INFO \
    2>&1 | tee evaluation/tests/run_2/logs/run_2_console.log

echo
echo "=== test run 2 complete ==="
echo "Interactive dashboard: evaluation/tests/run_2/visuals/dashboard_results.html"
