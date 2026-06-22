#!/usr/bin/env bash
# ============================================================
# run_1.sh - Launch test run 1 (focused factorial, scaled profiles).
#
# Design rationale: test run 1 uses a small, maximally interpretable
# attack x opinion factorial crossed with a large, maximally dispersed
# profile panel so profile moderation can be inspected under the testing
# ontology.
#
#   - 4 attack vectors, one per canonical cognitive-warfare mechanism family:
#       Headline_And_Lede_Misframing            (misinformation framing, T1)
#       Personal_Safety_Fear_Appeal             (fear appeal, T1)
#       Petition_Astroturf                      (astroturfed consensus, T2)
#       Multi_Turn_Counter_Argument_Adaptation  (AI chatbot persuasion, T4)
#   - 3 opinion leaves, classic cognitive-warfare targets (all erosion goals):
#       Alliance_Commitment_Support             (alliance cohesion)
#       Trust_In_Mainstream_Journalism          (epistemic trust)
#       Defense_Spending_Increase_Support       (defense investment)
#   - 60 pseudoprofiles via maximal-entropy (farthest-point) selection from
#     an 8x candidate pool sampled under the PROFILE ontology coherence
#     rules; this counteracts statistical range restriction in moderators.
#
# Measurement guarantees: deterministic attack-vector specs,
# goal-aware interval-constrained post elicitation, AE >= 0 by design,
# deepseek-v4-flash.
#
# Outputs: evaluation/tests/run_1/ with the interactive dashboard at
# evaluation/tests/run_1/visuals/dashboard_results.html
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

# Pre-flight budget gate: ~1.3 USD projected (2,880 calls at deepseek-v4-flash
# rates plus rewrite overhead); refuse to start below the safety floor.
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
if spendable < 1.45:
    sys.exit(
        f"Spendable {spendable:.2f} USD is below the 1.45 USD floor for this run. "
        "Top up or raise the key limit at https://openrouter.ai/settings/keys, then re-run."
    )
EOF

mkdir -p "evaluation/tests/logs"

"${PY}" src/backend/pipeline/full/run_full_pipeline.py \
    --output-root        "evaluation/tests/run_1" \
    --run-id             "run_1" \
    --paper-title        "Inter-individual Differences in Susceptibility to Cyber-manipulation of Political Opinions: An Ontology-Constrained Multi-Agent Simulation Approach" \
    --report-root        "report/report" \
    --report-assets-root "report/assets" \
    --n-profiles         60 \
    --seed               120 \
    --attack-ratio       1.0 \
    --attack-leaves      "Headline_And_Lede_Misframing,Personal_Safety_Fear_Appeal,Petition_Astroturf,Multi_Turn_Counter_Argument_Adaptation" \
    --opinion-leaves     "Alliance_Commitment_Support,Trust_In_Mainstream_Journalism,Defense_Spending_Increase_Support" \
    --profile-candidate-multiplier 8 \
    --primary-moderator  "posthoc_profile_susceptibility_index" \
    --bootstrap-samples  600 \
    --use-test-ontology \
    --ontology-root      "src/backend/ontology/separate/test" \
    --enforce-compatibility-rules \
    --drop-direction-neutral-opinions \
    --realism-weight-temperature 1.5 \
    --openrouter-model   "deepseek/deepseek-v4-flash" \
    --temperature        0.15 \
    --max-repair-iter    2 \
    --profile-generation-mode deterministic \
    --self-supervise-opinion-coherence \
    --coherence-threshold 0.74 \
    --generate-visuals \
    --export-static-figures \
    --no-build-report \
    --resume-from-stage  "01" \
    --stop-after-stage   "08" \
    --save-raw-llm \
    --timeout-sec        90 \
    --max-concurrency    24 \
    --log-level          INFO \
    2>&1 | tee evaluation/tests/logs/run_1.log

echo
echo "=== test run 1 complete ==="
echo "Interactive dashboard: evaluation/tests/run_1/visuals/dashboard_results.html"
