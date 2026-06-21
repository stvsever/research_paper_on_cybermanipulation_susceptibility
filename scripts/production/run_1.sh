#!/usr/bin/env bash
# ============================================================
# production/run_1.sh - Individual-layer production run (FULL 10,000 scenarios).
#
# This is the production run of the individual (private) susceptibility layer over
# the ENTIRE 10,000-row integrated scenario set. The empirical exposure-network
# layer is OFF; the whole LLM budget goes to the individual layer at full scale.
#
# Design:
#   * scenarios:  all 10,000 rows of the integrated production set (no sub-sampling),
#                 stratified across the 7 issue domains (about 1,429 per domain).
#   * domains:    all 7 opinion parent clusters.
#   * profile:    the deeply reduced research-core profile (about 159 features,
#                 roughly a 53 percent reduction on the previous reduced set and a
#                 71 percent reduction on the full 540-feature integrated profile).
#                 KEPT: full hierarchical Big Five, the core demographic markers
#                 (sex, age, gender identity, citizenship, country of birth,
#                 relationship status, self-identified ethnicity), and the complete
#                 political-psychology / ideology / moral-foundations battery.
#                 DROPPED: the political-participation taxonomy, the socioeconomic /
#                 employment / housing / migration / education life-circumstance
#                 taxonomies, and the over-detailed identity sub-spectra.
#   * stages:     01 to 05 ONLY (scenario build, baseline B, attack spec, post P,
#                 effectivity deltas). The post-hoc analyses (stage 06) and the
#                 visuals / publication assets (stages 07 to 08) are intentionally
#                 NOT run here; the deltas are the deliverable and the analyses are
#                 layered on afterwards.
#   * storage:    lean. Stage 05 keeps only the compact CSV delta tables (every B,
#                 P, delta and effectivity score per scenario and leaf); the large
#                 redundant JSONL mirrors are skipped and no _report / _report_assets
#                 are generated. Raw LLM provenance is written during the run only to
#                 drive the live progress monitor and is deleted at the end, so the
#                 final run does not retain it. The full source content of each
#                 scenario_id is recoverable by joining the
#                 integrated_scenarios_10000.jsonl file on scenario_id.
#   * model:      deepseek/deepseek-v4-flash through OpenRouter.
#   * LLM calls:  about 20,000 (10,000 baseline B + 10,000 post-attack P, plus a
#                 few JSON repairs). Stage 03 is deterministic. Failed calls auto
#                 repair (JSON) and retry, then fall back deterministically so the
#                 run always completes with full coverage.
#
# Output: evaluation/production/run_1/
#
# Usage:
#   bash scripts/production/run_1.sh            # quiet
#   bash scripts/production/run_1.sh --verbose  # live progress monitor
# ============================================================

set -euo pipefail

VERBOSE=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --verbose)    VERBOSE=1 ;;
        --no-verbose) VERBOSE=0 ;;
        -h|--help)
            grep '^# ' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "production/run_1.sh: unknown argument '$1' (try --verbose, --help)" >&2
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

OUT_ROOT="evaluation/production/run_1"
if [ "${VERBOSE}" = "1" ]; then VERBOSE_FLAG="--verbose"; else VERBOSE_FLAG="--no-verbose"; fi

echo "============================================================"
echo " PRODUCTION RUN 1  |  individual layer only (no network)"
echo " scenarios:        ALL 10000 (stratified over 7 domains)"
echo " profile:          deep research-core (~159 features)"
echo " stages:           01..05 only (deltas are the deliverable)"
echo " storage:          lean (compact CSVs; no raw_llm, no report)"
echo " output:           ${OUT_ROOT}"
echo "============================================================"

# Pre-flight budget gate. ~2 LLM calls per scenario (B + P); ~20,000 for 10,000
# scenarios. With the deeply reduced profile each prompt is much shorter, so the
# projected cost is roughly 20 to 26 USD. Require a comfortable floor before we start.
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
FLOOR = 26.00
if spendable < FLOOR:
    sys.exit(
        f"Spendable {spendable:.2f} USD is below the {FLOOR:.2f} USD floor for this full 10,000-scenario run. "
        "Top up or raise the key limit at https://openrouter.ai/settings/keys, then re-run."
    )
EOF

mkdir -p "${OUT_ROOT}/logs"
LOG="${OUT_ROOT}/logs/production_run_1_console.log"

# Individual-layer pipeline, stages 01..05 only. NO --with-network-exposure.
# --lean-storage keeps the footprint small; raw LLM provenance is saved during the
# run (so the live monitor can report call counts and ETA) and deleted at the end;
# --stop-after-stage 05 skips the post-hoc analyses and the visual / publication stages.
COMMON_ARGS=(
    --output-root           "${OUT_ROOT}"
    --run-id                "production_run_1"
    --integrated-scenarios-path "${INTEGRATED_SCENARIOS}"
    --n-scenarios           10000
    --no-max-entropy-subsample
    --seed                  1001
    --attack-ratio          1.0
    --primary-moderator     "profile_cont_age_years"
    --no-use-test-ontology
    --ontology-root         "src/backend/ontology/separate/production"
    --no-enforce-compatibility-rules
    --drop-direction-neutral-opinions
    --no-with-network-exposure
    --openrouter-model      "deepseek/deepseek-v4-flash"
    --temperature           0.15
    --max-repair-iter       1
    --profile-generation-mode deterministic
    --no-self-supervise-opinion-coherence
    --no-self-supervise-attack-realism
    --coherence-threshold   0.74
    "${VERBOSE_FLAG}"
    --no-generate-visuals
    --no-export-static-figures
    --no-build-report
    --save-raw-llm
    --lean-storage
    --timeout-sec           180
    --max-concurrency       80
    --log-level             INFO
)

# Measurement + effect construction only (stages 01..05).
"${PY}" src/backend/pipeline/full/run_full_pipeline.py "${COMMON_ARGS[@]}" \
    --resume-from-stage "01" --stop-after-stage "05" \
    2>&1 | tee "${LOG}"

# Tidy the run layout: keep stage_outputs (canonical, includes the deltas) and the
# config / logs / provenance; drop the empty analysis / visual scaffolding dirs that
# _copy_outputs always creates, the duplicate top-level datasets copy, and any
# _report scaffolding so nothing stale or redundant is left behind.
for d in sem report visuals publication_assets paper _report _report_assets datasets embeddings embeddings_production; do
    rm -rf "${OUT_ROOT:?}/${d}"
done
rm -rf "${OUT_ROOT}/provenance/raw_llm"

echo
echo "=== production run 1 complete ==="
echo "Per-scenario deltas (B, P, delta, effectivity):"
echo "  ${OUT_ROOT}/stage_outputs/05_compute_effectivity_deltas/sem_long_raw.csv"
echo "  ${OUT_ROOT}/stage_outputs/05_compute_effectivity_deltas/sem_long_encoded.csv"
echo "Baseline B:  ${OUT_ROOT}/stage_outputs/02_assess_baseline_opinions/"
echo "Post-attack P: ${OUT_ROOT}/stage_outputs/04_assess_post_attack_opinions/"
echo "Trace any scenario_id back to its full content in:"
echo "  ${INTEGRATED_SCENARIOS}"
du -sh "${OUT_ROOT}" 2>/dev/null | awk '{print "Total run size: "$1}'
