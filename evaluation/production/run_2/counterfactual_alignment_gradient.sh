#!/usr/bin/env bash
# Alignment-gradient branch launcher for run_2 H3/H4 rerun preparation.
#
# Modes:
#   prepare                 local; build condition inputs and design manifest
#   validate                local; validate prepared inputs and, if present, merged outputs
#   run-condition <id>      paid; run Stage 04b for one prepared condition
#   run-all                 paid; run Stage 04b for every prepared condition
#   merge                   local; merge completed condition Stage 04b outputs
#   analyze                 local; run branch-safe Stage 05 and H3/H4 report

set -euo pipefail

MODE="${1:-}"
CONDITION_ID="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

PY="${PROJECT_ROOT}/.venv/bin/python"
RUN_ROOT="${PROJECT_ROOT}/evaluation/production/run_2"
BRANCH_ROOT="${RUN_ROOT}/counterfactual_alignment_gradient"
ONTOLOGY_ROOT="${RUN_ROOT}/config/ontology_mixed"
RAW_LLM_DIR="${BRANCH_ROOT}/provenance/raw_llm"
MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-cybermanipulation}"
export MPLCONFIGDIR

usage() {
    cat <<'EOF'
Usage: bash evaluation/production/run_2/counterfactual_alignment_gradient.sh <mode> [condition_id]

Modes:
  prepare                 build branch design and 35 condition inputs; no LLM calls
  validate                validate prepared branch and merged outputs if present; no LLM calls
  run-condition <id>      run paid Stage 04b for a single condition
  run-all                 run paid Stage 04b for all prepared conditions
  merge                   merge completed condition Stage 04b outputs; no LLM calls
  analyze                 run branch-safe Stage 05 and H3/H4 report; no LLM calls
EOF
}

if [[ -z "${MODE}" || "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ ! -x "${PY}" ]]; then
    echo "ERROR: ${PY} is missing or not executable." >&2
    exit 2
fi

mkdir -p "${MPLCONFIGDIR}"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/.env"
    set +a
fi

require_openrouter() {
    if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
        echo "ERROR: OPENROUTER_API_KEY is required for paid Stage 04b branch runs." >&2
        exit 3
    fi
}

condition_input_path() {
    local condition_id="$1"
    echo "${BRANCH_ROOT}/condition_inputs/${condition_id}.jsonl"
}

condition_output_dir() {
    local condition_id="$1"
    echo "${BRANCH_ROOT}/condition_stage04b_outputs/${condition_id}"
}

run_condition() {
    local condition_id="$1"
    if [[ -z "${condition_id}" ]]; then
        echo "ERROR: run-condition requires a condition id." >&2
        exit 2
    fi
    local input_path
    input_path="$(condition_input_path "${condition_id}")"
    local output_dir
    output_dir="$(condition_output_dir "${condition_id}")"
    if [[ ! -f "${input_path}" ]]; then
        echo "ERROR: missing condition input: ${input_path}" >&2
        exit 2
    fi
    if [[ -f "${output_dir}/manifest.json" && "${ALIGNMENT_GRADIENT_FORCE:-0}" != "1" ]]; then
        echo "ERROR: condition output already exists: ${output_dir}" >&2
        echo "Set ALIGNMENT_GRADIENT_FORCE=1 to overwrite intentionally." >&2
        exit 2
    fi
    mkdir -p "${output_dir}" "${BRANCH_ROOT}/logs" "${RAW_LLM_DIR}"
    "${PY}" src/backend/pipeline/separate/04b_assess_post_attack_network_exposure_opinions/run_stage.py \
        --input-path "${input_path}" \
        --output-dir "${output_dir}" \
        --run-id "run_2_alignment_gradient_${condition_id}" \
        --seed 120 \
        --openrouter-model "deepseek/deepseek-v4-flash" \
        --temperature 0.15 \
        --max-repair-iter 3 \
        --self-supervise-opinion-coherence \
        --coherence-threshold 0.74 \
        --post-attack-network-exposure-top-k 8 \
        --post-attack-network-min-peers 1 \
        --save-raw-llm \
        --raw-llm-dir "${RAW_LLM_DIR}" \
        --timeout-sec 120 \
        --max-concurrency 8 \
        --log-file "${BRANCH_ROOT}/logs/stage04b_${condition_id}.log" \
        --log-level INFO
}

case "${MODE}" in
    prepare)
        "${PY}" -m src.backend.utils.alignment_gradient_branch prepare \
            --run-root "${RUN_ROOT}" \
            --branch-root "${BRANCH_ROOT}" \
            --seed 120
        "${PY}" -m src.backend.utils.alignment_gradient_branch validate-prepared \
            --run-root "${RUN_ROOT}" \
            --branch-root "${BRANCH_ROOT}"
        ;;
    validate)
        "${PY}" -m src.backend.utils.alignment_gradient_branch validate-prepared \
            --run-root "${RUN_ROOT}" \
            --branch-root "${BRANCH_ROOT}"
        if [[ -f "${BRANCH_ROOT}/merged_outputs/stage_outputs/04b_assess_post_attack_network_exposure_opinions/post_attack_network_exposure_summary.json" ]]; then
            "${PY}" -m src.backend.utils.alignment_gradient_branch validate-merged \
                --run-root "${RUN_ROOT}" \
                --branch-root "${BRANCH_ROOT}"
        fi
        ;;
    run-condition)
        require_openrouter
        run_condition "${CONDITION_ID}"
        ;;
    run-all)
        require_openrouter
        schedule="${BRANCH_ROOT}/design/condition_schedule.csv"
        if [[ ! -f "${schedule}" ]]; then
            echo "ERROR: missing schedule; run prepare first." >&2
            exit 2
        fi
        tail -n +2 "${schedule}" | cut -d, -f1 | while IFS= read -r condition_id; do
            run_condition "${condition_id}"
        done
        ;;
    merge)
        "${PY}" -m src.backend.utils.alignment_gradient_branch merge \
            --run-root "${RUN_ROOT}" \
            --branch-root "${BRANCH_ROOT}"
        "${PY}" -m src.backend.utils.alignment_gradient_branch validate-merged \
            --run-root "${RUN_ROOT}" \
            --branch-root "${BRANCH_ROOT}"
        ;;
    analyze)
        mkdir -p "${BRANCH_ROOT}/logs" "${BRANCH_ROOT}/merged_outputs/stage_outputs/05_compute_effectivity_deltas"
        "${PY}" src/backend/pipeline/separate/05_compute_effectivity_deltas/run_stage.py \
            --input-path "${BRANCH_ROOT}/merged_outputs/stage_outputs/04b_assess_post_attack_network_exposure_opinions/scenarios_with_post_attack_network_exposure.jsonl" \
            --output-dir "${BRANCH_ROOT}/merged_outputs/stage_outputs/05_compute_effectivity_deltas" \
            --run-id "run_2_alignment_gradient" \
            --seed 120 \
            --primary-moderator "profile_cont_big_five_neuroticism_mean_pct" \
            --ontology-root "${ONTOLOGY_ROOT}" \
            --exposure-assignment-scope condition_specific \
            --log-file "${BRANCH_ROOT}/logs/stage05_alignment_gradient.log" \
            --log-level INFO
        "${PY}" evaluation/production/run_2/counterfactual_alignment_gradient/scripts/build_alignment_gradient_h3h4_report.py \
            --branch-root "${BRANCH_ROOT}" \
            --run-id "run_2_alignment_gradient" \
            --report-mode full
        ;;
    *)
        usage
        exit 2
        ;;
esac
