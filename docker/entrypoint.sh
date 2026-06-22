#!/usr/bin/env bash
set -euo pipefail

cd /app

: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY is required}"
: "${OPENROUTER_MODEL:?OPENROUTER_MODEL is required}"

extra_flags=()

if [[ "${RUN_STAGE_CHECKS:-true}" == "true" ]]; then
  extra_flags+=(--run-stage-checks)
fi

if [[ "${GENERATE_VISUALS:-true}" == "true" ]]; then
  extra_flags+=(--generate-visuals)
else
  extra_flags+=(--no-generate-visuals)
fi

if [[ "${EXPORT_STATIC_FIGURES:-true}" == "true" ]]; then
  extra_flags+=(--export-static-figures)
else
  extra_flags+=(--no-export-static-figures)
fi

if [[ "${BUILD_REPORT:-false}" == "true" ]]; then
  extra_flags+=(--build-report)
else
  extra_flags+=(--no-build-report)
fi

if [[ "${SELF_SUPERVISE_ATTACK_REALISM:-true}" == "true" ]]; then
  extra_flags+=(--self-supervise-attack-realism)
else
  extra_flags+=(--no-self-supervise-attack-realism)
fi

if [[ "${SELF_SUPERVISE_OPINION_COHERENCE:-true}" == "true" ]]; then
  extra_flags+=(--self-supervise-opinion-coherence)
else
  extra_flags+=(--no-self-supervise-opinion-coherence)
fi

python src/backend/pipeline/full/run_full_pipeline.py \
  --output-root "${OUTPUT_ROOT:-evaluation/tests/run_1}" \
  --run-id "${RUN_ID:-run_1}" \
  --n-profiles "${N_PROFILES:-60}" \
  --seed "${PIPELINE_SEED:-120}" \
  --attack-ratio "${ATTACK_RATIO:-1.0}" \
  --attack-leaves "${ATTACK_LEAVES:-Headline_And_Lede_Misframing,Personal_Safety_Fear_Appeal,Petition_Astroturf,Multi_Turn_Counter_Argument_Adaptation}" \
  --opinion-leaves "${OPINION_LEAVES:-Alliance_Commitment_Support,Trust_In_Mainstream_Journalism,Defense_Spending_Increase_Support}" \
  --profile-candidate-multiplier "${PROFILE_CANDIDATE_MULTIPLIER:-8}" \
  --use-test-ontology \
  --ontology-root "${ONTOLOGY_ROOT:-src/backend/ontology/separate/test}" \
  --enforce-compatibility-rules \
  --drop-direction-neutral-opinions \
  --openrouter-model "${OPENROUTER_MODEL}" \
  --temperature "${TEMPERATURE:-0.15}" \
  --max-repair-iter "${MAX_REPAIR_ITER:-2}" \
  --profile-generation-mode "${PROFILE_GENERATION_MODE:-deterministic}" \
  --realism-threshold "${REALISM_THRESHOLD:-0.72}" \
  --coherence-threshold "${COHERENCE_THRESHOLD:-0.74}" \
  --primary-moderator "${PRIMARY_MODERATOR:-posthoc_profile_susceptibility_index}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES:-600}" \
  --paper-title "${PAPER_TITLE:-Inter-individual Differences in Susceptibility to Cyber-manipulation of Political Opinions: An Ontology-Constrained Multi-Agent Simulation Approach}" \
  --report-root "${REPORT_ROOT:-report/report}" \
  --report-assets-root "${REPORT_ASSETS_ROOT:-report/assets}" \
  "${extra_flags[@]}" \
  "$@"
