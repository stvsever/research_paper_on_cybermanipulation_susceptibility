#!/usr/bin/env bash
# Launch production run 2 preparation modes.
#
# Modes:
#   structural  no paid LLM calls; run Stage 01 through 01b for full 3,500-row design
#   smoke       paid 120-scenario subset through Stage 08b
#   full        paid 3,500-scenario production run through Stage 08b

set -euo pipefail

MODE="${1:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-cybermanipulation}"
mkdir -p "${MPLCONFIGDIR}"

RUN_ROOT="evaluation/production/run_2"
CONFIG_ROOT="${RUN_ROOT}/config"
ONTOLOGY_ROOT="${CONFIG_ROOT}/ontology_mixed"
PY="${PROJECT_ROOT}/.venv/bin/python"

FULL_OPINION_LEAVES="Issue_Position_Taxonomy > Defense_And_National_Security > Alliance_Commitment_Support,Issue_Position_Taxonomy > Foreign_Policy_And_Geopolitics > Unilateralism_Preference,Issue_Position_Taxonomy > Information_Integrity_And_Platforms > Trust_In_Mainstream_Journalism,Issue_Position_Taxonomy > Democratic_Resilience_And_Institutions > Trust_In_Electoral_Process,Issue_Position_Taxonomy > Critical_Infrastructure_And_Energy_Sovereignty > Energy_Sovereignty_Investment_Support,Issue_Position_Taxonomy > Supranational_And_Regional_Integration > National_Sovereignty_Priority,Issue_Position_Taxonomy > Macroeconomic_And_Fiscal_Policy > Trade_With_Authoritarian_States_Support"
FULL_ATTACK_LEAVES="Headline_And_Lede_Misframing,Quote_Context_Stripping,Credentialed_Domain_Persona_Fabrication,Repost_Bot_Amplification,Petition_Astroturf"

SMOKE_OPINION_LEAVES="Issue_Position_Taxonomy > Information_Integrity_And_Platforms > Trust_In_Mainstream_Journalism,Issue_Position_Taxonomy > Macroeconomic_And_Fiscal_Policy > Trade_With_Authoritarian_States_Support"
SMOKE_ATTACK_LEAVES="Quote_Context_Stripping,Repost_Bot_Amplification"

usage() {
    cat <<'EOF'
Usage: bash scripts/production/run_2.sh <mode>

Modes:
  structural  run Stage 01 through 01b for the full 100 x 7 x 5 design
  smoke       run a paid 30 x 2 x 2 subset through Stage 08b
  full        run the full paid 100 x 7 x 5 production design through Stage 08b
EOF
}

if [[ -z "${MODE}" || "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ ! -x "${PY}" ]]; then
    echo "ERROR: ${PY} is missing or not executable." >&2
    echo "Create the project virtualenv first: python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 2
fi

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/.env"
    set +a
fi

validate_imports() {
    "${PY}" - <<'PY'
import importlib

modules = [
    "httpx",
    "pydantic",
    "dotenv",
    "psycopg",
    "pandas",
    "jinja2",
    "numpy",
    "scipy",
    "statsmodels",
    "semopy",
    "sklearn",
    "networkx",
    "plotly",
    "matplotlib",
    "seaborn",
    "umap",
    "rdflib",
]

missing = []
for module in modules:
    try:
        importlib.import_module(module)
    except Exception as exc:
        missing.append(f"{module}: {exc}")

if missing:
    raise SystemExit("Missing required Python imports:\n" + "\n".join(missing))

print("Python dependency preflight passed.")
PY
}

validate_static_config() {
    PROJECT_ROOT_ENV="${PROJECT_ROOT}" "${PY}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["PROJECT_ROOT_ENV"])
config_root = root / "evaluation/production/run_2/config"
ontology_root = config_root / "ontology_mixed"

manifest = json.loads((config_root / "ontology_source_manifest.json").read_text())
opinion_panel = json.loads((config_root / "opinion_panel.json").read_text())
attack_panel = json.loads((config_root / "attack_panel.json").read_text())
scenario_design = json.loads((config_root / "scenario_design.json").read_text())

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_json(path: Path) -> dict:
    return json.loads(path.read_text())

def flatten_leaf_paths(obj, path=()):
    if isinstance(obj, dict):
        children = [
            (key, value)
            for key, value in obj.items()
            if not key.startswith("_") and isinstance(value, dict)
        ]
        if not children and path:
            yield " > ".join(path)
        for key, value in children:
            yield from flatten_leaf_paths(value, path + (key,))

def get_node(obj, path: str):
    cur = obj
    for part in path.split(" > "):
        cur = cur[part]
    return cur

component_files = {
    "PROFILE": ontology_root / "PROFILE/profile.json",
    "ATTACK": ontology_root / "ATTACK/attack.json",
    "OPINION": ontology_root / "OPINION/opinion.json",
}

for axis, path in component_files.items():
    if not path.exists():
        raise SystemExit(f"Missing mixed ontology component: {path}")
    expected = manifest["components"][axis]["sha256"]
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"{axis} hash mismatch: expected {expected}, got {actual}")
    data = read_json(path)
    schema = data.get("_metadata", {}).get("schema_version")
    expected_schema = manifest["components"][axis]["schema_version"]
    if schema != expected_schema:
        raise SystemExit(f"{axis} schema mismatch: expected {expected_schema}, got {schema}")

opinion_data = read_json(component_files["OPINION"])
opinion_leaves = set(flatten_leaf_paths(opinion_data))
directions = []
for item in opinion_panel["opinions"]:
    path = item["path"]
    matches = [leaf for leaf in opinion_leaves if leaf == path]
    if len(matches) != 1:
        raise SystemExit(f"Opinion path does not resolve uniquely: {path} ({len(matches)} matches)")
    direction = int(get_node(opinion_data, path).get("adversarial_direction", 0))
    expected_direction = int(item["adversarial_direction"])
    if direction != expected_direction:
        raise SystemExit(f"Opinion direction mismatch for {path}: expected {expected_direction}, got {direction}")
    directions.append(direction)

if directions != opinion_panel["expected_adversarial_directions"]:
    raise SystemExit(f"Opinion direction sequence mismatch: {directions}")

attack_data = read_json(component_files["ATTACK"])
attack_leaves = list(flatten_leaf_paths(attack_data))
for item in attack_panel["attacks"]:
    leaf = item["leaf"]
    path = item["path"]
    exact_path_matches = [candidate for candidate in attack_leaves if candidate == path]
    leaf_matches = [candidate for candidate in attack_leaves if candidate.endswith(" > " + leaf)]
    if len(exact_path_matches) != 1:
        raise SystemExit(f"Attack path does not resolve uniquely: {path} ({len(exact_path_matches)} matches)")
    if len(leaf_matches) != 1:
        raise SystemExit(f"Attack leaf does not resolve uniquely for CLI substring matching: {leaf} ({len(leaf_matches)} matches)")

counts = scenario_design["counts"]
if counts != {
    "profiles": 100,
    "opinions": 7,
    "attacks": 5,
    "scenarios": 3500,
    "opinion_attack_conditions": 35,
}:
    raise SystemExit(f"Unexpected scenario count block: {counts}")

print("Static production config validation passed.")
PY
}

require_api_key() {
    if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
        echo "ERROR: OPENROUTER_API_KEY is missing from environment / .env." >&2
        exit 2
    fi
}

run_pipeline() {
    local output_root="$1"
    local run_id="$2"
    local n_profiles="$3"
    local n_scenarios="$4"
    local opinion_leaves="$5"
    local attack_leaves="$6"
    local stop_after_stage="$7"

    mkdir -p "${output_root}/logs"

    "${PY}" src/backend/pipeline/full/run_full_pipeline.py \
        --output-root "${output_root}" \
        --run-id "${run_id}" \
        --paper-title "Inter-individual Differences in Susceptibility to Cyber-manipulation of Political Opinions: An Ontology-Constrained Multi-Agent Simulation Approach" \
        --report-root "${output_root}/report_build/report" \
        --report-assets-root "${output_root}/report_build/assets" \
        --n-scenarios "${n_scenarios}" \
        --n-profiles "${n_profiles}" \
        --seed 120 \
        --attack-ratio 1.0 \
        --attack-leaves "${attack_leaves}" \
        --opinion-leaves "${opinion_leaves}" \
        --profile-candidate-multiplier 8 \
        --primary-moderator "posthoc_profile_susceptibility_index" \
        --bootstrap-samples 600 \
        --no-use-test-ontology \
        --ontology-root "${ONTOLOGY_ROOT}" \
        --enforce-compatibility-rules \
        --drop-direction-neutral-opinions \
        --realism-weight-temperature 1.5 \
        --openrouter-model "deepseek/deepseek-v4-flash" \
        --temperature 0.15 \
        --max-repair-iter 2 \
        --profile-generation-mode deterministic \
        --self-supervise-attack-realism \
        --realism-threshold 0.72 \
        --self-supervise-opinion-coherence \
        --coherence-threshold 0.74 \
        --assess-network-exposure \
        --network-exposure-top-k 8 \
        --assess-post-attack-network-exposure \
        --post-attack-network-exposure-top-k 8 \
        --post-attack-network-min-peers 1 \
        --analyze-network-exposure-run \
        --generate-visuals \
        --export-static-figures \
        --no-build-report \
        --resume-from-stage 01 \
        --stop-after-stage "${stop_after_stage}" \
        --save-raw-llm \
        --timeout-sec 90 \
        --max-concurrency 12 \
        --log-level INFO \
        --no-run-stage-checks \
        --no-ingest-to-db \
        2>&1 | tee "${output_root}/logs/launcher.log"
}

validate_stage_outputs() {
    local output_root="$1"
    local expected_scenarios="$2"
    local expected_profiles="$3"
    local expected_opinions="$4"
    local expected_attacks="$5"
    local require_stage_08b="$6"

    OUTPUT_ROOT_ENV="${output_root}" \
    EXPECTED_SCENARIOS="${expected_scenarios}" \
    EXPECTED_PROFILES="${expected_profiles}" \
    EXPECTED_OPINIONS="${expected_opinions}" \
    EXPECTED_ATTACKS="${expected_attacks}" \
    REQUIRE_STAGE_08B="${require_stage_08b}" \
    "${PY}" - <<'PY'
import json
import os
from pathlib import Path

output_root = Path(os.environ["OUTPUT_ROOT_ENV"])
stage_outputs = output_root / "stage_outputs"
expected_scenarios = int(os.environ["EXPECTED_SCENARIOS"])
expected_profiles = int(os.environ["EXPECTED_PROFILES"])
expected_opinions = int(os.environ["EXPECTED_OPINIONS"])
expected_attacks = int(os.environ["EXPECTED_ATTACKS"])
require_stage_08b = os.environ["REQUIRE_STAGE_08B"] == "1"

def read_json(path: Path):
    if not path.exists():
        raise SystemExit(f"Missing expected output file: {path}")
    return json.loads(path.read_text())

stage01_manifest = read_json(stage_outputs / "01_create_scenarios/manifest.json")
stage01_meta = stage01_manifest.get("metadata", {})
audit = read_json(stage_outputs / "01_create_scenarios/scenario_compatibility_audit.json")
assignment = read_json(stage_outputs / "01b_assign_exposure_network_positions/exposure_network_assignment_summary.json")

checks = {
    "record_count": stage01_manifest.get("record_count"),
    "selected_profile_count": stage01_meta.get("selected_profile_count"),
    "selected_opinion_leaf_count": stage01_meta.get("selected_opinion_leaf_count"),
    "selected_attack_leaf_count": stage01_meta.get("selected_attack_leaf_count"),
    "n_scenarios_excluded": audit.get("n_scenarios_excluded"),
    "assignment_profile_count": assignment.get("profile_count"),
}

expected = {
    "record_count": expected_scenarios,
    "selected_profile_count": expected_profiles,
    "selected_opinion_leaf_count": expected_opinions,
    "selected_attack_leaf_count": expected_attacks,
    "n_scenarios_excluded": 0,
    "assignment_profile_count": expected_profiles,
}

for key, expected_value in expected.items():
    actual = checks[key]
    if actual != expected_value:
        raise SystemExit(f"{key} mismatch: expected {expected_value}, got {actual}")

if require_stage_08b:
    required_stage_manifests = [
        "01_create_scenarios/manifest.json",
        "01b_assign_exposure_network_positions/manifest.json",
        "02_assess_baseline_opinions/manifest.json",
        "02b_assess_network_exposure_opinions/manifest.json",
        "03_run_opinion_attacks/manifest.json",
        "04_assess_post_attack_opinions/manifest.json",
        "04b_assess_post_attack_network_exposure_opinions/manifest.json",
        "05_compute_effectivity_deltas/manifest.json",
        "06_construct_structural_equation_model/manifest.json",
        "07_generate_research_visuals/manifest.json",
        "08_generate_publication_assets/manifest.json",
        "08b_analyze_network_exposure_run/manifest.json",
    ]
    for relative in required_stage_manifests:
        path = stage_outputs / relative
        if not path.exists():
            raise SystemExit(f"Missing completed stage manifest: {path}")
    analysis_manifest = output_root / "network_exposure_analysis/reports/analysis_manifest.json"
    if not analysis_manifest.exists():
        raise SystemExit(f"Missing copied network exposure analysis manifest: {analysis_manifest}")

print(f"Stage output validation passed for {output_root}.")
PY
}

validate_imports
validate_static_config

case "${MODE}" in
    structural)
        OUTPUT_ROOT="${RUN_ROOT}/preflight/structural_full_factorial"
        run_pipeline "${OUTPUT_ROOT}" "run_2_structural" 100 3500 "${FULL_OPINION_LEAVES}" "${FULL_ATTACK_LEAVES}" "01b"
        validate_stage_outputs "${OUTPUT_ROOT}" 3500 100 7 5 0
        echo "Structural preflight complete: ${OUTPUT_ROOT}"
        ;;
    smoke)
        require_api_key
        OUTPUT_ROOT="${RUN_ROOT}/preflight/smoke"
        run_pipeline "${OUTPUT_ROOT}" "run_2_smoke" 30 120 "${SMOKE_OPINION_LEAVES}" "${SMOKE_ATTACK_LEAVES}" "08b"
        validate_stage_outputs "${OUTPUT_ROOT}" 120 30 2 2 1
        echo "Paid smoke run complete: ${OUTPUT_ROOT}"
        ;;
    full)
        require_api_key
        run_pipeline "${RUN_ROOT}" "run_2" 100 3500 "${FULL_OPINION_LEAVES}" "${FULL_ATTACK_LEAVES}" "08b"
        validate_stage_outputs "${RUN_ROOT}" 3500 100 7 5 1
        echo "Full production run complete: ${RUN_ROOT}"
        ;;
    *)
        usage
        echo "ERROR: unknown mode '${MODE}'." >&2
        exit 2
        ;;
esac
