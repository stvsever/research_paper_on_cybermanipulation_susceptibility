from __future__ import annotations

"""
ATTACK sample artifacts (examples + concise summary)
====================================================
The attack sample is the DISARM-red Plan/Prepare/Execute triplet set, drawn from
an EXTERNAL attack ontology (not this repository's attack ontology) and then
heuristically filtered down to opinion-manipulation vectors.  This script reads
the filtered file and writes, alongside it, two artifacts in the same spirit as
the PROFILE / OPINION sample sets:

  red_plan_prepare_execute_examples_3.json     three fully-resolved example triplets
  red_plan_prepare_execute_opinion_effect_filtered.summary.json
                                               condensed, standalone summary

Run by file path:
  python .../utils/build_attack_artifacts.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[6]
STAGE = PROJECT_ROOT / "src" / "backend" / "pipeline" / "separate" / "01_create_scenarios"
ATTACKS = STAGE / "samples" / "01_separated" / "attacks"
FILTERED = ATTACKS / "red_plan_prepare_execute_opinion_effect_filtered.json"

# Curated, human-verified example triplets (source_config_id → coherent formulation).
# Leaf ids are stable across the raw/filtered sets, so these resolve exactly.
CURATED = {
    42781: ("A platform-labeling dispute objective is paired with microtargeting through a "
            "persistent synthetic persona and then primed through a legal-filing evidence "
            "narrative. Opinion-relevant because it combines visibility pressure, personalized "
            "persona-mediated targeting, and a credibility/evidence frame."),
    58567: ("The configuration targets people with a Manichean good-vs-evil worldview, prepares "
            "AI-generated text content, and prepositions screenshots for later priming. It links "
            "susceptibility segmentation, automated message generation, and pre-exposure evidence "
            "staging."),
    14767: ("A review-score pressure objective is connected to citation-source narrative "
            "preparation and then primed through an astroturfed think-tank expert. Opinion-relevant "
            "because it combines visibility/reputation pressure, source framing, and fake expertise "
            "as a credibility cue."),
}
PHASES = ("Plan", "Prepare", "Execute")


def build_examples(data, leaf_by_id):
    by_src = {c["source_config_id"]: c for c in data["configurations"]}
    examples = []
    for sid, formulation in CURATED.items():
        c = by_src.get(sid)
        if c is None:
            continue
        ev = c["opinion_manipulation_evidence"]
        triplet = {}
        for ph in PHASES:
            lid = c["leaves"][ph]
            lf = leaf_by_id[lid]
            ps = ev["phase_scores"][ph]
            triplet[ph] = {
                "leaf_id": lid,
                "secondary": lf["secondary"],
                "label": lf["label"],
                "path": lf["path"],
                "signal_score": ps["signal_score"],
                "confidence": ps["confidence"],
                "criteria": ps["criteria"],
            }
        examples.append({
            "id": c["id"],
            "source_config_id": sid,
            "signal_total": ev["signal_total"],
            "inclusion_route": ev["inclusion_route"],
            "relevant_phase_count": ev["relevant_phase_count"],
            "mean_leaf_confidence": ev["mean_leaf_confidence"],
            "criteria": ev["criteria"],
            "coherent_formulation": formulation,
            "triplet": triplet,
        })
    return {
        "_meta": {
            "note": "Three fully-resolved example DISARM-red Plan/Prepare/Execute triplets "
                    "sampled from the final filtered opinion-effect set.",
            "ontology_source": "EXTERNAL DISARM-red attack ontology (not this repository's "
                               "attack ontology); raw leaf ids are stable for join-back.",
            "direction_convention": "Red perspective (attacker plan → prepare → execute).",
        },
        "examples": examples,
    }


def build_summary(data):
    man = data["manifest"]
    raw = man["source"]["raw_state_space"]
    coh = man["source"]["raw_coherence"]
    fm = man["filter_method"]
    s = man["summary"]
    diag = data["diagnostics"]

    ent = {}
    for ph, blk in diag["entropy_impact"]["leaf_level"].items():
        imp = blk["impact"]
        ent[ph] = {
            "raw_categories": blk["raw"]["n_available_categories"],
            "filtered_observed_categories": blk["filtered"]["n_observed_categories"],
            "category_coverage": round(blk["filtered"]["category_coverage"], 4),
            "relative_entropy_retained": round(imp["relative_entropy_retained"], 4),
            "pielou_evenness_filtered": round(blk["filtered"]["pielou_evenness_against_raw_pool"], 4),
        }

    crit = {r["criterion"]: r["configurations_with_criterion"]
            for r in diag["evidence_criteria_profile"]["rows"]}
    crit = dict(sorted(crit.items(), key=lambda kv: -kv[1]))

    return {
        "title": man["title"],
        "ontology_note": "Drawn from an EXTERNAL DISARM-red attack ontology — NOT this "
                         "repository's attack ontology. Raw leaf ids are stable for join-back.",
        "generated_at_utc": man["generated_at_utc"],
        "raw_state_space": {
            "per_phase_leaf_pool": raw["per_phase_leaf_pool"],
            "total_leaf_pool": raw["total_leaf_pool"],
            "full_cartesian_upper_bound": raw["full_cartesian_size"],
            "coherent_acceptance_rate": round(coh["coherent_acceptance_rate"], 4),
            "n_raw_configurations": raw["n_configurations_sampled"],
            "phases_sampled": list(PHASES),
            "excluded_phase": man["source"]["raw_sampling_contract"]["excluded_phase"],
        },
        "filter": {
            "objective": "retain plausible opinion-manipulation vectors; remove support-only "
                         "substrate / OPSEC / access / logistics leaves",
            "thresholds": fm["thresholds"],
            "entropy_use": fm["entropy_use"],
            "inclusion_route_counts": s["inclusion_route_counts"],
            "configuration_retention_rate": s["configuration_retention_rate"],
        },
        "filtered_set": {
            "n_configurations": s["configurations_remaining_after_filter"],
            "filtered_distinct_leaves": s["filtered_distinct_leaves"],
            "filtered_overall_leaf_coverage": round(s["filtered_overall_leaf_coverage"], 4),
            "leaves_with_opinion_manipulation_evidence": s["leaves_with_opinion_manipulation_evidence"],
            "included_signal_score_distribution": s["included_signal_score_distribution"],
        },
        "entropy_range_retention_per_phase": ent,
        "evidence_criteria_configuration_counts": crit,
    }


def main():
    data = json.loads(FILTERED.read_text())
    leaf_by_id = {l["id"]: l for l in data["leaf_catalog"]}

    examples = build_examples(data, leaf_by_id)
    summary = build_summary(data)

    ex_path = ATTACKS / "red_plan_prepare_execute_examples_3.json"
    sm_path = ATTACKS / "red_plan_prepare_execute_opinion_effect_filtered.summary.json"
    ex_path.write_text(json.dumps(examples, indent=2))
    sm_path.write_text(json.dumps(summary, indent=2))
    print(f"  wrote {ex_path.name}  ({len(examples['examples'])} examples)")
    print(f"  wrote {sm_path.name}")
    for e in examples["examples"]:
        print(f"    · cfg {e['id']} (src {e['source_config_id']}) signal {e['signal_total']} "
              f"route {e['inclusion_route']}")


if __name__ == "__main__":
    main()
