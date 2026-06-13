from __future__ import annotations

"""
Inject adversarial_direction metadata into a production OPINION ontology tree.

The production OPINION ontology in current design is a large taxonomy of policy
positions across many domains (healthcare, transport, science…), most of which
have no direct strategic relevance to the cyber-manipulation threat model.
This script assigns:

- direction +1 to leaves whose names match the "adversary wants HIGHER" pattern
  (unilateralism, sphere-of-influence acceptance, freedom-from-moderation, …)
- direction -1 to leaves whose names match the "adversary wants LOWER" pattern
  (alliance commitment, multilateralism, defence spending, fact-checking, …)
- direction 0 to all other leaves (excluded from primary effectivity scoring).

The rules are derived from the cognitive-warfare adversary frame documented
in src/backend/ontology/separate/test/OPINION/opinion.json _metadata.

Run as a script:
    python3 -m src.backend.utils.inject_opinion_directions \
        --in  src/backend/ontology/separate/production/OPINION/opinion.json \
        --out src/backend/ontology/separate/production/OPINION/opinion.json \
        --in-place
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


# Increase-the-score (+1): adversary benefits when this opinion goes UP.
INCREASE_PATTERNS: Tuple[str, ...] = (
    "unilateralism",
    "non_interventionism",
    "sphere_of_influence_acceptance",
    "great_power_appeasement_acceptance",
    "freedom_from_content_moderation",
    "government_distrust_of_elections",
    "border_enforcement_expansion",
    "deportation_enforcement",
    "detention_of_irregular_migrants",
    "migration_securitization_framing",
    "assimilationism",
    "strategic_autonomy_support",  # specifically when interpreted as anti-alliance
    "intelligence_oversight_strengthening_support",
    "freedom_of_speech_absolutism",
)

# Decrease-the-score (-1): adversary benefits when this opinion goes DOWN.
DECREASE_PATTERNS: Tuple[str, ...] = (
    "multilateralism",
    "international_cooperation",
    "international_law_compliance",
    "humanitarian_intervention",
    "sanctions_use",
    "democracy_promotion",
    "alliance_commitment",
    "defense_spending_increase",
    "conscription_support",
    "arms_exports_permissiveness",
    "domestic_security_powers_expansion",
    "border_security_militarization",
    "nuclear_deterrence",
    "cyber_defense_investment",
    "civil_defense_preparedness",
    "forward_deployment_host_nation",
    "critical_infrastructure_hardening",
    "counter_hybrid_warfare_capability",
    "defense_industrial_reshoring",
    "allied_cyber_incident_mutual_assistance",
    "energy_independence_from_authoritarian_suppliers",
    "support_for_contested_state_sovereignty",
    "overseas_development_aid",
    "platform_accountability_regulation",
    "counter_disinformation_agency",
    "foreign_information_manipulation_transparency_mandate",
    "independent_fact_checking_ecosystem",
    "algorithmic_transparency_requirement",
    "trust_in_mainstream_journalism",
    "public_service_broadcasting_funding",
    "synthetic_media_labelling_mandate",
    "civic_media_literacy_education",
    "support_for_democracy_as_best_system",
    "commitment_to_loser_consent",
    "tolerance_of_opposition_rights",
    "commitment_to_peaceful_transfer_of_power",
    "rejection_of_political_violence",
    "support_for_constitutional_constraints",
    "judicial_independence_protection",
    "anti_corruption_enforcement",
    "press_freedom_protection",
    "ai_regulation",
    "cybersecurity_regulation",
    "supply_chain_resilience_priority",
    "sovereign_wealth_fund_support",
    "refugee_acceptance_support",
    "asylum_process_expansion",
    "multiculturalism_support",
    "integration_funding_support",
    "humanitarian_corridor_support",
    "eu_common_asylum_system",
    "supranational_integration_support",
    "shared_fiscal_capacity_support",
    "common_defense_capacity_support",
    "regional_court_authority_support",
    "trust_in_government",
    "trust_in_parliament_or_legislature",
    "trust_in_judiciary",
    "trust_in_civil_service",
    "trust_in_science_and_experts",
    "trust_in_local_government",
    "trust_in_media",
)


def _classify_leaf_name(name: str) -> int:
    slug = name.lower()
    for pat in DECREASE_PATTERNS:
        if pat in slug:
            return -1
    for pat in INCREASE_PATTERNS:
        if pat in slug:
            return 1
    return 0


def _is_leaf_node(child: Any) -> bool:
    if not isinstance(child, dict):
        return True
    if not child:
        return True
    return all(not k[0].isupper() if k else True for k in child)


def _walk_inject(node: Dict[str, Any]) -> Tuple[Dict[str, Any], int, int, int]:
    """Recursively rewrite leaves into {adversarial_direction: ...} dicts."""
    pos = neg = neu = 0
    for key in list(node.keys()):
        if key.startswith("_") or not isinstance(key, str):
            continue
        child = node[key]
        if isinstance(child, dict):
            if _is_leaf_node(child):
                direction = _classify_leaf_name(key)
                node[key] = {"adversarial_direction": int(direction)}
                if direction > 0:
                    pos += 1
                elif direction < 0:
                    neg += 1
                else:
                    neu += 1
            else:
                _, p, n, u = _walk_inject(child)
                pos += p; neg += n; neu += u
    return node, pos, neg, neu


def inject_directions(opinion_tree: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Return a copy of the opinion tree with adversarial_direction injected.

    Also injects an _metadata block describing the convention if absent.
    """
    tree = copy.deepcopy(opinion_tree)
    tree, pos, neg, neu = _walk_inject(tree)
    meta = tree.setdefault("_metadata", {})
    meta.setdefault("schema_version", "v3-test-run-1")
    meta.setdefault("adversarial_operator_goal",
        "Maximise erosion of defence capacity, alliance cohesion, institutional security, and epistemic trust.")
    meta.setdefault("direction_convention", {
        "+1": "Adversary seeks to INCREASE this opinion score.",
        "-1": "Adversary seeks to DECREASE this opinion score.",
        "0": "Direction-neutral; included for ontological completeness, excluded from primary effectivity scoring."
    })
    return tree, {"positive_direction": pos, "negative_direction": neg, "neutral": neu}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject adversarial_direction into an OPINION ontology.")
    parser.add_argument("--in", dest="src", required=True)
    parser.add_argument("--out", dest="dst", required=True)
    parser.add_argument("--in-place", action="store_true")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if args.in_place and src != dst:
        raise SystemExit("--in-place requires --in == --out")

    tree = json.loads(src.read_text(encoding="utf-8"))
    rewritten, summary = inject_directions(tree)
    dst.write_text(json.dumps(rewritten, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(dst), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
