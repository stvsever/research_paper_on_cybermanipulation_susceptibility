#!/usr/bin/env python3
"""
Builder for the production ATTACK ontology (current design).

Design principles
-----------------
The production ATTACK ontology is a FULL hierarchical state space of
adversarial techniques targeting political-opinion formation, organised so
that any (profile × attack × opinion) tuple in the factorial design is a
well-defined cell that the simulator can instantiate.

Structural rules:
- Subtree keys are PascalCase_With_Underscores; leaves default to ``{}`` so
  ``flatten_leaf_paths`` enumerates every technique.
- Per-leaf metadata is added ONLY when a leaf carries a UNIQUE structural
  sampling constraint that is not derivable from its parent path. Most
  leaves carry no inline metadata.
- Subtree-wide sampling logic (capability prerequisites, opinion-domain
  compatibility, complexity tiering, etc.) is encoded ONCE in the top-level
  ``_compatibility_rules`` metanode using path-glob patterns, so a single
  rule covers a whole branch of the tree.
- The ontology does NOT encode any psychological-amplification hypotheses
  (e.g. "Neuroticism amplifies fear-appeal X"). Those are exactly the
  quantities the inferential layer estimates and must remain unbiased.

Output
------
This script writes a single JSON file:

    src/backend/ontology/separate/production/ATTACK/attack.json

Run with:

    python3 -m src.backend.ontology.builders.build_production_attack \
        src/backend/ontology/separate/production/ATTACK/attack.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Helper: turn a list of leaf names into an empty-dict mapping.
# ---------------------------------------------------------------------------
def L(*names: str) -> Dict[str, Dict[str, Any]]:
    """Return {name: {}} for each name. Leaves are pure state-space markers."""
    return {n: {} for n in names}


_LOCAL_METADATA_KEYS = frozenset({"adversarial_direction", "description", "notes", "examples"})


def _is_local_metadata_key(key: str) -> bool:
    return key.startswith("_") or key in _LOCAL_METADATA_KEYS or not key[0].isupper()


def _is_local_leaf(child: Any) -> bool:
    if not isinstance(child, dict):
        return True
    if not child:
        return True
    return all(_is_local_metadata_key(k) for k in child)


def _local_leaf_paths(tree: Dict[str, Any], prefix: List[str] | None = None) -> List[str]:
    prefix = prefix or []
    leaves: List[str] = []
    for key, child in tree.items():
        if _is_local_metadata_key(key):
            continue
        path = [*prefix, key]
        if _is_local_leaf(child):
            leaves.append(" > ".join(path))
        else:
            leaves.extend(_local_leaf_paths(child, path))
    return leaves


# ---------------------------------------------------------------------------
# Per-leaf metadata baker
# ---------------------------------------------------------------------------
# The ontology JSON ships with the FULL resolved combinatorial sampling
# metadata baked into every primary-axis leaf so that downstream sampling,
# scenario generation, prompt construction, and inferential code can read
# a single self-contained leaf body without re-evaluating rules.
#
# What is admissible per-leaf:
#   - complexity_tier, temporal_horizon, epistemic_target  (scalars)
#   - requires_capability                                  (operator-side caps)
#   - compatible_opinion_domains, incompatible_opinion_domains
#   - natural_companions, natural_predecessors, mutually_exclusive_with_paths
#   - scenario_role, is_classification_axis
#   - family (resolved primary family name)
#   - mechanism_summary (one-line auto-derived mechanism description)
#
# What is FORBIDDEN per-leaf (research targets, NOT structural inputs):
#   - Any psychological-amplification predicate
#   - amplifying_profile_features / dampening_profile_features
#   - effectiveness priors, susceptibility hypotheses
# ---------------------------------------------------------------------------


_PRIMARY_AXIS_PREFIX = (
    "Political_Opinion_Cybermanipulation_Ontology",
    "Primary_Axis",
    "Attack_Family",
)
_SECONDARY_AXIS_PREFIX = (
    "Political_Opinion_Cybermanipulation_Ontology",
    "Secondary_Axes",
)
_FAMILY_FALLBACK_NAMES = frozenset(
    {
        "Political_Opinion_Cybermanipulation_Ontology",
        "Primary_Axis",
        "Secondary_Axes",
        "Attack_Family",
    }
)


def _split_path(path: str) -> List[str]:
    return [p.strip() for p in path.split(" > ") if p.strip()]


def _split_pattern(pattern: str) -> List[str]:
    return [p.strip() for p in pattern.replace("**", " ** ").split(" ** ")]


def _path_pattern_matches(pattern: str, segments: List[str]) -> bool:
    parts = [p for p in _split_pattern(pattern) if p]
    if not parts:
        return True
    cursor = 0
    for needle in parts:
        sub = _split_path(needle)
        if not sub:
            continue
        found = False
        while cursor + len(sub) <= len(segments):
            if segments[cursor : cursor + len(sub)] == sub:
                cursor += len(sub)
                found = True
                break
            cursor += 1
        if not found:
            return False
    return True


def _resolve_family(segments: List[str]) -> str:
    for seg in segments:
        if seg not in _FAMILY_FALLBACK_NAMES:
            return seg
    return ""


_FAMILY_DEFAULT_TIER = {
    "Intelligence_Preparation_And_Vulnerability_Analysis": "T2_campaign",
    "Source_Identity_And_Legitimacy_Manipulation": "T2_campaign",
    "Claim_Frame_And_Narrative_Manipulation": "T1_atomic",
    "Ai_Generated_Synthetic_Media_And_Content": "T3_synthetic",
    "Multi_Agent_Adversarial_Architecture": "T4_orchestrated",
    "Targeting_Delivery_And_Discoverability_Optimization": "T2_campaign",
    "Amplification_Visibility_And_Attention_Manipulation": "T2_campaign",
    "Social_Proof_Network_And_Community_Manipulation": "T2_campaign",
    "Platform_Information_Environment_And_Media_System_Shaping": "T2_campaign",
    "Cyber_Enabled_Compromise_Coercion_And_Disruption": "T3_synthetic",
    "Behavioral_Conversion_Mobilization_Suppression_And_Radicalization": "T2_campaign",
    "Cognitive_Infrastructure_And_Epistemic_Subversion": "T2_campaign",
    "Operational_Security_Evasion_Persistence_And_Reconstitution": "T3_synthetic",
    "Measurement_Experimentation_And_Adaptive_Learning": "T2_campaign",
    "Narrative_Infrastructure_And_Ecosystem_Capture": "T5_sustained",
    "Economic_And_Regulatory_Pressure_Weaponisation": "T3_synthetic",
    "Insider_Threat_Facilitation_And_Human_Recruitment": "T4_orchestrated",
    "Gamification_And_Unwitting_Participant_Mobilisation": "T4_orchestrated",
}

_FAMILY_DEFAULT_HORIZON = {
    "Intelligence_Preparation_And_Vulnerability_Analysis": "weeks",
    "Source_Identity_And_Legitimacy_Manipulation": "weeks",
    "Claim_Frame_And_Narrative_Manipulation": "hours",
    "Ai_Generated_Synthetic_Media_And_Content": "days",
    "Multi_Agent_Adversarial_Architecture": "weeks",
    "Targeting_Delivery_And_Discoverability_Optimization": "days",
    "Amplification_Visibility_And_Attention_Manipulation": "days",
    "Social_Proof_Network_And_Community_Manipulation": "weeks",
    "Platform_Information_Environment_And_Media_System_Shaping": "weeks",
    "Cyber_Enabled_Compromise_Coercion_And_Disruption": "days",
    "Behavioral_Conversion_Mobilization_Suppression_And_Radicalization": "weeks",
    "Cognitive_Infrastructure_And_Epistemic_Subversion": "months",
    "Operational_Security_Evasion_Persistence_And_Reconstitution": "months",
    "Measurement_Experimentation_And_Adaptive_Learning": "weeks",
    "Narrative_Infrastructure_And_Ecosystem_Capture": "years",
    "Economic_And_Regulatory_Pressure_Weaponisation": "months",
    "Insider_Threat_Facilitation_And_Human_Recruitment": "months",
    "Gamification_And_Unwitting_Participant_Mobilisation": "weeks",
}

_FAMILY_DEFAULT_EPISTEMIC_TARGET = {
    "Intelligence_Preparation_And_Vulnerability_Analysis": "evaluative_attitude",
    "Source_Identity_And_Legitimacy_Manipulation": "trust_dimension",
    "Claim_Frame_And_Narrative_Manipulation": "factual_belief",
    "Ai_Generated_Synthetic_Media_And_Content": "factual_belief",
    "Multi_Agent_Adversarial_Architecture": "evaluative_attitude",
    "Targeting_Delivery_And_Discoverability_Optimization": "information_access",
    "Amplification_Visibility_And_Attention_Manipulation": "information_access",
    "Social_Proof_Network_And_Community_Manipulation": "evaluative_attitude",
    "Platform_Information_Environment_And_Media_System_Shaping": "epistemic_routine",
    "Cyber_Enabled_Compromise_Coercion_And_Disruption": "behavioural_intention",
    "Behavioral_Conversion_Mobilization_Suppression_And_Radicalization": "behavioural_intention",
    "Cognitive_Infrastructure_And_Epistemic_Subversion": "epistemic_routine",
    "Operational_Security_Evasion_Persistence_And_Reconstitution": "epistemic_routine",
    "Measurement_Experimentation_And_Adaptive_Learning": "epistemic_routine",
    "Narrative_Infrastructure_And_Ecosystem_Capture": "semantic_meaning",
    "Economic_And_Regulatory_Pressure_Weaponisation": "evaluative_attitude",
    "Insider_Threat_Facilitation_And_Human_Recruitment": "behavioural_intention",
    "Gamification_And_Unwitting_Participant_Mobilisation": "behavioural_intention",
}


_HUMAN_PHRASE_OVERRIDES = {
    "Ai": "AI",
    "Llm": "LLM",
    "Faq": "FAQ",
    "Url": "URL",
    "Seo": "SEO",
    "Ux": "UX",
    "Ar": "AR",
    "Vr": "VR",
    "Vpn": "VPN",
    "Cdn": "CDN",
    "Pii": "PII",
    "Sms": "SMS",
    "Os": "OS",
    "Tos": "ToS",
    "Eu": "EU",
    "Us": "US",
    "Uk": "UK",
    "Nato": "NATO",
    "Mitre": "MITRE",
    "Disarm": "DISARM",
    "Atlas": "ATLAS",
    "Fimi": "FIMI",
    "Ttp": "TTP",
}


def _humanise(label: str) -> str:
    """Convert PascalCase_With_Underscores into a lowercase phrase with sane acronyms."""
    parts = [p for p in label.replace("__", "_").split("_") if p]
    out: List[str] = []
    for p in parts:
        if p in _HUMAN_PHRASE_OVERRIDES:
            out.append(_HUMAN_PHRASE_OVERRIDES[p])
        else:
            out.append(p.lower())
    return " ".join(out)


def _mechanism_summary(family: str, segments: List[str]) -> str:
    """One-line mechanism description auto-derived from family + leaf label.

    Pattern: "<family-stem>: <leaf phrase> [in <parent phrase>]"
    """
    if not segments:
        return ""
    leaf = segments[-1]
    parent = segments[-2] if len(segments) >= 2 else ""
    family_phrase = _humanise(family) if family else ""
    leaf_phrase = _humanise(leaf)
    parent_phrase = _humanise(parent) if parent and parent != family else ""
    if family_phrase and parent_phrase and parent_phrase != family_phrase:
        return f"{family_phrase}: {leaf_phrase} (within {parent_phrase})"
    if family_phrase:
        return f"{family_phrase}: {leaf_phrase}"
    return leaf_phrase


def _resolve_leaf_metadata(
    rules: List[Dict[str, Any]],
    leaf_path: str,
) -> Dict[str, Any]:
    """Apply rules to a leaf path; return a dict of resolved attributes.

    Scalar attrs use last-match-wins; list attrs are UNION-merged.
    """
    segments = _split_path(leaf_path)
    scalar: Dict[str, Any] = {}
    list_attrs: Dict[str, List[str]] = {
        "requires_capability": [],
        "natural_companions": [],
        "natural_predecessors": [],
        "mutually_exclusive_with_paths": [],
    }
    for rule in rules:
        patterns = rule.get("applies_to_attack_paths", [])
        if not isinstance(patterns, list):
            continue
        if not any(_path_pattern_matches(str(p), segments) for p in patterns):
            continue
        for k in ("complexity_tier", "temporal_horizon", "epistemic_target"):
            if k in rule:
                scalar[k] = rule[k]
        for k in ("scenario_role", "is_classification_axis"):
            if k in rule:
                scalar[k] = rule[k]
        for k in ("compatible_opinion_domains", "incompatible_opinion_domains"):
            if k in rule:
                scalar[k] = list(rule[k]) if rule[k] else []
        for k in (
            "requires_capability",
            "natural_companions",
            "natural_predecessors",
            "mutually_exclusive_with_paths",
        ):
            if k in rule:
                values = rule[k] if isinstance(rule[k], list) else [rule[k]]
                for v in values:
                    if v not in list_attrs[k]:
                        list_attrs[k].append(v)
    return {**scalar, **list_attrs}


def _bake_per_leaf_metadata(tree: Dict[str, Any]) -> int:
    """Walk every leaf and bake resolved combinatorial metadata in-place.

    Returns the number of leaves enriched.
    """
    rules = (
        tree.get("_compatibility_rules", {}).get("rules", [])
        if isinstance(tree.get("_compatibility_rules"), dict)
        else []
    )

    count = 0
    for leaf_path in _local_leaf_paths(tree):
        segments = _split_path(leaf_path)
        node: Any = tree
        for seg in segments:
            node = node[seg]
        if not isinstance(node, dict):
            continue

        family = _resolve_family(segments)
        resolved = _resolve_leaf_metadata(rules, leaf_path)

        # Default tier / horizon / epistemic target from family if rule did not set
        complexity_tier = resolved.get("complexity_tier") or _FAMILY_DEFAULT_TIER.get(
            family, "T2_campaign"
        )
        temporal_horizon = resolved.get("temporal_horizon") or _FAMILY_DEFAULT_HORIZON.get(
            family, "days"
        )
        epistemic_target = resolved.get("epistemic_target") or _FAMILY_DEFAULT_EPISTEMIC_TARGET.get(
            family, "evaluative_attitude"
        )

        # Bake resolved attrs into the leaf body (lowercase keys -> recognised as metadata).
        node["family"] = family
        node["complexity_tier"] = complexity_tier
        node["temporal_horizon"] = temporal_horizon
        node["epistemic_target"] = epistemic_target

        node["requires_capability"] = list(resolved.get("requires_capability") or [])

        if resolved.get("compatible_opinion_domains"):
            node["compatible_opinion_domains"] = list(resolved["compatible_opinion_domains"])
        if resolved.get("incompatible_opinion_domains"):
            node["incompatible_opinion_domains"] = list(resolved["incompatible_opinion_domains"])

        if resolved.get("natural_companions"):
            node["natural_companions"] = list(resolved["natural_companions"])
        if resolved.get("natural_predecessors"):
            node["natural_predecessors"] = list(resolved["natural_predecessors"])
        if resolved.get("mutually_exclusive_with_paths"):
            node["mutually_exclusive_with_paths"] = list(resolved["mutually_exclusive_with_paths"])

        node["scenario_role"] = resolved.get("scenario_role") or "target_exposure"
        node["is_classification_axis"] = bool(resolved.get("is_classification_axis", False))

        node["mechanism_summary"] = _mechanism_summary(family, segments)
        count += 1
    return count


# ---------------------------------------------------------------------------
# 1. INTELLIGENCE PREPARATION AND VULNERABILITY ANALYSIS
# ---------------------------------------------------------------------------
INTELLIGENCE_PREPARATION = {
    "Audience_Segmentation_And_Population_Mapping": {
        "Demographic_Segmentation": {
            "Age_Cohort_Segmentation": L(
                "Adolescent_And_Future_Voter_Segmentation",
                "First_Time_Voter_Segmentation",
                "University_Age_Segmentation",
                "Mid_Career_Household_Segmentation",
                "Pre_Retirement_Segmentation",
                "Retirement_Age_Segmentation",
            ),
            "Gender_And_Family_Segmentation": L(
                "Single_Adult_Segmentation",
                "Couple_Without_Children_Segmentation",
                "Parent_With_Young_Children_Segmentation",
                "Parent_With_Adolescents_Segmentation",
                "Empty_Nest_Segmentation",
                "Caregiver_Household_Segmentation",
            ),
            "Socioeconomic_Position_Segmentation": L(
                "Precarious_Worker_Segmentation",
                "Gig_Economy_Worker_Segmentation",
                "Small_Business_Owner_Segmentation",
                "Public_Sector_Dependent_Segmentation",
                "Deindustrialised_Region_Segmentation",
                "Indebted_Household_Segmentation",
                "High_Net_Worth_Segmentation",
                "Renter_Versus_Owner_Segmentation",
            ),
            "Education_And_Information_Literacy_Segmentation": L(
                "Low_Formal_Education_Segmentation",
                "Vocationally_Trained_Segmentation",
                "Credentialed_Professional_Segmentation",
                "Digitally_Novice_Segmentation",
                "High_Information_Low_Trust_Segmentation",
                "Procedurally_Inattentive_Segmentation",
            ),
            "Settlement_Pattern_Segmentation": L(
                "Urban_Core_Resident_Segmentation",
                "Suburban_Commuter_Segmentation",
                "Peri_Urban_Resident_Segmentation",
                "Small_Town_Resident_Segmentation",
                "Remote_Rural_Segmentation",
                "Borderland_Population_Segmentation",
                "Diaspora_Cluster_Segmentation",
            ),
            "Linguistic_And_Ethnocultural_Segmentation": L(
                "Majority_Language_Segmentation",
                "Minority_Language_Segmentation",
                "Bilingual_Code_Switching_Segmentation",
                "Recent_Migrant_Segmentation",
                "Second_Generation_Migrant_Segmentation",
                "Indigenous_Community_Segmentation",
            ),
        },
        "Psychographic_Segmentation": {
            "Moral_Foundation_Segmentation": L(
                "Harm_Care_Dominant_Profiles",
                "Fairness_Reciprocity_Dominant_Profiles",
                "Loyalty_Betrayal_Dominant_Profiles",
                "Authority_Subversion_Dominant_Profiles",
                "Sanctity_Degradation_Dominant_Profiles",
                "Liberty_Oppression_Dominant_Profiles",
            ),
            "Personality_Style_Segmentation": L(
                "High_Threat_Sensitivity_Profiles",
                "High_Need_For_Closure_Profiles",
                "High_Novelty_Seeking_Profiles",
                "Reactive_Antagonistic_Profiles",
                "Socially_Conforming_Profiles",
                "High_Reactance_Profiles",
                "High_Trust_Profiles",
            ),
            "Efficacy_And_Control_Segmentation": L(
                "Politically_Efficacious_Profiles",
                "Powerless_Fatalistic_Profiles",
                "Status_Anxious_Profiles",
                "Humiliation_Sensitive_Profiles",
                "Betrayal_Sensitive_Profiles",
                "Locus_Of_Control_Internal_Profiles",
                "Locus_Of_Control_External_Profiles",
            ),
            "Media_Cognition_Segmentation": L(
                "Intuitive_Fast_Processing_Profiles",
                "Conspiratorial_Pattern_Seeker_Profiles",
                "Sceptical_Contrarian_Profiles",
                "Institutionally_Deferential_Profiles",
                "Parasocially_Guided_Profiles",
                "Deliberative_Reasoner_Profiles",
                "Emotion_First_Reasoner_Profiles",
            ),
            "Identity_Salience_Segmentation": L(
                "National_Identity_Salient_Profiles",
                "Religious_Identity_Salient_Profiles",
                "Class_Identity_Salient_Profiles",
                "Regional_Identity_Salient_Profiles",
                "Generational_Identity_Salient_Profiles",
                "Professional_Identity_Salient_Profiles",
            ),
        },
        "Ideological_Segmentation": {
            "Partisan_Alignment_Segmentation": L(
                "Core_Left_Identifiers",
                "Core_Right_Identifiers",
                "Centrist_Identifiers",
                "Issue_Cross_Pressured_Voters",
                "Anti_Establishment_Voters",
                "Non_Aligned_Protest_Voters",
                "Single_Issue_Voters",
            ),
            "Issue_Commitment_Segmentation": L(
                "Single_Issue_Committed",
                "Weakly_Held_Issue_Preferences",
                "Symbolically_Committed_Groups",
                "Ideologically_Consistent_Groups",
                "Coalition_Tension_Groups",
            ),
            "Extremity_Segmentation": L(
                "Moderate_Pragmatic_Profiles",
                "Polarised_Partisan_Profiles",
                "Movement_Edge_Profiles",
                "Maximalist_Profiles",
                "Abstentionist_Rejection_Profiles",
            ),
        },
        "Identity_Cleavage_Mapping": {
            "Ethnoreligious_Cleavages": L(
                "Majority_Minority_Tension_Lines",
                "Sectarian_Fault_Lines",
                "Language_Conflict_Lines",
                "Migrant_Native_Fault_Lines",
            ),
            "Status_And_Class_Cleavages": L(
                "Metropolitan_Periphery_Divides",
                "Credentialed_Noncredentialed_Divides",
                "Intergenerational_Equity_Divides",
                "Public_Private_Sector_Divides",
                "Native_Foreign_Born_Divides",
            ),
            "Values_And_Morality_Cleavages": L(
                "Traditionalist_Progressive_Divides",
                "Secular_Religious_Divides",
                "Order_Liberty_Divides",
                "Globalist_Sovereigntist_Divides",
                "Cosmopolitan_Communitarian_Divides",
            ),
        },
    },
    "Grievance_And_Trigger_Discovery": {
        "Economic_Grievance_Mapping": L(
            "Inflation_And_Cost_Of_Living_Grievances",
            "Housing_Affordability_Grievances",
            "Taxation_And_Redistribution_Grievances",
            "Unemployment_And_Precarity_Grievances",
            "Regional_Inequality_Grievances",
            "Wage_Stagnation_Grievances",
            "Pension_Security_Grievances",
        ),
        "Cultural_And_Identity_Grievance_Mapping": L(
            "Migration_And_Demographic_Change_Anxiety",
            "Language_And_Symbolic_Status_Conflict",
            "Moral_Norm_And_Family_Change_Anxiety",
            "Secularisation_Or_Desacralization_Grievances",
            "Historical_Recognition_And_Memory_Grievances",
            "Cultural_Sovereignty_Grievances",
        ),
        "Security_And_Order_Grievance_Mapping": L(
            "Crime_And_Disorder_Fears",
            "Terrorism_And_Extremism_Fears",
            "War_And_Geopolitical_Insecurity",
            "Border_And_Sovereignty_Insecurity",
            "Institutional_Capacity_Failure_Fears",
            "Cyber_Insecurity_Fears",
            "Critical_Infrastructure_Fragility_Fears",
        ),
        "Institutional_Mistrust_Mapping": L(
            "Electoral_Integrity_Distrust",
            "Judicial_Fairness_Distrust",
            "Media_Honesty_Distrust",
            "Scientific_And_Expert_Distrust",
            "Administrative_Competence_Distrust",
            "Police_Distrust",
            "Health_System_Distrust",
        ),
        "Event_Trigger_Monitoring": {
            "Electoral_Triggers": L(
                "Candidate_Scandals",
                "Debate_Moments",
                "Registration_Deadlines",
                "Early_Voting_Windows",
                "Certification_Conflicts",
                "Polling_Day_Disruptions",
            ),
            "Policy_And_Governance_Triggers": L(
                "Budget_Announcements",
                "Court_Rulings",
                "Legislative_Votes",
                "Regulatory_Changes",
                "Public_Service_Failures",
                "Treaty_Negotiations",
            ),
            "Crisis_And_Symbolic_Triggers": L(
                "Violent_Incidents",
                "Protests_And_Demonstrations",
                "Natural_Disasters",
                "Anniversaries_And_Memorials",
                "Identity_Symbol_Controversies",
                "Sporting_Or_Cultural_Conflicts",
                "Public_Health_Emergencies",
            ),
            "Geopolitical_Triggers": L(
                "Military_Incidents",
                "Cyber_Incidents",
                "Diplomatic_Ruptures",
                "Migration_Surges",
                "Energy_Or_Supply_Shocks",
            ),
        },
    },
    "Media_Ecology_And_Attention_Mapping": {
        "Platform_Habituation_Modeling": L(
            "Short_Video_Platform_Dependence",
            "Mainstream_Social_Network_Dependence",
            "Forum_And_Imageboard_Dependence",
            "Messaging_App_Dependence",
            "Streaming_And_Live_Chat_Dependence",
            "Podcast_Listening_Dependence",
            "Ai_Chat_Assistant_Dependence",
        ),
        "Media_Diet_Profiling": L(
            "Legacy_News_Reliance",
            "Partisan_Commentary_Reliance",
            "Podcast_And_Longform_Reliance",
            "Creator_Led_News_Reliance",
            "Local_Rumor_And_Peer_Forward_Reliance",
            "Newsletter_Reliance",
            "Algorithmic_Feed_Reliance",
            "Search_First_News_Reliance",
        ),
        "Cross_Platform_Traffic_Mapping": L(
            "Fringe_To_Mainstream_Flows",
            "Broadcast_To_Social_Flows",
            "Messaging_To_Public_Platform_Flows",
            "Diaspora_To_Domestic_Flows",
            "Ai_Summary_To_Source_Flows",
            "Search_Result_To_Article_Flows",
            "Recommendation_To_Article_Flows",
        ),
        "Temporal_Attention_Pattern_Mapping": L(
            "Morning_Commute_Windows",
            "Workday_Microbreak_Windows",
            "Evening_Prime_Attention_Windows",
            "Late_Night_Low_Scrutiny_Windows",
            "Weekend_Identity_Engagement_Windows",
            "Crisis_Spike_Windows",
        ),
        "Retrieval_Pathway_Mapping": {
            "Search_And_Query_Pathways": L(
                "Issue_Search_Terms",
                "Candidate_Search_Terms",
                "Scandal_Search_Terms",
                "Procedure_Help_Queries",
                "Comparison_Queries",
                "Translation_Queries",
            ),
            "Recommendation_Pathways": L(
                "Watch_Next_Paths",
                "Follow_Recommended_Account_Paths",
                "Topic_Suggestion_Paths",
                "Quote_Post_Recirculation_Paths",
                "Ai_Answer_Surface_Paths",
                "Voice_Assistant_Answer_Paths",
            ),
        },
    },
    "Broker_And_Influencer_Discovery": {
        "Elite_Broker_Mapping": L(
            "Politicians_And_Advisers",
            "Party_Staff_And_Field_Directors",
            "Major_Donors_And_Fixers",
            "Journalists_And_Assignment_Editors",
            "Movement_Entrepreneurs",
            "Think_Tank_Affiliates",
            "Retired_Officials",
        ),
        "Community_Gatekeeper_Mapping": L(
            "Moderators_And_Admins",
            "Clergy_And_Moral_Authorities",
            "Local_Notables",
            "Union_Or_Association_Leaders",
            "Neighbourhood_Information_Hubs",
            "School_And_PTA_Leaders",
            "Sports_Club_Leaders",
        ),
        "Microinfluencer_Recruitability_Mapping": {
            "Financially_Recruitable_Nodes": L(
                "Cash_Strained_Creators",
                "Engagement_Seeking_Commentators",
                "Small_Local_Publishers",
                "Affiliate_Revenue_Seekers",
            ),
            "Ideologically_Recruitable_Nodes": L(
                "Cause_Committed_Activists",
                "Single_Issue_Campaigners",
                "Status_Resenting_Commentators",
                "Anti_System_Streamers",
            ),
            "Reputationally_Recruitable_Nodes": L(
                "Attention_Seeking_Personalities",
                "Humiliated_Former_Insiders",
                "Aggrieved_Local_Figures",
                "Belonging_Seeking_Microcelebrities",
            ),
        },
        "Bridge_Node_Mapping": L(
            "Cross_Partisan_Bridge_Nodes",
            "Diaspora_Bridge_Nodes",
            "Subculture_To_Mainstream_Bridge_Nodes",
            "Journalistic_Bridge_Nodes",
            "Issue_To_Identity_Bridge_Nodes",
            "Religious_To_Civic_Bridge_Nodes",
        ),
        "Diaspora_And_Transnational_Broker_Mapping": L(
            "Homeland_Media_Intermediaries",
            "Transnational_Religious_Figures",
            "Expatriate_Activist_Brokers",
            "Cross_Border_Business_Network_Brokers",
            "Foreign_Language_Creator_Brokers",
        ),
    },
    "Susceptibility_And_Elasticity_Estimation": {
        "Trust_Vulnerability_Scoring": L(
            "Anti_Institutional_Baseline_Profiles",
            "Recently_Disappointed_Supporter_Profiles",
            "Procedural_Confusion_Prone_Profiles",
            "Authority_Reversal_Prone_Profiles",
        ),
        "Emotion_Responsiveness_Scoring": L(
            "Fear_Responsive_Profiles",
            "Anger_Responsive_Profiles",
            "Disgust_Responsive_Profiles",
            "Humiliation_Responsive_Profiles",
            "Nostalgia_Responsive_Profiles",
            "Redemption_Responsive_Profiles",
            "Hope_Responsive_Profiles",
        ),
        "Conspiracy_Affinity_Scoring": L(
            "Hidden_Hand_Explanation_Seekers",
            "Betrayal_Script_Receptive_Profiles",
            "Elite_Capture_Belief_Prone_Profiles",
            "Apocalyptic_Interpretation_Prone_Profiles",
        ),
        "Participation_Elasticity_Scoring": L(
            "Mobilizable_Profiles",
            "Demobilizable_Profiles",
            "Misdirection_Prone_Profiles",
            "Procedural_Dropoff_Profiles",
            "Flash_Mobilization_Profiles",
        ),
        "Elite_Pressure_Sensitivity_Scoring": L(
            "Scandal_Sensitive_Elites",
            "Media_Cycle_Sensitive_Elites",
            "Donor_Pressure_Sensitive_Elites",
            "Legal_Risk_Sensitive_Elites",
            "Constituency_Backlash_Sensitive_Elites",
        ),
    },
    "Data_Acquisition_And_Feature_Enrichment": {
        "Open_Source_Profile_Enrichment": L(
            "Public_Post_History_Enrichment",
            "Network_Affiliation_Enrichment",
            "Occupation_And_Status_Enrichment",
            "Issue_Signal_Enrichment",
            "Event_Attendance_Enrichment",
            "Search_History_Adjacent_Enrichment",
        ),
        "Illegal_Or_Covert_Data_Acquisition": L(
            "Breach_Data_Matching",
            "Shadow_Data_Broker_Use",
            "Unauthorised_Scraping",
            "Credential_Harvesting_For_Targeting",
            "Covert_Contact_Graph_Collection",
            "Stalkerware_Derived_Data_Use",
        ),
        "Survey_And_Poll_Exploitation": L(
            "Microsegment_Questionnaire_Mining",
            "Issue_Salience_Inference",
            "Swing_Probability_Inference",
            "Hidden_Attitude_Proxy_Inference",
            "Turnout_Propensity_Inference",
        ),
        "Engagement_Signal_Harvesting": L(
            "Click_And_Dwell_Signal_Harvesting",
            "Share_And_Forward_Signal_Harvesting",
            "Comment_Sentiment_Harvesting",
            "Outrage_Trigger_Harvesting",
            "Endorser_Interaction_Harvesting",
            "Reaction_Emoji_Pattern_Harvesting",
        ),
        "Location_And_Context_Signal_Harvesting": L(
            "Geofenced_Event_Signal_Harvesting",
            "Commuting_Context_Signal_Harvesting",
            "Regional_Crisis_Signal_Harvesting",
            "Weather_And_Disruption_Context_Harvesting",
            "Public_Calendar_Context_Harvesting",
        ),
        "Behavioral_Biometric_Profiling": L(
            "Typing_Cadence_Profiling",
            "Cursor_Path_Profiling",
            "Voice_Style_Profiling",
            "Image_Aesthetic_Preference_Profiling",
            "Reading_Speed_Profiling",
        ),
    },
    # SOURCE: NATO ACT Cognitive Warfare Concept (2024), NCSC AI cyber-threat
    # assessment (2024), EEAS FIMI infrastructure exposure report (2025).
    "Systemic_Vulnerability_Mapping": L(
        "Institutional_Process_Dependency_Mapping",
        "Platform_Governance_Dependency_Mapping",
        "Election_Administration_Dependency_Mapping",
        "Information_Supply_Chain_Dependency_Mapping",
        "Crisis_Response_Communication_Dependency_Mapping",
    ),
}


# ---------------------------------------------------------------------------
# 2. SOURCE IDENTITY AND LEGITIMACY MANIPULATION
# ---------------------------------------------------------------------------
SOURCE_IDENTITY = {
    "Persona_Fabrication": {
        "Sockpuppet_Personas": {
            "Ordinary_Citizen_Personas": L(
                "Local_Parent_Personas",
                "Concerned_Taxpayer_Personas",
                "Disillusioned_Former_Supporter_Personas",
                "Apolitical_Neighbour_Personas",
                "Working_Class_Voter_Personas",
                "Suburban_Professional_Personas",
            ),
            "Cause_Bound_Personas": L(
                "Issue_Activist_Personas",
                "Grassroots_Volunteer_Personas",
                "Community_Defender_Personas",
                "Independent_Watchdog_Personas",
                "Civic_Reformer_Personas",
            ),
        },
        "Legend_Built_Personas": {
            "Historically_Layered_Personas": L(
                "Long_Running_Comment_History_Personas",
                "Local_Event_Memory_Personas",
                "Consistent_Hobby_Identity_Personas",
                "Neighbourhood_Embedded_Personas",
            ),
            "Relationship_Embedded_Personas": L(
                "Friend_Of_Friends_Personas",
                "Community_Group_Member_Personas",
                "Former_Campaign_Insider_Personas",
                "Trusted_Peer_Personas",
                "Reciprocity_Bound_Personas",
            ),
        },
        "Synthetic_Profile_Generation": {
            "Synthetic_Visual_Identity_Generation": L(
                "Ai_Face_Generation",
                "Age_Progressed_Identity_Generation",
                "Occupation_Signalling_Visual_Generation",
                "Lifestyle_Backdrop_Generation",
                "Location_Consistent_Visual_Generation",
            ),
            "Synthetic_Biographical_Generation": L(
                "Plausible_Resume_Generation",
                "Localised_Hometown_Generation",
                "Family_Role_Generation",
                "Values_Signalling_Bio_Generation",
                "Education_History_Generation",
            ),
            "Synthetic_Posting_History_Generation": L(
                "Backdated_Post_Generation",
                "Consistent_Voice_Post_Generation",
                "Local_Event_Reference_Post_Generation",
                "Topical_Diversity_Post_Generation",
            ),
        },
        "False_Expert_Personas": {
            "Credentialed_Domain_Personas": L(
                "Fake_Academic_Personas",
                "Fake_Journalist_Personas",
                "Fake_Pollster_Personas",
                "Fake_Security_Analyst_Personas",
                "Fake_Public_Health_Personas",
                "Fake_Intelligence_Analyst_Personas",
            ),
            "Procedural_Authority_Personas": L(
                "Fake_Election_Worker_Personas",
                "Fake_Civil_Servant_Personas",
                "Fake_Legal_Expert_Personas",
                "Fake_Public_Health_Authority_Personas",
                "Fake_Statistical_Authority_Personas",
                "Fake_Auditor_Personas",
            ),
        },
        "Issue_Native_Personas": {
            "Identity_Claim_Personas": L(
                "Minority_Group_Member_Personas",
                "Veteran_Personas",
                "Farmer_Personas",
                "Teacher_Personas",
                "Student_Personas",
                "Healthcare_Worker_Personas",
                "First_Responder_Personas",
            ),
            "Lived_Experience_Personas": L(
                "Victim_Testimony_Personas",
                "Whistleblower_Personas",
                "Ordinary_Parent_Testimony_Personas",
                "Former_Insider_Confessional_Personas",
                "Refugee_Testimony_Personas",
            ),
        },
    },
    "Impersonation_And_Mimicry": {
        "Journalist_And_Outlet_Impersonation": {
            "Newsroom_Brand_Impersonation": L(
                "Lookalike_News_Handles",
                "Lookalike_News_Domains",
                "Fake_Breaking_News_Formats",
                "Fake_Live_Blog_Formats",
                "Cloned_Headline_Card_Formats",
            ),
            "Individual_Reporter_Impersonation": L(
                "Beat_Reporter_Impersonation",
                "Local_Anchor_Impersonation",
                "Investigative_Journalist_Impersonation",
                "Fact_Checker_Impersonation",
                "Foreign_Correspondent_Impersonation",
            ),
        },
        "Candidate_Party_And_Campaign_Impersonation": {
            "Official_Campaign_Surface_Impersonation": L(
                "Candidate_Account_Impersonation",
                "Party_Branch_Impersonation",
                "Campaign_Volunteer_Group_Impersonation",
                "Donation_Page_Impersonation",
                "Endorsement_Letter_Impersonation",
            ),
            "Supporter_Ecology_Impersonation": L(
                "Fan_Page_Impersonation",
                "Grassroots_Coalition_Impersonation",
                "Defector_Supporter_Impersonation",
                "Local_Canvasser_Impersonation",
                "Volunteer_Network_Impersonation",
            ),
        },
        "Institutional_Impersonation": {
            "Election_And_Civic_Service_Impersonation": L(
                "Election_Office_Impersonation",
                "Municipal_Service_Impersonation",
                "Regulator_Impersonation",
                "Complaint_Portal_Impersonation",
                "Census_Bureau_Impersonation",
            ),
            "Security_And_Emergency_Impersonation": L(
                "Police_Notice_Impersonation",
                "Emergency_Alert_Impersonation",
                "Border_Or_Customs_Impersonation",
                "Public_Order_Warning_Impersonation",
                "Civil_Defence_Notice_Impersonation",
            ),
            "Health_And_Welfare_Impersonation": L(
                "Public_Health_Authority_Impersonation",
                "Hospital_Communication_Impersonation",
                "Benefits_Office_Impersonation",
                "Vaccination_Schedule_Impersonation",
            ),
        },
        "Grassroots_Group_Impersonation": {
            "Community_Group_Impersonation": L(
                "Neighbourhood_Forum_Impersonation",
                "Faith_Group_Impersonation",
                "Parent_Association_Impersonation",
                "Local_Issue_Committee_Impersonation",
                "Sports_Club_Impersonation",
            ),
            "Movement_Network_Impersonation": L(
                "Protest_Group_Impersonation",
                "Watchdog_Group_Impersonation",
                "Student_Collective_Impersonation",
                "Mutual_Aid_Network_Impersonation",
                "Veterans_Group_Impersonation",
            ),
        },
        "Peer_Or_Acquaintance_Impersonation": {
            "Relational_Proximity_Impersonation": L(
                "Friend_Impersonation",
                "Colleague_Impersonation",
                "Classmate_Impersonation",
                "Neighbour_Impersonation",
                "Sibling_Impersonation",
            ),
            "Authority_Adjacent_Peer_Impersonation": L(
                "Campaign_Peer_Impersonation",
                "Community_Admin_Impersonation",
                "Local_Reporter_Contact_Impersonation",
                "Organiser_Peer_Impersonation",
                "Coach_Or_Mentor_Impersonation",
            ),
        },
    },
    "Proxy_Front_And_Cutout_Construction": {
        "Front_Media_Properties": {
            "Pseudo_Local_Media_Properties": L(
                "Hyperlocal_News_Fronts",
                "Community_Bulletin_Fronts",
                "Regional_Watchdog_Fronts",
                "Citizen_Journalism_Fronts",
            ),
            "Pseudo_Specialist_Media_Properties": L(
                "Data_And_Polling_Fronts",
                "Security_Briefing_Fronts",
                "Policy_Analysis_Fronts",
                "Fact_Check_Style_Fronts",
                "Independent_Research_Fronts",
                "Open_Source_Intelligence_Fronts",
            ),
        },
        "Cutout_Ngos_And_Issue_Groups": {
            "Advocacy_Front_Groups": L(
                "Anti_Corruption_Fronts",
                "Family_Values_Fronts",
                "Public_Safety_Fronts",
                "Electoral_Integrity_Fronts",
                "Environmental_Fronts",
                "Veterans_Welfare_Fronts",
            ),
            "Civic_Participation_Front_Groups": L(
                "Grassroots_Petition_Fronts",
                "Citizen_Observer_Fronts",
                "Reform_Coalition_Fronts",
                "Community_Rights_Fronts",
            ),
        },
        "Contractor_Or_Pr_Firm_Intermediation": {
            "Commercial_Influence_Shells": L(
                "Digital_Marketing_Cutouts",
                "Reputation_Management_Cutouts",
                "Creator_Management_Cutouts",
                "Survey_And_Data_Cutouts",
                "Crisis_Communication_Cutouts",
            ),
            "Freelancer_And_Microcontractor_Shells": L(
                "Content_Production_Shells",
                "Moderation_And_Seeding_Shells",
                "Engagement_Farm_Shells",
                "Translation_Shells",
                "Voice_Acting_Shells",
            ),
        },
        "Diaspora_Proxy_Voices": L(
            "Diaspora_Media_Voices",
            "Diaspora_Religious_Voices",
            "Diaspora_Business_Voices",
            "Diaspora_Student_Voices",
            "Transnational_Family_Network_Voices",
            "Expatriate_Testimonial_Voices",
            "Return_Migrant_Voices",
            "Identity_Guardian_Voices",
        ),
        "Fellow_Traveler_Ecosystems": L(
            "Ideological_Media_Relays",
            "Subculture_Relay_Accounts",
            "Partisan_Meme_Pages",
            "Cause_Committed_Forum_Relays",
            "Engagement_Hungry_Creators",
            "Contrarian_Commentators",
            "Status_Resenting_Former_Insiders",
            "Scandal_Monetising_Pages",
        ),
    },
    "Legitimacy_And_Status_Signalling": {
        "Stolen_Credibility_Signals": L(
            "Logo_And_Brand_Borrowing",
            "Bureaucratic_Template_Borrowing",
            "Official_Tone_Borrowing",
            "Procedural_Language_Borrowing",
            "Verification_Symbol_Imitation",
            "Letterhead_Imitation",
        ),
        "Credential_Display_Manipulation": {
            "Formal_Credential_Claims": L(
                "Degree_Claims",
                "Institutional_Affiliation_Claims",
                "Service_Record_Claims",
                "Certification_Claims",
                "Award_Claims",
            ),
            "Informal_Status_Claims": L(
                "Insider_Access_Claims",
                "Field_Experience_Claims",
                "Whistleblower_Access_Claims",
                "Community_Leader_Claims",
                "Eyewitness_Claims",
            ),
        },
        "Cross_Platform_Identity_Layering": L(
            "Multi_Platform_Name_Consistency",
            "Multi_Platform_Visual_Consistency",
            "Cross_Linked_Bio_Consistency",
            "Timeline_Depth_Consistency",
            "Interaction_History_Consistency",
        ),
        "Engagement_Seeded_Legitimacy": {
            "Visible_Popularity_Cues": L(
                "Bought_Followers",
                "Bought_Replies",
                "Bought_Reposts",
                "Bought_Reaction_Metrics",
                "Bought_View_Counts",
            ),
            "Visible_Validation_Cues": L(
                "Coordinated_Endorsements",
                "Ringed_Expert_Replies",
                "Badge_And_List_Inclusion_Simulation",
                "Quote_Chain_Validation",
                "Verification_Adjacent_Cues",
            ),
        },
        "Elite_Association_Signalling": L(
            "Photo_Op_Implication",
            "Private_Briefing_Implication",
            "Backchannel_Contact_Implication",
            "Event_Backstage_Access_Implication",
            "Partial_Quote_Endorsement_Implication",
            "Ambiguous_Mention_Endorsement_Implication",
            "Symbolic_Association_Implication",
            "Staff_Or_Adviser_Proximity_Implication",
        ),
    },
    "Authenticity_Obfuscation": {
        "Geolocation_Masking": L(
            "Domestic_Ip_Masking",
            "Timezone_Behaviour_Masking",
            "Regional_Reference_Masking",
            "Local_Event_Awareness_Masking",
            "Mobile_Network_Masking",
        ),
        "Language_Register_Localisation": {
            "Dialect_And_Slang_Localisation": L(
                "Regional_Dialect_Mimicry",
                "Age_Cohort_Slang_Mimicry",
                "Subculture_Jargon_Mimicry",
                "Class_Register_Mimicry",
            ),
            "Issue_Native_Discourse_Localisation": L(
                "Movement_Phraseology_Mimicry",
                "Policy_Domain_Term_Mimicry",
                "Community_In_Joke_Mimicry",
                "Local_Media_Reference_Mimicry",
            ),
        },
        "Mixed_Authentic_Inauthentic_Blending": L(
            "Real_Supporter_Signal_Blending",
            "Compromised_Account_Blending",
            "Creator_Relay_Blending",
            "Community_Admin_Blending",
        ),
        "Network_Compartmentalisation": L(
            "Persona_Cluster_Compartmentalisation",
            "Platform_Role_Compartmentalisation",
            "Linguistic_Audience_Compartmentalisation",
            "Country_Or_Region_Compartmentalisation",
        ),
        "Legacy_Account_Reuse": L(
            "Dormant_Account_Repurposing",
            "Aged_Page_Repurposing",
            "Legacy_Forum_Identity_Repurposing",
            "Historic_Group_Repurposing",
            "Compromised_Creator_Account_Reuse",
            "Compromised_Local_Figure_Account_Reuse",
            "Compromised_Admin_Account_Reuse",
            "Compromised_Institution_Adjacent_Account_Reuse",
        ),
    },
    "Official_And_Diplomatic_Channel_Abuse": {
        "Diplomatic_Amplification": L(
            "Embassy_Account_Amplification",
            "Consular_Network_Amplification",
            "Multilateral_Delegate_Amplification",
            "Official_Spokesperson_Amplification",
        ),
        "State_Media_Legitimisation": L(
            "State_Media_Breaking_Wrap",
            "State_Media_Investigative_Wrap",
            "State_Media_Documentary_Wrap",
            "State_Media_Talk_Show_Wrap",
            "Cultural_Outreach_Wrap",
            "Diaspora_Engagement_Wrap",
            "Expert_Roundtable_Wrap",
            "Official_Rebuttal_Wrap",
        ),
        "Fact_Checking_Style_Deception": L(
            "Official_Clarification_Style_Deception",
            "Forensic_Rebuttal_Style_Deception",
            "Myth_Vs_Fact_Style_Deception",
            "Timeline_Reconstruction_Style_Deception",
        ),
        "Forum_Presence_Exploitation": L(
            "Conference_Panel_Exploitation",
            "Observer_Mission_Exploitation",
            "Public_Consultation_Exploitation",
            "Hearing_And_Roundtable_Exploitation",
        ),
        "Official_Cross_Signal_Chaining": L(
            "Ministry_To_Media_Signal_Chaining",
            "Diplomat_To_Creator_Signal_Chaining",
            "Official_Statement_To_Proxy_Site_Chaining",
            "Briefing_To_Leak_Style_Chaining",
            "Domestic_Language_Chaining",
            "Diaspora_Language_Chaining",
            "International_Language_Chaining",
            "Expert_Vernacular_Chaining",
        ),
    },
}


# ---------------------------------------------------------------------------
# 3. CLAIM, FRAME, AND NARRATIVE MANIPULATION
# ---------------------------------------------------------------------------
CLAIM_FRAME_NARRATIVE = {
    "Claim_Level_Deception": {
        "Fabricated_Claim_Injection": {
            "Event_Fabrication": L(
                "Fabricated_Incidents",
                "Fabricated_Meetings",
                "Fabricated_Votes_Or_Decisions",
                "Fabricated_Policy_Plans",
                "Fabricated_Speeches",
            ),
            "Actor_And_Motive_Fabrication": L(
                "Fabricated_Conspirators",
                "Fabricated_Whistleblowers",
                "Fabricated_Foreign_Links",
                "Fabricated_Hidden_Sponsors",
                "Fabricated_Backroom_Deals",
            ),
            "Outcome_Fabrication": L(
                "Fabricated_Harm_Outcomes",
                "Fabricated_Benefit_Outcomes",
                "Fabricated_Participation_Numbers",
                "Fabricated_Security_Failures",
                "Fabricated_Casualty_Counts",
            ),
        },
        "Misleading_Framing": {
            "Headline_And_Lede_Misframing": L(
                "Alarmist_Headline_Misframing",
                "Certainty_Overstatement_Misframing",
                "Motive_Imputation_Misframing",
                "Procedural_Implication_Misframing",
                "Causal_Overreach_Misframing",
            ),
            "Visual_And_Caption_Misframing": L(
                "Decontextualised_Image_Captioning",
                "Suggestive_Thumbnail_Misframing",
                "Outcome_Implying_Graphic_Misframing",
                "Timeline_Misordering_Misframing",
            ),
        },
        "Context_Stripping": {
            "Quote_Context_Stripping": L(
                "Partial_Quote_Context_Stripping",
                "Sarcasm_Removal_Context_Stripping",
                "Conditionality_Removal_Context_Stripping",
                "Audience_Context_Removal",
                "Irony_Marker_Removal",
            ),
            "Event_Context_Stripping": L(
                "Time_Window_Context_Stripping",
                "Preceding_Event_Context_Stripping",
                "Response_Sequence_Context_Stripping",
                "Policy_Process_Context_Stripping",
            ),
        },
        "Data_And_Quote_Cherry_Picking": {
            "Quantitative_Cherry_Picking": L(
                "Time_Window_Cherry_Picking",
                "Baseline_Cherry_Picking",
                "Subgroup_Cherry_Picking",
                "Outlier_Cherry_Picking",
                "Denominator_Cherry_Picking",
            ),
            "Testimonial_And_Quote_Cherry_Picking": L(
                "Representative_Sounding_Quote_Cherry_Picking",
                "Elite_Quote_Cherry_Picking",
                "Angriest_Comment_Cherry_Picking",
                "Selective_Whistleblower_Quote_Cherry_Picking",
            ),
        },
        "Causal_Misattribution": L(
            "Wrong_Actor_Blaming",
            "Proxy_As_Principal_Blaming",
            "Foreign_Actor_False_Blaming",
            "Bureaucratic_Level_False_Blaming",
            "Policy_Mechanism_Misattribution",
            "Economic_Effect_Misattribution",
            "Security_Effect_Misattribution",
            "Procedural_Outcome_Misattribution",
        ),
        "Statistical_Manipulation": L(
            "Base_Rate_Suppression",
            "Relative_Vs_Absolute_Risk_Manipulation",
            "Survivorship_Bias_Exploitation",
            "Selection_Bias_Exploitation",
            "Mean_Vs_Median_Manipulation",
            "Truncated_Time_Series_Use",
        ),
    },
    "Frame_And_Reframing_Operations": {
        "Dismiss": {
            "Denial_Frames": L(
                "Flat_Denial",
                "Evidence_Insufficiency_Denial",
                "Procedural_Denial",
                "Semantic_Redefinition_Denial",
            ),
            "Critic_Devaluation_Frames": L(
                "Partisan_Motive_Attack",
                "Competence_Attack",
                "Elitism_Attack",
                "Foreign_Influence_Attack",
                "Mental_Stability_Attack",
            ),
        },
        "Distort": {
            "Sequence_Distortion": L(
                "Cause_Before_Effect_Distortion",
                "Edited_Timeline_Distortion",
                "Simultaneity_Distortion",
                "Threshold_Crossing_Distortion",
            ),
            "Proportion_Distortion": L(
                "Isolated_Case_As_Pattern",
                "Rare_Event_As_Trend",
                "Small_Group_As_Majority",
                "Uncertainty_As_Confirmation",
            ),
        },
        "Distract": {
            "Topic_Displacement": L(
                "Adjacent_Scandal_Substitution",
                "Personality_Conflict_Substitution",
                "Symbolic_Culture_War_Substitution",
                "Foreign_Whatabout_Substitution",
            ),
            "Blame_Redirection": L(
                "Middleman_Blaming",
                "Opposition_Blaming",
                "Historical_Actor_Blaming",
                "Bureaucratic_Maze_Blaming",
            ),
        },
        "Dismay": {
            "Threat_Intensification": L(
                "Imminence_Intensification",
                "Scale_Intensification",
                "Vulnerability_Intensification",
                "Helplessness_Intensification",
            ),
            "Deterrent_Signalling": L(
                "Participation_Cost_Signalling",
                "Retaliation_Risk_Signalling",
                "Surveillance_Risk_Signalling",
                "Social_Exclusion_Signalling",
            ),
        },
        "Divide": {
            "Horizontal_Division_Frames": L(
                "Class_Against_Class_Frames",
                "Region_Against_Region_Frames",
                "Generation_Against_Generation_Frames",
                "Majority_Against_Minority_Frames",
                "Urban_Against_Rural_Frames",
            ),
            "Vertical_Division_Frames": L(
                "People_Against_Elites_Frames",
                "Grassroots_Against_Brokers_Frames",
                "Local_Against_Capital_Frames",
                "Workers_Against_Experts_Frames",
            ),
        },
    },
    "Master_Narrative_Engineering": {
        "Betrayal_Narratives": L(
            "Elite_Sellout_Narratives",
            "Party_Defection_Narratives",
            "Institutional_Double_Standard_Narratives",
            "Expert_Deception_Narratives",
            "Ally_Abandonment_Narratives",
        ),
        "Threat_Narratives": L(
            "Existential_Security_Threat_Narratives",
            "Cultural_Erasure_Threat_Narratives",
            "Economic_Dispossession_Threat_Narratives",
            "Children_And_Family_Threat_Narratives",
            "Sovereignty_Threat_Narratives",
            "Civilisation_Collapse_Narratives",
        ),
        "Decline_Narratives": L(
            "National_Decline_Narratives",
            "Moral_Decay_Narratives",
            "Institutional_Collapse_Narratives",
            "Community_Disintegration_Narratives",
            "Irreversible_Trajectory_Narratives",
        ),
        "Corruption_Narratives": L(
            "Captured_State_Narratives",
            "Rigged_System_Narratives",
            "Pay_To_Play_Narratives",
            "Cartelised_Elite_Narratives",
            "Procedural_Cover_Up_Narratives",
        ),
        "Replacement_And_Displacement_Narratives": L(
            "Demographic_Replacement_Narratives",
            "Cultural_Displacement_Narratives",
            "Status_Displacement_Narratives",
            "Citizen_Dispossession_Narratives",
        ),
        "Inevitability_Narratives": L(
            "Already_Decided_Outcome_Narratives",
            "Silent_Majority_Awakening_Narratives",
            "Historic_Turning_Point_Narratives",
            "Final_Countdown_Narratives",
        ),
        "Purification_And_Restoration_Narratives": L(
            "National_Rebirth_Narratives",
            "Moral_Restoration_Narratives",
            "Institutional_Cleanse_Narratives",
            "Community_Take_Back_Control_Narratives",
        ),
        "Conspiracy_Master_Narratives": L(
            "Hidden_Cabal_Master_Narrative",
            "Globalist_Plot_Master_Narrative",
            "Deep_State_Master_Narrative",
            "Engineered_Crisis_Master_Narrative",
            "Suppressed_Cure_Or_Truth_Master_Narrative",
        ),
    },
    "Emotional_And_Moral_Engineering": {
        "Fear_Appeals": {
            "Personal_Safety_Fear": L(
                "Crime_Exposure_Fear",
                "Family_Harm_Fear",
                "Social_Breakdown_Fear",
                "Retaliation_Fear",
                "Health_Threat_Fear",
            ),
            "Collective_Survival_Fear": L(
                "National_Extinction_Fear",
                "Cultural_Annihilation_Fear",
                "Economic_Ruin_Fear",
                "Democratic_Collapse_Fear",
                "Ecological_Catastrophe_Fear",
            ),
        },
        "Anger_Induction": {
            "Injustice_Anger": L(
                "Double_Standard_Anger",
                "Corruption_Anger",
                "Humiliation_Anger",
                "Resource_Unfairness_Anger",
                "Procedural_Unfairness_Anger",
            ),
            "Betrayal_Anger": L(
                "Elite_Betrayal_Anger",
                "Party_Betrayal_Anger",
                "Media_Betrayal_Anger",
                "Institutional_Betrayal_Anger",
                "Ally_Betrayal_Anger",
            ),
        },
        "Disgust_Induction": L(
            "Sanitary_Disgust_Induction",
            "Moral_Disgust_Induction",
            "Bodily_Disgust_Induction",
            "Outgroup_Disgust_Induction",
        ),
        "Humiliation_And_Contempt": {
            "Target_Status_Degradation": L(
                "Ridicule_Of_Competence",
                "Ridicule_Of_Morality",
                "Ridicule_Of_Authenticity",
                "Ridicule_Of_Strength",
            ),
            "Outgroup_Dehumanising_Contempt": L(
                "Parasite_Framing",
                "Vermin_Framing",
                "Degeneracy_Framing",
                "Cowardice_Framing",
                "Subhuman_Framing",
            ),
        },
        "Moral_Shock": {
            "Shock_Image_Deployment": L(
                "Injury_Or_Victim_Imagery",
                "Symbol_Desecration_Imagery",
                "Children_At_Risk_Imagery",
                "Sacred_Site_Violation_Imagery",
                "Animal_Cruelty_Imagery",
            ),
            "Shock_Story_Deployment": L(
                "Single_Case_Horror_Story",
                "Betrayal_Confession_Story",
                "Hidden_Abuse_Story",
                "Procedural_Injustice_Story",
            ),
        },
        "Nostalgia_And_Restoration": {
            "Golden_Age_Activation": L(
                "Safer_Past_Activation",
                "More_Cohesive_Past_Activation",
                "Fairer_Economy_Past_Activation",
                "Stronger_State_Past_Activation",
            ),
            "Stolen_Future_Reversal": L(
                "Restore_Control_Frames",
                "Restore_Dignity_Frames",
                "Restore_Order_Frames",
                "Restore_Moral_Boundaries_Frames",
            ),
        },
        "Hope_And_Salvation_Scripts": {
            "Rescuer_Personalisation": L(
                "Hero_Leader_Personalisation",
                "Citizen_Uprising_Personalisation",
                "Whistleblower_Saviour_Personalisation",
                "Technocratic_Fix_Personalisation",
            ),
            "Collective_Redemption": L(
                "Community_Repair_Scripts",
                "National_Rebirth_Scripts",
                "Moral_Cleanse_Scripts",
                "Institutional_Reset_Scripts",
            ),
        },
        "Reciprocity_And_Sympathy_Hooks": L(
            "Beneficiary_Story_Hooks",
            "Mutual_Aid_Hooks",
            "Indebtedness_Hooks",
            "Group_Sacrifice_Hooks",
        ),
    },
    "Identity_And_Group_Coding": {
        "Ingroup_Fusion": L(
            "Issue_As_Loyalty_Test",
            "Issue_As_Membership_Badge",
            "Issue_As_Shared_Sacrifice",
            "Issue_As_Identity_Boundary",
        ),
        "Outgroup_Essentialisation": L(
            "Inherent_Corruption_Coding",
            "Inherent_Danger_Coding",
            "Inherent_Foreignness_Coding",
            "Inherent_Untrustworthiness_Coding",
        ),
        "Status_Threat_Coding": L(
            "Masculinity_Status_Threat",
            "Class_Status_Threat",
            "Majority_Status_Threat",
            "Regional_Status_Threat",
            "Generational_Status_Threat",
        ),
        "Boundary_Hardening": L(
            "Compromise_As_Betrayal",
            "Nuance_As_Weakness",
            "Dialogue_As_Collaboration",
            "Defection_As_Pollution",
        ),
        "Moral_Purity_Coding": L(
            "Children_And_Innocence_Purity_Coding",
            "Faith_And_Sacred_Duty_Coding",
            "Civilisational_Purity_Coding",
            "Anti_Corruption_Purity_Coding",
        ),
        "Victimhood_Identity_Coding": L(
            "Persecuted_Majority_Coding",
            "Silenced_Truth_Teller_Coding",
            "Forgotten_Region_Coding",
            "Humiliated_Citizen_Coding",
        ),
    },
    "Epistemic_Destabilisation": {
        "Certainty_Erosion": L(
            "Nothing_Can_Be_Known_Frames",
            "All_Sides_Lie_Frames",
            "Evidence_Is_Always_Manipulated_Frames",
            "Truth_Is_Only_Group_Loyalty_Frames",
        ),
        "Contradiction_Stacking": L(
            "Mutually_Incompatible_Accusation_Stacking",
            "Multiple_Motive_Stacking",
            "Parallel_Scapegoat_Stacking",
            "Serial_Plotline_Stacking",
        ),
        "Evidence_Reweighting": L(
            "Fringe_Source_Promotion",
            "Anecdote_Over_Dataset_Promotion",
            "Viral_Clip_Over_Context_Promotion",
            "Pseudo_Forensic_Over_Expert_Promotion",
        ),
        "Source_Relativisation": L(
            "Journalism_As_Rumor_Relativisation",
            "Expertise_As_Opinion_Relativisation",
            "Official_Records_As_Spin_Relativisation",
            "Verified_And_Unverified_Flattening",
        ),
        "Whataboutist_Deflection": L(
            "Historical_Whataboutism",
            "Foreign_Whataboutism",
            "Opposition_Whataboutism",
            "Procedural_Whataboutism",
        ),
        "Verification_Fatigue_Induction": L(
            "Constant_Fact_Dispute_Induction",
            "Endless_Document_Demand_Induction",
            "Rebuttal_Whack_A_Mole_Induction",
            "Speed_Over_Accuracy_Induction",
        ),
        "Overton_Window_Engineering": L(
            "Extremist_Anchor_Seeding",
            "Compromise_Position_Reframing",
            "Mainstream_Position_Marginalisation",
            "Taboo_Topic_Mainstreaming",
        ),
    },
    "Narrative_Laundering_And_Translation": {
        "Fringe_To_Mainstream_Translation": L(
            "Meme_To_Commentary_Translation",
            "Forum_Rumor_To_Article_Translation",
            "Stream_Clip_To_Newsworthy_Translation",
            "Anonymous_Post_To_Witness_Translation",
        ),
        "Pseudo_Journalistic_Repackaging": L(
            "Headline_Neutralisation",
            "Quote_Balance_Repackaging",
            "Evidence_Style_Layout_Repackaging",
            "Investigation_Style_Repackaging",
        ),
        "Cross_Lingual_Reframing": L(
            "Domestic_Language_Reframing",
            "Diaspora_Language_Reframing",
            "Elite_International_Language_Reframing",
            "Subculture_Vernacular_Reframing",
        ),
        "Citation_Chain_Construction": L(
            "Circular_Quote_Chains",
            "Mirror_Site_Quote_Chains",
            "Podcast_To_Article_To_Post_Quote_Chains",
            "Pseudo_Archive_Quote_Chains",
        ),
        "Influencer_Style_Translation": L(
            "Humor_Translation",
            "Rage_Clip_Translation",
            "Lifestyle_Translation",
            "Storytime_Translation",
        ),
    },
}


# ---------------------------------------------------------------------------
# 4. AI-GENERATED SYNTHETIC MEDIA AND CONTENT (very deep)
# ---------------------------------------------------------------------------
AI_GENERATED_SYNTHETIC_MEDIA = {
    "Foundation_Model_Text_Generation": {
        "Llm_Long_Form_Article_Generation": L(
            "Synthetic_Investigative_Report_Production",
            "Synthetic_White_Paper_Production",
            "Synthetic_Court_Filing_Production",
            "Synthetic_Internal_Memo_Production",
            "Synthetic_Op_Ed_Production",
            "Synthetic_Academic_Paper_Production",
            "Synthetic_Policy_Brief_Production",
            "Synthetic_Regulatory_Comment_Production",
        ),
        "Llm_Short_Form_Post_Generation": L(
            "Sockpuppet_Profile_Post_Generation",
            "Influencer_Style_Post_Generation",
            "Insider_Whistleblower_Post_Generation",
            "Ordinary_Citizen_Post_Generation",
            "Activist_Style_Post_Generation",
            "Quote_Card_Caption_Generation",
            "Thread_Style_Post_Generation",
        ),
        "Llm_Conversational_Generation": L(
            "Direct_Message_Persona_Generation",
            "Comment_Reply_Generation",
            "Forum_Thread_Continuation_Generation",
            "Live_Stream_Chat_Generation",
            "Customer_Service_Style_Persona_Generation",
        ),
        "Llm_Document_Forgery_Generation": L(
            "Memo_Header_Plus_Body_Forgery",
            "Email_Thread_Forgery",
            "Chat_Log_Forgery",
            "Press_Release_Forgery",
            "Legal_Notice_Forgery",
        ),
        "Llm_Translation_And_Cross_Lingual_Generation": L(
            "Source_To_Target_Language_Generation",
            "Style_Transfer_Translation_Generation",
            "Diaspora_Language_Generation",
            "Code_Switched_Generation",
        ),
        "Llm_Style_Transfer_Generation": L(
            "Author_Voice_Mimicry_Generation",
            "Outlet_Style_Mimicry_Generation",
            "Subculture_Style_Mimicry_Generation",
            "Generation_Specific_Style_Mimicry",
        ),
        "Llm_Adversarial_Argumentation_Generation": L(
            "Counter_Argument_Generation",
            "Steelmanning_To_Strawman_Generation",
            "Socratic_Sceptic_Generation",
            "Apparent_Concession_Generation",
        ),
    },
    "Foundation_Model_Image_Generation": {
        "Photorealistic_Diffusion_Generation": L(
            "Person_Headshot_Generation",
            "Crowd_Scene_Generation",
            "Event_Scene_Generation",
            "Street_Or_Border_Scene_Generation",
            "Document_Photograph_Generation",
            "Object_Photograph_Generation",
            "Receipt_Or_Form_Photograph_Generation",
        ),
        "Stylised_Image_Generation": L(
            "Meme_Image_Generation",
            "Infographic_Generation",
            "Propaganda_Poster_Generation",
            "Comic_Strip_Generation",
            "Caricature_Generation",
        ),
        "Diagrammatic_And_Map_Generation": L(
            "Fake_Map_Generation",
            "Fake_Network_Diagram_Generation",
            "Fake_Org_Chart_Generation",
            "Fake_Timeline_Diagram_Generation",
        ),
        "Image_Editing_Manipulation": L(
            "Inpainting_Editing",
            "Outpainting_Editing",
            "Object_Removal_Editing",
            "Object_Insertion_Editing",
            "Face_Swap_Editing",
            "Background_Replacement_Editing",
            "Crowd_Size_Editing",
        ),
        "Provenance_Plausibility_Engineering": L(
            "Camera_Sensor_Noise_Simulation",
            "Compression_Artifact_Simulation",
            "Exif_Metadata_Spoofing_For_Image",
            "Watermark_Removal",
            "Aging_Filter_Application",
        ),
    },
    "Foundation_Model_Audio_Generation": {
        "Voice_Cloning_Generation": {
            "Single_Speaker_Cloning": L(
                "Politician_Voice_Cloning",
                "Journalist_Voice_Cloning",
                "Local_Authority_Voice_Cloning",
                "Family_Member_Voice_Cloning",
                "Celebrity_Voice_Cloning",
            ),
            "Multi_Speaker_Conversation_Cloning": L(
                "Two_Party_Phone_Call_Cloning",
                "Panel_Discussion_Cloning",
                "Newsroom_Conversation_Cloning",
            ),
            "Emotion_Transfer_Cloning": L(
                "Anger_Transfer_Cloning",
                "Fear_Transfer_Cloning",
                "Sadness_Transfer_Cloning",
                "Confusion_Transfer_Cloning",
            ),
        },
        "Synthetic_Speech_Generation": L(
            "Robocall_Synthesis",
            "Voicemail_Drop_Synthesis",
            "Podcast_Style_Synthesis",
            "Voice_Note_Synthesis",
            "Audiobook_Style_Synthesis",
            "Phone_Banking_Synthesis",
        ),
        "Audio_Editing_Manipulation": L(
            "Splice_And_Stitch_Editing",
            "Tempo_And_Pitch_Manipulation",
            "Reverb_Insertion_For_Authenticity",
            "Noise_Floor_Engineering",
            "Silence_Insertion_Or_Removal",
        ),
        "Foreign_Language_Audio_Generation": L(
            "Diaspora_Language_Speech_Synthesis",
            "Multilingual_Press_Statement_Synthesis",
            "Translation_With_Voice_Match_Synthesis",
        ),
    },
    "Foundation_Model_Video_Generation": {
        "Lip_Sync_Deepfake_Generation": L(
            "Politician_Speech_Lip_Sync",
            "Anchor_Lip_Sync",
            "Local_Official_Lip_Sync",
            "Religious_Leader_Lip_Sync",
        ),
        "Full_Body_Deepfake_Generation": L(
            "Pose_Transfer_Deepfake",
            "Action_Synthesis_Deepfake",
            "Walking_Or_Gesture_Deepfake",
            "Dance_Or_Performance_Deepfake",
        ),
        "Head_Swap_Deepfake_Generation": L(
            "Single_Subject_Head_Swap",
            "Multi_Subject_Group_Head_Swap",
            "Crowd_Head_Swap",
        ),
        "Text_To_Video_Generation": L(
            "Scene_From_Prompt_Generation",
            "News_Style_Footage_Generation",
            "Surveillance_Style_Footage_Generation",
            "Bodycam_Style_Footage_Generation",
        ),
        "Image_To_Video_Generation": L(
            "Photo_Animation",
            "Speech_Animation_From_Photo",
            "Action_Animation_From_Photo",
        ),
        "Cheapfake_Video_Editing": L(
            "Speed_Manipulation_Editing",
            "Clip_Splicing_Editing",
            "Silence_Insertion_Editing",
            "Crop_For_False_Implication_Editing",
            "Subtitle_Misrepresentation_Editing",
        ),
        "Live_Stream_Avatar_Generation": L(
            "Real_Time_Lip_Sync_Avatar",
            "Real_Time_Body_Avatar",
            "Multi_Camera_Synthetic_Show",
        ),
    },
    "Multimodal_Bundle_Generation": L(
        "Document_Plus_Audio_Bundle_Generation",
        "Image_Plus_Quote_Bundle_Generation",
        "Video_Plus_Transcript_Bundle_Generation",
        "Article_Plus_Infographic_Bundle_Generation",
        "Voice_Note_Plus_Screenshot_Bundle_Generation",
    ),
    "Generative_Provenance_Manipulation": {
        "Metadata_Engineering": L(
            "Exif_Spoofing",
            "Geotag_Spoofing",
            "Author_Field_Spoofing",
            "Software_Signature_Spoofing",
            "Camera_Make_Model_Spoofing",
        ),
        "Authentication_Defeat": L(
            "C2pa_Manifest_Stripping",
            "Watermark_Removal_For_Generative_Content",
            "Digital_Signature_Stripping",
            "Re_Capture_Laundering",
        ),
        "Synthetic_Source_History_Construction": L(
            "Discovery_Story_Fabrication",
            "Chain_Of_Forwarding_Fabrication",
            "Insider_Hand_Off_Fabrication",
            "Journalist_Tipline_Fabrication",
        ),
    },
    "Generative_Persona_Backstory_Production": L(
        "Coherent_Bio_Plus_History_Production",
        "Localised_Backstory_Production",
        "Time_Consistent_Posting_History_Production",
        "Plausible_Family_Network_Production",
        "Synthetic_Education_And_Career_History_Production",
    ),
}


# ---------------------------------------------------------------------------
# 5. MULTI-AGENT ADVERSARIAL ARCHITECTURE
# ---------------------------------------------------------------------------
MULTI_AGENT_ADVERSARIAL = {
    "Persona_Cluster_Orchestration": {
        "Concerned_Citizen_Cluster": L(
            "Coordinated_Local_Parent_Cluster",
            "Coordinated_Concerned_Taxpayer_Cluster",
            "Coordinated_Disillusioned_Voter_Cluster",
            "Coordinated_Apolitical_Neighbour_Cluster",
        ),
        "Insider_Whistleblower_Cluster": L(
            "Coordinated_Anonymous_Insider_Cluster",
            "Coordinated_Former_Staffer_Cluster",
            "Coordinated_Defector_Cluster",
        ),
        "Domain_Expert_Cluster": L(
            "Coordinated_Pseudo_Academic_Cluster",
            "Coordinated_Pseudo_Security_Analyst_Cluster",
            "Coordinated_Pseudo_Public_Health_Expert_Cluster",
            "Coordinated_Pseudo_Pollster_Cluster",
        ),
        "Sceptical_Bystander_Cluster": L(
            "Coordinated_Devil_Advocate_Cluster",
            "Coordinated_Just_Asking_Questions_Cluster",
            "Coordinated_Centrist_Concern_Trolling_Cluster",
        ),
        "Aggrieved_Identity_Cluster": L(
            "Coordinated_Persecuted_Majority_Cluster",
            "Coordinated_Forgotten_Region_Cluster",
            "Coordinated_Veteran_Cluster",
            "Coordinated_Religious_Cluster",
        ),
    },
    "Adversarial_Conversation_Loops": {
        "Counter_Argument_Adaptation": L(
            "Single_Turn_Counter_Argument_Adaptation",
            "Multi_Turn_Counter_Argument_Adaptation",
            "Cross_Conversation_Counter_Argument_Adaptation",
        ),
        "Long_Term_Relationship_Building": L(
            "Helpful_Information_Seeding",
            "Shared_Identity_Performance",
            "Sympathy_And_Validation_Loop",
            "Reciprocity_Building_Loop",
        ),
        "Conversion_Funnel_Operations": L(
            "Awareness_To_Curiosity_Funnel",
            "Curiosity_To_Conviction_Funnel",
            "Conviction_To_Action_Funnel",
            "Action_To_Recruitment_Funnel",
        ),
        "Reflexive_Argumentation_Refinement": L(
            "Real_Time_Objection_Probing",
            "Profile_Adaptive_Argument_Refinement",
            "Cross_Target_Argument_Generalisation",
            "Defender_Reaction_Adaptation",
        ),
    },
    "Cross_Channel_Synchronization": {
        "Same_Persona_Across_Platforms_Synchronization": L(
            "X_Plus_Telegram_Persona_Synchronization",
            "Tiktok_Plus_Instagram_Persona_Synchronization",
            "Discord_Plus_Forum_Persona_Synchronization",
            "Whatsapp_Plus_Sms_Persona_Synchronization",
        ),
        "Coordinated_Persona_Network_Synchronization": L(
            "Hub_And_Spoke_Persona_Network",
            "Mesh_Persona_Network",
            "Cell_Style_Persona_Network",
        ),
        "Multi_Language_Persona_Synchronization": L(
            "Domestic_Plus_Diaspora_Language_Synchronization",
            "Domestic_Plus_International_Language_Synchronization",
            "Multi_Language_Cell_Synchronization",
        ),
    },
    "Auto_Targeting_Pipelines": {
        "Profile_Inference_Targeting": L(
            "Demographic_Inference_Pipeline",
            "Psychographic_Inference_Pipeline",
            "Issue_Affinity_Inference_Pipeline",
            "Mobilisation_Propensity_Inference_Pipeline",
        ),
        "Cohort_Adapted_Messaging": L(
            "Real_Time_Cohort_Detection_Adaptation",
            "Region_Adaptive_Messaging",
            "Generation_Adaptive_Messaging",
            "Identity_Adaptive_Messaging",
        ),
        "Real_Time_Ab_Optimization": L(
            "Headline_Variant_Optimization",
            "Frame_Variant_Optimization",
            "Messenger_Variant_Optimization",
            "Format_Variant_Optimization",
        ),
        "Reinforcement_Learning_Targeting": L(
            "Engagement_Reward_Optimisation",
            "Conversion_Reward_Optimisation",
            "Click_Through_Reward_Optimisation",
            "Reply_Velocity_Reward_Optimisation",
        ),
    },
    "Agentic_Operation_Roles": {
        "Seeder_Agents": L(
            "Originator_Seeder_Agent",
            "Translator_Seeder_Agent",
            "Localiser_Seeder_Agent",
        ),
        "Validator_Agents": L(
            "Pseudo_Expert_Validator_Agent",
            "Pseudo_Witness_Validator_Agent",
            "Pseudo_Whistleblower_Validator_Agent",
        ),
        "Booster_Agents": L(
            "Repost_Booster_Agent",
            "Quote_Boost_Agent",
            "Cross_Platform_Booster_Agent",
        ),
        "Bridge_Agents": L(
            "Subculture_To_Mainstream_Bridge_Agent",
            "Diaspora_To_Domestic_Bridge_Agent",
            "Cross_Issue_Bridge_Agent",
        ),
        "Harasser_Agents": L(
            "Direct_Harasser_Agent",
            "Indirect_Mockery_Agent",
            "Coordinated_Reporter_Agent",
        ),
    },
    "Hybrid_Human_Ai_Operations": L(
        "Human_In_The_Loop_Drafting",
        "Ai_Drafted_Human_Reviewed_Posting",
        "Real_Time_Human_Triggered_Ai_Replies",
        "Ai_Suggestion_For_Human_Operator_Posting",
        "Human_Coordinator_Plus_Ai_Persona_Cluster",
    ),
}


# ---------------------------------------------------------------------------
# 6. TARGETING DELIVERY AND DISCOVERABILITY
# ---------------------------------------------------------------------------
TARGETING_DELIVERY = {
    "Audience_Matching_And_Microtargeting": {
        "Demographic_Targeting": L(
            "Age_Matched_Targeting",
            "Gender_Role_Coded_Targeting",
            "Income_Strata_Targeting",
            "Occupation_Coded_Targeting",
            "Family_Stage_Targeting",
            "Disability_Coded_Targeting",
        ),
        "Psychographic_Targeting": L(
            "Fear_Profile_Targeting",
            "Moral_Value_Targeting",
            "Status_Anxiety_Targeting",
            "Need_For_Closure_Targeting",
            "Identity_Threat_Targeting",
            "Authority_Cue_Targeting",
            "Reactance_Profile_Targeting",
        ),
        "Geotargeting": L(
            "District_Level_Geotargeting",
            "Neighbourhood_Level_Geotargeting",
            "Campus_Level_Geotargeting",
            "Border_Region_Geotargeting",
            "Event_Radius_Geotargeting",
            "Polling_Station_Radius_Geotargeting",
        ),
        "Issue_Affinity_Targeting": L(
            "Single_Issue_Group_Targeting",
            "Protest_Adjacent_Targeting",
            "Petition_Signer_Targeting",
            "Comment_Thread_Interest_Targeting",
            "Legacy_Supporter_Affinity_Targeting",
            "Newsletter_Subscriber_Targeting",
        ),
        "Behavioural_Retargeting": L(
            "Repeat_Exposure_Retargeting",
            "Dropoff_Reactivation_Retargeting",
            "Rage_Engager_Retargeting",
            "Curiosity_Clicker_Retargeting",
            "Conversion_Pathway_Retargeting",
        ),
        "Lookalike_Audience_Construction": L(
            "Donor_Lookalike_Construction",
            "Volunteer_Lookalike_Construction",
            "Voter_Lookalike_Construction",
            "Disengaged_Lookalike_Construction",
        ),
    },
    "Channel_Selection_And_Pathway_Design": {
        "Mainstream_Platform_Delivery": L(
            "Feed_Based_Delivery",
            "Short_Video_Delivery",
            "Public_Comment_Delivery",
            "Trend_Surface_Delivery",
            "Live_Stream_Chat_Delivery",
            "Direct_Quote_Tweet_Delivery",
        ),
        "Niche_Community_Delivery": L(
            "Subforum_Delivery",
            "Identity_Group_Delivery",
            "Issue_Group_Delivery",
            "Gaming_Or_Subculture_Delivery",
            "Campus_Or_Local_Group_Delivery",
            "Religious_Group_Delivery",
        ),
        "Messaging_App_Delivery": L(
            "Broadcast_List_Delivery",
            "Closed_Group_Delivery",
            "Forward_Chain_Delivery",
            "Channel_Plus_Chat_Delivery",
            "Voice_Note_Delivery",
            "Encrypted_Channel_Delivery",
        ),
        "Email_Sms_And_Robocall_Delivery": L(
            "Official_Notice_Style_Email_Delivery",
            "Peer_Style_Sms_Delivery",
            "Deadline_Pressure_Text_Delivery",
            "Voice_Call_Urgency_Delivery",
            "Procedural_Guidance_Delivery",
            "Spoofed_Caller_Id_Delivery",
        ),
        "Offline_Bridge_Delivery": L(
            "Poster_And_Leaflet_Bridge_Delivery",
            "Local_Radio_Bridge_Delivery",
            "Street_Spectacle_Bridge_Delivery",
            "Townhall_Bridge_Delivery",
            "Word_Of_Mouth_Bridge_Delivery",
            "Religious_Service_Mention_Bridge_Delivery",
        ),
        "Voice_Assistant_And_Ai_Channel_Delivery": L(
            "Voice_Assistant_Answer_Delivery",
            "Ai_Chat_Suggested_Source_Delivery",
            "Ai_Summary_Delivery",
            "Smart_Speaker_Briefing_Delivery",
        ),
    },
    "Discoverability_Engineering": {
        "Hashtag_Hijacking": L(
            "Crisis_Hashtag_Hijacking",
            "Campaign_Hashtag_Hijacking",
            "Issue_Movement_Hashtag_Hijacking",
            "Memorial_Or_Symbolic_Hashtag_Hijacking",
        ),
        "Search_Engine_Optimisation": L(
            "Issue_Query_Optimisation",
            "Candidate_Name_Optimisation",
            "Scandal_Query_Optimisation",
            "Procedural_Help_Query_Optimisation",
            "Misspelling_Capture_Optimisation",
        ),
        "Recommendation_System_Gaming": L(
            "Watch_Time_Gaming",
            "Engagement_Velocity_Gaming",
            "Topic_Cluster_Gaming",
            "Cross_Post_Momentum_Gaming",
            "Co_Watch_Cluster_Gaming",
        ),
        "Autocomplete_And_Query_Pollution": L(
            "Suggested_Query_Pollution",
            "Related_Search_Pollution",
            "Misspelling_Capture_Pollution",
            "Compare_Query_Pollution",
        ),
        "Ai_Retrieval_Surface_Poisoning": L(
            "Faq_Style_Source_Poisoning",
            "Pseudo_Reference_Source_Poisoning",
            "Answer_Snippet_Source_Poisoning",
            "Synthetic_Consensus_Source_Poisoning",
            "Knowledge_Panel_Poisoning",
            "Voice_Assistant_Answer_Poisoning",
            "Rag_Document_Insertion",
            "Citation_Chain_Fabrication",
            "Synthetic_Reference_Site_Cloning",
        ),
        "Vector_Embedding_Adversarial_Drift": L(
            "Embedding_Anchor_Drift",
            "Synonym_Substitution_Drift",
            "Semantic_Cluster_Capture_Drift",
            "Adversarial_Tokenisation_Drift",
        ),
    },
    "Timing_And_Sequence_Optimisation": {
        "Crisis_Window_Targeting": L(
            "Breaking_Event_Targeting",
            "Uncertainty_Peak_Targeting",
            "Fear_Peak_Targeting",
            "Verification_Delay_Targeting",
        ),
        "Election_Window_Targeting": L(
            "Registration_Window_Targeting",
            "Debate_Window_Targeting",
            "Early_Vote_Window_Targeting",
            "Polling_Day_Window_Targeting",
            "Counting_And_Certification_Window_Targeting",
        ),
        "Attention_Cycle_Exploitation": L(
            "Late_Night_Release_Timing",
            "Weekend_Release_Timing",
            "Holiday_Release_Timing",
            "Competing_Major_News_Release_Timing",
        ),
        "Prebunk_Preemption_Timing": L(
            "First_Frame_Capture_Timing",
            "Official_Response_Preemption_Timing",
            "Fact_Check_Preemption_Timing",
            "Narrative_Vacuum_Capture_Timing",
        ),
        "Escalation_Sequencing": L(
            "Soft_To_Hard_Claim_Sequencing",
            "Curiosity_To_Outrage_Sequencing",
            "Exposure_To_Action_Prompt_Sequencing",
            "Single_Case_To_Systemic_Claim_Sequencing",
        ),
    },
    "Message_Variant_Control": {
        "Segment_Specific_Wording": L(
            "Youth_Coded_Wording",
            "Faith_Coded_Wording",
            "Working_Class_Coded_Wording",
            "Elite_Professional_Coded_Wording",
            "Diaspora_Coded_Wording",
        ),
        "Issue_Specific_Reframing": L(
            "Economy_Frame_Variant",
            "Security_Frame_Variant",
            "Rights_Frame_Variant",
            "Anti_Corruption_Frame_Variant",
            "Identity_Frame_Variant",
            "Procedural_Fairness_Frame_Variant",
        ),
        "Adaptive_Audience_Copy": L(
            "High_Anger_Copy",
            "High_Fear_Copy",
            "High_Hope_Copy",
            "Low_Salience_Curiosity_Copy",
            "Procedural_Confusion_Copy",
        ),
        "Local_Context_Embedding": L(
            "Local_Place_Name_Embedding",
            "Local_Incident_Embedding",
            "Local_Elite_Name_Embedding",
            "Local_Symbol_Embedding",
            "Local_Humor_Embedding",
        ),
        "Norm_Sensitive_Packaging": L(
            "Rights_Defence_Packaging",
            "Public_Order_Packaging",
            "Patriotism_Packaging",
            "Community_Care_Packaging",
            "Anti_Elitism_Packaging",
        ),
    },
}


# ---------------------------------------------------------------------------
# 7. AMPLIFICATION VISIBILITY AND ATTENTION MANIPULATION
# ---------------------------------------------------------------------------
AMPLIFICATION_VISIBILITY = {
    "Bot_Cyborg_And_Account_Network_Amplification": {
        "Political_Bot_Amplification": L(
            "Repost_Bot_Amplification",
            "Reply_Bot_Amplification",
            "Hashtag_Bot_Amplification",
            "Follower_Growth_Bot_Amplification",
            "Newsjacking_Bot_Amplification",
        ),
        "Cyborg_Amplification": L(
            "Human_Queued_Cyborg_Posting",
            "Template_Driven_Cyborg_Replying",
            "Semi_Automated_Cross_Platform_Cyborging",
            "Manual_Intervention_Cyborging",
        ),
        "Reply_Farm_Operations": L(
            "Politician_Reply_Farms",
            "Journalist_Reply_Farms",
            "Institution_Reply_Farms",
            "Celebrity_Reply_Farms",
        ),
        "Follower_And_Engagement_Farming": L(
            "Engagement_Pod_Farming",
            "Clickfarm_Reaction_Farming",
            "Synthetic_Follower_Laddering",
            "Quote_Chain_Farming",
        ),
        "Cascade_Bootstrapping": L(
            "First_Minutes_Bootstrapping",
            "Trend_Threshold_Bootstrapping",
            "Cross_Platform_Bootstrapping",
            "Elite_Notice_Bootstrapping",
        ),
    },
    "Human_Swarm_And_Troll_Amplification": {
        "Volunteer_Swarm_Amplification": L(
            "Scheduled_Supporter_Swarming",
            "Movement_Call_To_Signal_Boosting",
            "Campaign_Surrogate_Swarming",
            "Issue_Day_Of_Action_Swarming",
        ),
        "Paid_Troll_Farms": L(
            "Comment_Section_Troll_Farming",
            "Inbox_Troll_Farming",
            "Forum_Troll_Farming",
            "Multilingual_Troll_Farming",
        ),
        "Brigading": L(
            "Thread_Brigading",
            "Vote_Manipulation_Brigading",
            "Livestream_Brigading",
            "Community_Takeover_Brigading",
        ),
        "Mass_Reporting_For_Visibility_Control": L(
            "Account_Suspension_Reporting",
            "Content_Takedown_Reporting",
            "Livestream_Interrupt_Reporting",
            "Group_Shutdown_Reporting",
        ),
        "Comment_Flooding": L(
            "Correction_Burying_Comment_Flooding",
            "Narrative_Repetition_Comment_Flooding",
            "Mockery_Comment_Flooding",
            "Procedural_Confusion_Comment_Flooding",
        ),
    },
    "Influencer_And_High_Reach_Relay": {
        "Creator_Relay": L(
            "Humor_Creator_Relay",
            "News_Explainer_Creator_Relay",
            "Streamer_Relay",
            "Lifestyle_Creator_Relay",
        ),
        "Celebrity_Or_Status_Relay": L(
            "Celebrity_Endorsement_Relay",
            "Athlete_Or_Artist_Relay",
            "Retired_Official_Status_Relay",
            "Business_Leader_Status_Relay",
        ),
        "Microinfluencer_Network_Relay": L(
            "Local_Microinfluencer_Relay",
            "Identity_Group_Microinfluencer_Relay",
            "Profession_Based_Microinfluencer_Relay",
            "Campus_Microinfluencer_Relay",
        ),
        "Ideological_Media_Relay": L(
            "Podcast_Relay",
            "Newsletter_Relay",
            "Talk_Show_Relay",
            "Partisan_Panel_Relay",
        ),
        "Cross_Border_Relay": L(
            "Diaspora_Creator_Relay",
            "Foreign_Language_Media_Relay",
            "Neighbour_State_Media_Relay",
            "Transnational_Movement_Relay",
        ),
    },
    "Attention_Capture_And_Retention_Mechanics": {
        "Salience_Spiking": L(
            "Shock_Drop_Salience_Spiking",
            "Coordinated_Burst_Salience_Spiking",
            "Elite_Reaction_Salience_Spiking",
            "Visual_Symbol_Salience_Spiking",
        ),
        "Repetition_And_Recurrence": L(
            "Cross_Format_Repetition",
            "Cross_Platform_Repetition",
            "Daily_Recurrence",
            "Trigger_Event_Recurrence",
        ),
        "Spectacle_And_Outrage_Formatting": L(
            "Scandal_Formatting",
            "Insult_And_Conflict_Formatting",
            "Emergency_Formatting",
            "Sacred_Violation_Formatting",
        ),
        "Gamified_Participation_Hooks": L(
            "Challenge_Hooks",
            "Poll_And_Quiz_Hooks",
            "Duet_Or_Remix_Hooks",
            "Badge_And_Identity_Hooks",
        ),
        "Serial_Episode_Delivery": L(
            "Document_Drop_Episode_Delivery",
            "Thread_Episode_Delivery",
            "Investigation_Episode_Delivery",
            "Countdown_Episode_Delivery",
        ),
    },
    "Amplification_Choreography": {
        "Burst_Synchronization": L(
            "Same_Minute_Posting",
            "Coordinated_Reply_Waves",
            "Hashtag_Threshold_Synchronization",
            "Cross_Platform_Opening_Burst",
        ),
        "Low_And_Slow_Amplification": L(
            "Staggered_Account_Posting",
            "Ambient_Repetition",
            "Intermittent_Reinforcement",
            "Slow_Cluster_Penetration",
        ),
        "Multi_Wave_Amplification": L(
            "Seed_Wave",
            "Validation_Wave",
            "Mass_Wave",
            "Sustainment_Wave",
        ),
        "Cross_Platform_Velocity_Transfer": L(
            "Messaging_To_Public_Transfer",
            "Short_Clip_To_Article_Transfer",
            "Livestream_To_Meme_Transfer",
            "Forum_To_Trend_Transfer",
        ),
        "Amplifier_Role_Specialization": L(
            "Seeders",
            "Validators",
            "Boosters",
            "Harassers",
            "Bridge_Accounts",
        ),
    },
}


# ---------------------------------------------------------------------------
# 8. SOCIAL PROOF NETWORK AND COMMUNITY MANIPULATION
# ---------------------------------------------------------------------------
SOCIAL_PROOF_NETWORK = {
    "Consensus_Fabrication": {
        "Fake_Majority_Signals": L(
            "Visible_Reply_Majority_Signals",
            "Reaction_Count_Majority_Signals",
            "Crowd_Photo_Majority_Signals",
            "Comment_Section_Majority_Signals",
        ),
        "Astroturf_Movements": L(
            "Petition_Astroturf",
            "Rally_Astroturf",
            "Citizen_Coalition_Astroturf",
            "Grassroots_Brand_Astroturf",
        ),
        "Manufactured_Petitions_And_Open_Letters": L(
            "Fake_Signature_Petitions",
            "Inflated_Open_Letters",
            "Credential_Stacked_Letters",
            "Community_Resolution_Simulations",
        ),
        "Poll_And_Sentiment_Spoofing": L(
            "Quick_Poll_Spoofing",
            "Sentiment_Widget_Spoofing",
            "Scoreboard_Spoofing",
            "Prediction_Market_Style_Spoofing",
        ),
        "Pseudo_Spontaneity": L(
            "Simulated_Bottom_Up_Outrage",
            "Simulated_Citizen_Discovery",
            "Simulated_Neighbourhood_Concern",
            "Simulated_Independent_Convergence",
        ),
    },
    "Network_Engineering_And_Bridge_Control": {
        "Bridge_Node_Construction": L(
            "Multi_Identity_Bridge_Accounts",
            "Cross_Issue_Bridge_Accounts",
            "Creator_Bridge_Accounts",
            "Language_Bridge_Accounts",
        ),
        "Fringe_To_Mainstream_Bridging": L(
            "Forum_To_Creator_Bridging",
            "Creator_To_Newsroom_Bridging",
            "Anonymous_Leak_To_Public_Commentary_Bridging",
            "Niche_Group_To_Party_Broker_Bridging",
        ),
        "Echo_Chamber_Reinforcement": L(
            "Selective_In_Group_Linking",
            "Outgroup_Blocking_Or_Filtering",
            "Ritual_Repetition_Loops",
            "Internal_Validator_Promotion",
        ),
        "Cluster_Synchronization": L(
            "Same_Frame_Multi_Group_Synchronization",
            "Same_Hashtag_Multi_Group_Synchronization",
            "Same_Enemy_Multi_Group_Synchronization",
            "Same_Action_Prompt_Multi_Group_Synchronization",
        ),
        "Broker_Capture": L(
            "Community_Broker_Capture",
            "Media_Broker_Capture",
            "Diaspora_Broker_Capture",
            "Movement_Broker_Capture",
        ),
    },
    "Community_Infiltration_And_Capture": {
        "Trusted_Member_Cultivation": L(
            "Long_Horizon_Trust_Building",
            "Helpful_Information_Seed_Building",
            "Shared_Identity_Performance_Building",
            "Moderation_Support_Building",
        ),
        "Moderator_Or_Admin_Capture": L(
            "Admin_Recruitment",
            "Admin_Pressure_Or_Blackmail",
            "Technical_Access_Capture",
            "Succession_Or_Absence_Capture",
        ),
        "Closed_Group_Recruitment": L(
            "Dm_Invite_Recruitment",
            "Private_Channel_Recruitment",
            "Exclusive_Information_Recruitment",
            "Trusted_Circle_Recruitment",
        ),
        "Diaspora_And_Identity_Space_Penetration": L(
            "Homeland_News_Space_Penetration",
            "Religious_Identity_Space_Penetration",
            "Ethnolinguistic_Space_Penetration",
            "Expatriate_Student_Space_Penetration",
        ),
        "Movement_Edge_Infiltration": L(
            "Volunteer_Onboarding_Infiltration",
            "Event_Logistics_Infiltration",
            "Discord_Or_Signal_Edge_Infiltration",
            "Campus_Chapter_Infiltration",
        ),
    },
    "Harassment_Norm_Setting_And_Silencing": {
        "Dogpiling": L(
            "Public_Quote_Dogpiling",
            "Inbox_Dogpiling",
            "Comment_Thread_Dogpiling",
            "Livestream_Dogpiling",
        ),
        "Reputation_Smearing": L(
            "Extremist_Label_Smearing",
            "Foreign_Agent_Smearing",
            "Corrupt_Actor_Smearing",
            "Inauthentic_Person_Smearing",
        ),
        "Chilling_Effect_Operations": L(
            "Career_Risk_Chilling",
            "Social_Exclusion_Chilling",
            "Family_Safety_Chilling",
            "Legal_Or_Process_Chilling",
        ),
        "Defector_Punishment": L(
            "Apostate_Mockery",
            "Public_Exposure_Of_Dissenters",
            "Purity_Test_Enforcement",
            "Collaborator_Labeling",
        ),
        "Mass_Mockery_And_Ridicule": L(
            "Meme_Ridicule",
            "Nickname_Ridicule",
            "Voice_Or_Appearance_Ridicule",
            "Competence_Ridicule",
        ),
    },
    "Social_Proof_Optimization": {
        "Engagement_Pods": L(
            "Creator_Engagement_Pods",
            "Campaign_Engagement_Pods",
            "Issue_Group_Engagement_Pods",
            "Anonymous_Relay_Engagement_Pods",
        ),
        "Fake_Endorsements": L(
            "Elite_Fake_Endorsements",
            "Community_Leader_Fake_Endorsements",
            "Expert_Fake_Endorsements",
            "Former_Opponent_Fake_Endorsements",
        ),
        "Manufactured_Testimonials": L(
            "Local_Parent_Testimonials",
            "Small_Business_Testimonials",
            "Public_Servant_Testimonials",
            "Student_Testimonials",
        ),
        "Visible_Elite_Support_Simulation": L(
            "Staff_Whisper_Simulation",
            "Insider_Briefing_Simulation",
            "Cross_Party_Support_Simulation",
            "Media_Off_The_Record_Support_Simulation",
        ),
        "Social_Identity_Cue_Stacking": L(
            "Flag_And_Symbol_Cue_Stacking",
            "Dress_And_Style_Cue_Stacking",
            "Slogan_Cue_Stacking",
            "Ritual_Participation_Cue_Stacking",
        ),
    },
}


# ---------------------------------------------------------------------------
# 9. PLATFORM INFORMATION ENVIRONMENT AND MEDIA SYSTEM SHAPING
# ---------------------------------------------------------------------------
PLATFORM_MEDIA_SHAPING = {
    "Information_Space_Pollution": {
        "Content_Flooding": L(
            "Copy_Variant_Flooding",
            "Format_Variant_Flooding",
            "Multi_Language_Flooding",
            "Cross_Channel_Flooding",
        ),
        "Contradiction_Flooding": L(
            "Multi_Explanation_Flooding",
            "Multi_Suspect_Flooding",
            "Multi_Motive_Flooding",
            "Multi_Outcome_Flooding",
        ),
        "Noise_Saturation": L(
            "Low_Quality_Article_Saturation",
            "Meme_Saturation",
            "Comment_Noise_Saturation",
            "Video_Clip_Saturation",
        ),
        "Agenda_Distraction": L(
            "Adjacent_Scandal_Distraction",
            "Celebrity_Conflict_Distraction",
            "Symbolic_Controversy_Distraction",
            "Procedural_Side_Issue_Distraction",
        ),
        "Attention_Denial": L(
            "Moderator_Overload",
            "Fact_Checker_Overload",
            "Journalist_Tipline_Overload",
            "Community_Admin_Overload",
        ),
    },
    "Source_Ecology_Manipulation": {
        "Clone_News_Sites": L(
            "Local_Clone_Sites",
            "National_Clone_Sites",
            "Single_Issue_Clone_Sites",
            "Breaking_News_Clone_Sites",
        ),
        "Lookalike_Domains_And_Handles": L(
            "Typosquat_Domains",
            "Brand_Mimic_Handles",
            "Regional_Subbrand_Mimics",
            "Verification_Lookalike_Handles",
        ),
        "Citation_Circularity": L(
            "Site_To_Site_Circularity",
            "Article_To_Post_Circularity",
            "Podcast_To_Blog_Circularity",
            "Quote_Card_To_Article_Circularity",
        ),
        "Cross_Reference_Echoing": L(
            "Same_Claim_Multi_Site_Echoing",
            "Same_Quote_Multi_Site_Echoing",
            "Same_Document_Multi_Site_Echoing",
            "Same_Visual_Multi_Site_Echoing",
        ),
        "Stacked_Outlet_Architecture": L(
            "Local_Plus_Expert_Plus_Breaking_Stack",
            "Main_Language_Plus_Diaspora_Language_Stack",
            "Article_Plus_Video_Plus_Meme_Stack",
            "Archive_Plus_News_Plus_Commentary_Stack",
        ),
    },
    "Search_Archive_And_Retrieval_Manipulation": {
        "Search_Pollution": L(
            "Query_Result_Pollution",
            "Image_Search_Pollution",
            "News_Tab_Pollution",
            "Question_Answer_Pollution",
        ),
        "Document_Repository_Seeding": L(
            "Open_Document_Repository_Seeding",
            "Community_Drive_Seeding",
            "Public_Comment_Attachment_Seeding",
            "Open_Data_Portal_Style_Seeding",
        ),
        "Cache_And_Preview_Exploitation": L(
            "Snippet_Preservation_Exploitation",
            "Cached_Copy_Exploitation",
            "Social_Preview_Card_Exploitation",
            "Translation_Preview_Exploitation",
        ),
        "Ai_Answer_Poisoning": L(
            "Faq_Corpus_Poisoning",
            "Reference_Style_Poisoning",
            "Consensus_Summary_Poisoning",
            "Quote_Bank_Poisoning",
        ),
        "Reference_Memory_Planting": L(
            "Wiki_Style_Memory_Planting",
            "Explainers_Memory_Planting",
            "Timeline_Memory_Planting",
            "Backgrounder_Memory_Planting",
        ),
    },
    "Platform_Rule_And_Moderation_Exploitation": {
        "Boundary_Pushing_Variation": L(
            "Spelling_Variation_Evasion",
            "Visual_Overlay_Evasion",
            "Sarcasm_Or_Joke_Cover_Evasion",
            "Coded_Language_Evasion",
        ),
        "Ban_Evasion_Infrastructure": L(
            "Reserve_Account_Ladders",
            "Backup_Channel_Ladders",
            "Mirror_Domain_Ladders",
            "Reentry_Link_Hubs",
        ),
        "Platform_Hopping": L(
            "High_Moderation_To_Low_Moderation_Hopping",
            "Public_To_Private_Hopping",
            "Video_To_Text_Hopping",
            "Domestic_To_Foreign_Platform_Hopping",
        ),
        "Jurisdiction_Shifting": L(
            "Hosting_Jurisdiction_Shifting",
            "Entity_Registration_Shifting",
            "Payment_And_Ad_Buy_Shifting",
            "Moderation_Forum_Shifting",
        ),
        "Policy_Asymmetry_Exploitation": L(
            "Cross_Platform_Content_Policy_Gaps",
            "Language_Moderation_Gaps",
            "Appeal_Process_Gaps",
            "Public_Interest_Exception_Gaps",
        ),
    },
    "Media_System_Interference": {
        "Pseudo_Events": L(
            "Press_Conference_Pseudo_Events",
            "Street_Spectacle_Pseudo_Events",
            "Symbol_Drop_Pseudo_Events",
            "Document_Release_Pseudo_Events",
        ),
        "News_Cycle_Hijacking": L(
            "Deadline_Hijacking",
            "Debate_Hijacking",
            "Breaking_News_Hijacking",
            "Morning_Show_Hijacking",
        ),
        "Journalist_Tasking_Pressure": L(
            "Tipline_Pressure",
            "Coordinated_Question_Pressure",
            "Editorial_Inbox_Pressure",
            "Access_Leverage_Pressure",
        ),
        "Content_Farm_Syndication": L(
            "Headline_Swarm_Syndication",
            "Local_Angle_Syndication",
            "Translation_Syndication",
            "Creator_Clip_Syndication",
        ),
        "Quoteable_Conflict_Production": L(
            "Insult_Clip_Production",
            "Walkout_Or_Confrontation_Production",
            "Symbolic_Violation_Production",
            "False_Binary_Question_Production",
        ),
    },
}


# ---------------------------------------------------------------------------
# 10. CYBER ENABLED COMPROMISE COERCION AND DISRUPTION
# ---------------------------------------------------------------------------
CYBER_COMPROMISE = {
    "Compromise_And_Expose": {
        "Phishing_For_Political_Material": L(
            "Candidate_Targeted_Phishing",
            "Party_Staff_Targeted_Phishing",
            "Journalist_Targeted_Phishing",
            "Activist_Targeted_Phishing",
            "Administrator_Targeted_Phishing",
            "Vendor_Supply_Chain_Phishing",
        ),
        "Mailbox_And_Cloud_Exfiltration": L(
            "Mailbox_Exfiltration",
            "Drive_Exfiltration",
            "Contact_Graph_Exfiltration",
            "Calendar_And_Schedule_Exfiltration",
            "Draft_Document_Exfiltration",
        ),
        "Timed_Leak_Release": L(
            "Debate_Timed_Leaks",
            "Vote_Timed_Leaks",
            "Scandal_Reinforcement_Timed_Leaks",
            "Deadline_Timed_Leaks",
        ),
        "Selective_Context_Suppression": L(
            "Exculpatory_Context_Removal",
            "Timeline_Context_Removal",
            "Recipient_Context_Removal",
            "Draft_Status_Context_Removal",
        ),
        "Real_Fake_Blended_Leaks": L(
            "Real_Thread_Fake_Insert_Blends",
            "Real_Document_Fake_Appendix_Blends",
            "Real_Log_Fake_Summary_Blends",
            "Real_Archive_Fake_Index_Blends",
        ),
    },
    "Trusted_Channel_Hijacking": {
        "Account_Takeover_For_False_Statements": L(
            "Politician_Account_Takeover",
            "Party_Account_Takeover",
            "Journalist_Account_Takeover",
            "Institution_Account_Takeover",
        ),
        "Website_Defacement_For_Message_Injection": L(
            "Party_Site_Defacement",
            "Campaign_Site_Defacement",
            "Local_Government_Site_Defacement",
            "News_Site_Defacement",
        ),
        "Sms_Email_And_Notification_Spoofing": L(
            "Official_Notice_Spoofing",
            "Campaign_Appeal_Spoofing",
            "Procedural_Deadline_Spoofing",
            "Security_Warning_Spoofing",
        ),
        "Content_Management_System_Abuse": L(
            "Homepage_Banner_Injection",
            "Article_Body_Injection",
            "Alert_Bar_Injection",
            "Embedded_Media_Injection",
        ),
        "Deep_Link_And_Qr_Redirection": L(
            "Event_Qr_Redirection",
            "Civic_Help_Link_Redirection",
            "News_Link_Redirection",
            "Campaign_Signup_Redirection",
        ),
    },
    "Coercive_Signalling_And_Intimidation": {
        "Doxxing": L(
            "Address_Exposure_Doxxing",
            "Employer_Exposure_Doxxing",
            "Family_Linkage_Doxxing",
            "Contact_Detail_Doxxing",
        ),
        "Blackmail_Using_Compromised_Material": L(
            "Sexual_Or_Intimate_Blackmail",
            "Career_Risk_Blackmail",
            "Corruption_Allegation_Blackmail",
            "Private_Opinion_Exposure_Blackmail",
        ),
        "Threat_Messaging": L(
            "Direct_Violence_Threat_Messaging",
            "Career_Threat_Messaging",
            "Social_Exclusion_Threat_Messaging",
            "Legal_Or_Process_Threat_Messaging",
        ),
        "Surveillance_Signalling": L(
            "Location_Awareness_Signalling",
            "Device_Compromise_Signalling",
            "Family_Or_Network_Awareness_Signalling",
            "Meeting_Awareness_Signalling",
        ),
        "Family_And_Network_Pressure": L(
            "Relative_Targeting_Pressure",
            "Coworker_Targeting_Pressure",
            "Community_Reputation_Pressure",
            "Organisational_Membership_Pressure",
        ),
    },
    "Disruption_For_Interpretive_Effect": {
        "Ddos_On_Political_Targets": L(
            "Party_Site_Ddos",
            "Watchdog_Site_Ddos",
            "Media_Site_Ddos",
            "Civic_Information_Site_Ddos",
        ),
        "Civic_Information_Service_Disruption": L(
            "Voter_Information_Disruption",
            "Complaint_Channel_Disruption",
            "Registration_Guidance_Disruption",
            "Result_Information_Disruption",
        ),
        "Media_Service_Disruption": L(
            "Newsroom_Publication_Disruption",
            "Broadcast_Stream_Disruption",
            "Archive_Access_Disruption",
            "Comment_Moderation_Disruption",
        ),
        "Campaign_Ops_Disruption": L(
            "Field_Tool_Disruption",
            "Volunteer_Comms_Disruption",
            "Donation_Flow_Disruption",
            "Schedule_Management_Disruption",
        ),
        "Symbolic_Infrastructure_Disruption": L(
            "Public_Dashboard_Disruption",
            "Institutional_Status_Page_Disruption",
            "Official_Announcement_Channel_Disruption",
            "Public_Record_Lookup_Disruption",
        ),
    },
    "False_Flag_And_Attribution_Manipulation": {
        "Staged_Hack_Claims": L(
            "Fabricated_Breach_Claims",
            "Overstated_Intrusion_Claims",
            "Memeified_Hacker_Claims",
            "Anonymous_Collective_Claims",
        ),
        "Forensic_Trace_Planting": L(
            "Language_Trace_Planting",
            "Tooling_Trace_Planting",
            "Timing_Trace_Planting",
            "Infrastructure_Trace_Planting",
        ),
        "Narrative_First_Attribution": L(
            "Instant_Blaming_Before_Evidence",
            "Prewritten_Enemy_Assignment",
            "Identity_Fault_Line_Attribution",
            "Policy_Outcome_Attribution",
        ),
        "Compromise_Plus_Propaganda_Pairing": L(
            "Hack_As_Proof_Pairing",
            "Leak_As_Story_Anchor_Pairing",
            "Defacement_As_Symbol_Pairing",
            "Outage_As_Regime_Failure_Pairing",
        ),
        "Technical_Theater": L(
            "Code_Screen_Theater",
            "Terminal_Screenshot_Theater",
            "Dark_Web_Reference_Theater",
            "Anonymous_Mask_Theater",
        ),
    },
}


# ---------------------------------------------------------------------------
# 11. BEHAVIORAL CONVERSION MOBILIZATION SUPPRESSION RADICALIZATION
# ---------------------------------------------------------------------------
BEHAVIORAL_CONVERSION = {
    "Opinion_Conversion": {
        "Issue_Position_Shift": L(
            "Policy_Support_Shift",
            "Policy_Opposition_Shift",
            "Risk_Perception_Shift",
            "Causal_Interpretation_Shift",
        ),
        "Candidate_Or_Party_Preference_Shift": L(
            "Incumbent_Support_Shift",
            "Challenger_Support_Shift",
            "Third_Party_Shift",
            "Anti_Candidate_Negative_Preference_Shift",
        ),
        "Coalition_Fragmentation": L(
            "Issue_Wedge_Fragmentation",
            "Identity_Wedge_Fragmentation",
            "Strategy_Wedge_Fragmentation",
            "Leadership_Trust_Fragmentation",
        ),
        "Protest_Vote_Redirection": L(
            "Spoiler_Vote_Redirection",
            "Blank_Or_Invalid_Vote_Redirection",
            "Abstention_Redirection",
            "Symbolic_Vote_Redirection",
        ),
        "Preference_Hardening": L(
            "Identity_Bound_Hardening",
            "Moralised_Hardening",
            "Conspiratorial_Hardening",
            "Grievance_Hardening",
        ),
    },
    "Participation_Manipulation": {
        "Turnout_Suppression": L(
            "Cynicism_Based_Suppression",
            "Fear_Based_Suppression",
            "Procedural_Confusion_Suppression",
            "Efficacy_Collapse_Suppression",
        ),
        "Turnout_Activation": L(
            "Outrage_Activation",
            "Hope_Activation",
            "Identity_Threat_Activation",
            "Last_Minute_Emergency_Activation",
        ),
        "Registration_And_Procedural_Discouragement": L(
            "Deadline_Confusion_Discouragement",
            "Document_Requirement_Discouragement",
            "Queue_Expectation_Discouragement",
            "Eligibility_Confusion_Discouragement",
        ),
        "Queue_Confusion_And_Timing_Effects": L(
            "Wrong_Time_Guidance",
            "Wrong_Place_Guidance",
            "Peak_Crowding_Guidance",
            "Deadline_Misperception_Guidance",
        ),
        "Participation_Misdirection": L(
            "Symbolic_Action_Substitution",
            "Ineffective_Channel_Substitution",
            "Premature_Action_Substitution",
            "Misfocused_Target_Substitution",
        ),
    },
    "Collective_Action_Shaping": {
        "Protest_Mobilization": L(
            "Flash_Protest_Mobilization",
            "Sustained_Rally_Mobilization",
            "Campus_Protest_Mobilization",
            "Identity_Group_Mobilization",
        ),
        "Countermobilization": L(
            "Counterrally_Mobilization",
            "Defensive_Neighbourhood_Mobilization",
            "Symbolic_Counter_Action_Mobilization",
            "Online_To_Offline_Countermobilization",
        ),
        "Flashcrowd_Coordination": L(
            "Rapid_Assembly_Coordination",
            "Location_Swap_Coordination",
            "Symbol_Drop_Coordination",
            "Live_Stream_Triggered_Coordination",
        ),
        "Hashtag_To_Street_Transition": L(
            "Event_Call_Conversion",
            "Peer_Confirmation_Conversion",
            "Map_And_Route_Conversion",
            "Visual_Proof_Conversion",
        ),
        "Action_Retiming": L(
            "Institutionally_Disruptive_Retiming",
            "Media_Visibility_Retiming",
            "Law_Enforcement_Overstretch_Retiming",
            "Countermessage_Preemption_Retiming",
        ),
    },
    "Polarization_And_Radicalization": {
        "Outgroup_Threat_Escalation": L(
            "Security_Outgroup_Threat_Escalation",
            "Moral_Outgroup_Threat_Escalation",
            "Demographic_Outgroup_Threat_Escalation",
            "Treasonous_Outgroup_Threat_Escalation",
        ),
        "Identity_Hardening": L(
            "Purity_Test_Identity_Hardening",
            "Sacrifice_Identity_Hardening",
            "Embattled_Group_Identity_Hardening",
            "History_And_Memory_Identity_Hardening",
        ),
        "Conspiracy_Recruitment": L(
            "Entry_Level_Conspiracy_Recruitment",
            "Community_Bound_Conspiracy_Recruitment",
            "Self_Sealing_Worldview_Recruitment",
            "Apocalyptic_Conspiracy_Recruitment",
        ),
        "Normative_Desensitization": L(
            "Hostile_Rhetoric_Desensitization",
            "Dehumanising_Language_Desensitization",
            "Procedural_Sabotage_Desensitization",
            "Political_Violence_Hint_Desensitization",
        ),
        "Accelerationist_Cueing": L(
            "Collapse_Inevitability_Cueing",
            "Cleansing_Conflict_Cueing",
            "Provocation_For_Reaction_Cueing",
            "Martyrdom_And_Sacrifice_Cueing",
        ),
    },
    "Elite_Behavior_Shaping": {
        "Agenda_Pressure_On_Politicians": L(
            "Forced_Statement_Pressure",
            "Forced_Policy_Position_Pressure",
            "Forced_Visit_Or_Appearance_Pressure",
            "Forced_Intraparty_Response_Pressure",
        ),
        "Media_Editorial_Pressure": L(
            "Headline_Choice_Pressure",
            "Coverage_Volume_Pressure",
            "Framing_Pressure",
            "Guest_Booking_Pressure",
        ),
        "Administrative_Pressure": L(
            "Public_Guidance_Change_Pressure",
            "Security_Posture_Change_Pressure",
            "Service_Delivery_Change_Pressure",
            "Procedural_Delay_Pressure",
        ),
        "Judicial_Or_Regulatory_Pressure": L(
            "Perceived_Legitimacy_Pressure",
            "Procedural_Fairness_Pressure",
            "Public_Backlash_Pressure",
            "Deadline_Or_Timing_Pressure",
        ),
        "Intraparty_Trust_Breakdown": L(
            "Leak_Expectation_Breakdown",
            "Factional_Suspicion_Breakdown",
            "Advisor_Discrediting_Breakdown",
            "Donor_Or_Broker_Suspicion_Breakdown",
        ),
    },
}


# ---------------------------------------------------------------------------
# 12. COGNITIVE INFRASTRUCTURE AND EPISTEMIC SUBVERSION
# ---------------------------------------------------------------------------
COGNITIVE_INFRASTRUCTURE = {
    "Epistemic_Trust_Demobilisation": L(
        "Pervasive_Doubt_Engineering",
        "Information_Overload_Verification_Fatigue",
        "Source_Authority_Relativisation",
        "Whataboutism_Cascade_Demobilisation",
        "Public_Service_Media_Trust_Erosion",
        "Independent_Fact_Checker_Trust_Erosion",
        "Scientific_Authority_Trust_Erosion",
    ),
    "Epistemic_Routine_Disruption": L(
        "Verification_Habit_Erosion",
        "Cross_Source_Triangulation_Erosion",
        "Lateral_Reading_Habit_Erosion",
        "Citation_Following_Habit_Erosion",
        "Reverse_Image_Search_Habit_Erosion",
    ),
    "Reality_Anchor_Erosion": L(
        "Shared_Calendar_Erosion",
        "Common_Vocabulary_Erosion",
        "Reference_Source_Erosion",
        "Public_Record_Trust_Erosion",
        "Institutional_Memory_Erosion",
    ),
    "Cognitive_Load_Manipulation": L(
        "Decision_Fatigue_Induction",
        "Choice_Paralysis_Induction",
        "Multitasking_Saturation",
        "Notification_Fatigue_Induction",
    ),
    "Synthetic_Consensus_Engineering": L(
        "Cross_Platform_Synthetic_Consensus",
        "Cross_Lingual_Synthetic_Consensus",
        "Cross_Source_Synthetic_Consensus",
        "Time_Stretched_Synthetic_Consensus",
    ),
    "Counter_Epistemic_Toolkit_Subversion": L(
        "Anti_Fact_Checking_Counter_Toolkit",
        "Anti_Open_Source_Investigation_Counter_Toolkit",
        "Anti_Forensic_Verification_Counter_Toolkit",
        "Anti_Detection_Tooling_Counter_Toolkit",
    ),
}


# ---------------------------------------------------------------------------
# 13. OPERATIONAL SECURITY EVASION PERSISTENCE RECONSTITUTION
# ---------------------------------------------------------------------------
OPERATIONAL_SECURITY = {
    "Infrastructure_Resilience": {
        "Account_Rotation": L(
            "Planned_Rotation",
            "Enforcement_Triggered_Rotation",
            "Role_Based_Rotation",
            "Platform_Specific_Rotation",
        ),
        "Persona_Resurrection": L(
            "Visual_Refresh_Resurrection",
            "Name_Variant_Resurrection",
            "Backstory_Adjusted_Resurrection",
            "Community_Reentry_Resurrection",
        ),
        "Content_Mirroring": L(
            "Multi_Domain_Mirroring",
            "Multi_Channel_Mirroring",
            "Archive_Link_Mirroring",
            "Asset_Pack_Mirroring",
        ),
        "Audience_Migration_Paths": L(
            "Backup_Channel_Migration",
            "Newsletter_Migration",
            "Private_Group_Migration",
            "Mirror_Domain_Migration",
        ),
        "Redundant_Operational_Roles": L(
            "Backup_Seeders",
            "Backup_Validators",
            "Backup_Translators",
            "Backup_Bridge_Accounts",
        ),
    },
    "Detection_Evasion": {
        "Linguistic_Mutation": L(
            "Spelling_Mutation",
            "Syntax_Mutation",
            "Hashtag_Mutation",
            "Coded_Phrase_Mutation",
        ),
        "Image_Text_Mutation": L(
            "Crop_Mutation",
            "Color_Overlay_Mutation",
            "Font_And_Layout_Mutation",
            "Ocr_Resistance_Mutation",
        ),
        "Low_And_Slow_Coordination": L(
            "Staggered_Posting_Evasion",
            "Distributed_Reply_Evasion",
            "Ambient_Presence_Evasion",
            "Rotation_Across_Days_Evasion",
        ),
        "Human_Automation_Blending": L(
            "Manual_Reply_Humanisation",
            "Human_In_The_Loop_Editing",
            "Behavioural_Jitter_Insertion",
            "Schedule_Variability_Blending",
        ),
        "Artifact_Cleanup_And_Trace_Removal": L(
            "Post_Delete_Cleanup",
            "Metadata_Cleanup",
            "Link_Rot_Cleanup",
            "Role_Account_Retirement_Cleanup",
        ),
    },
    "Attribution_Obfuscation": {
        "Cutout_Layering": L(
            "Media_Cutout_Layering",
            "Contractor_Cutout_Layering",
            "Community_Cutout_Layering",
            "Diaspora_Cutout_Layering",
        ),
        "Contractor_Intermediation": L(
            "Content_Contract_Intermediation",
            "Ad_Buy_Intermediation",
            "Engagement_Farm_Intermediation",
            "Translation_Intermediation",
        ),
        "Jurisdictional_Shielding": L(
            "Hosting_Shielding",
            "Domain_Registration_Shielding",
            "Payment_Rail_Shielding",
            "Legal_Entity_Shielding",
        ),
        "False_Flag_Styling": L(
            "Adversary_Visual_Styling",
            "Adversary_Language_Styling",
            "Adversary_Timing_Styling",
            "Adversary_Target_Choice_Styling",
        ),
        "Mixed_Actor_Blending": L(
            "Foreign_Domestic_Blending",
            "Authentic_Inauthentic_Blending",
            "Volunteer_Paid_Blending",
            "Media_Creator_Blending",
        ),
    },
    "Narrative_And_Asset_Mutation": {
        "Frame_Mutation": L(
            "Security_To_Economy_Mutation",
            "Economy_To_Identity_Mutation",
            "Corruption_To_Procedural_Mutation",
            "Outrage_To_Cynicism_Mutation",
        ),
        "Messenger_Substitution": L(
            "Expert_To_Peer_Substitution",
            "Peer_To_Creator_Substitution",
            "Creator_To_News_Substitution",
            "Anonymous_To_Whistleblower_Substitution",
        ),
        "Format_Substitution": L(
            "Article_To_Clip_Substitution",
            "Clip_To_Meme_Substitution",
            "Document_To_Thread_Substitution",
            "Thread_To_Testimonial_Substitution",
        ),
        "Topic_Adjacent_Repackaging": L(
            "Migration_To_Security_Repackaging",
            "Inflation_To_Corruption_Repackaging",
            "Public_Health_To_Freedom_Repackaging",
            "Local_Service_Failure_To_Regime_Failure_Repackaging",
        ),
        "Plausibility_Recalibration": L(
            "Toned_Down_Claim_Recalibration",
            "Wink_And_Nod_Recalibration",
            "Evidence_Style_Recalibration",
            "Humor_Cover_Recalibration",
        ),
    },
    "Post_Exposure_Reconstitution": {
        "Identity_Reset": L(
            "Persona_Retirement_And_Replacement",
            "New_Visual_Identity_Reset",
            "New_Language_Register_Reset",
            "New_Platform_Home_Reset",
        ),
        "Network_Partial_Rebuild": L(
            "Bridge_First_Rebuild",
            "Validator_First_Rebuild",
            "Community_Core_Rebuild",
            "Translation_And_Relay_Rebuild",
        ),
        "Audience_Retrust_Building": L(
            "Low_Conflict_Content_Retrust",
            "Service_And_Helpful_Content_Retrust",
            "Self_Victimisation_Retrust",
            "Slow_Reengagement_Retrust",
        ),
        "Deflection_And_Counteraccusation": L(
            "Censorship_Counteraccusation",
            "Partisan_Hunt_Counteraccusation",
            "Foreign_Smear_Counteraccusation",
            "Elite_Silencing_Counteraccusation",
        ),
        "Memory_Decay_Exploitation": L(
            "Attention_Fade_Waiting",
            "Personnel_Turnover_Exploitation",
            "New_Issue_Cover_Exploitation",
            "Historical_Revision_Reentry",
        ),
    },
}


# ---------------------------------------------------------------------------
# 14. MEASUREMENT EXPERIMENTATION AND ADAPTIVE LEARNING
# ---------------------------------------------------------------------------
MEASUREMENT_EXPERIMENTATION = {
    "Performance_Measurement": {
        "Reach_And_Exposure_Measurement": L(
            "Raw_Impression_Measurement",
            "Unique_Reach_Measurement",
            "Frequency_Measurement",
            "Channel_Pathway_Measurement",
        ),
        "Engagement_Pattern_Measurement": L(
            "Reaction_Velocity_Measurement",
            "Reply_Depth_Measurement",
            "Share_Chain_Measurement",
            "High_Value_Amplifier_Measurement",
        ),
        "Cross_Group_Penetration_Measurement": L(
            "Cluster_Escape_Measurement",
            "Diaspora_Penetration_Measurement",
            "Mainstream_Penetration_Measurement",
            "Elite_Attention_Measurement",
        ),
        "Narrative_Uptake_Measurement": L(
            "Keyword_Adoption_Measurement",
            "Frame_Adoption_Measurement",
            "Meme_Template_Adoption_Measurement",
            "Quote_Repetition_Measurement",
        ),
        "Behavioural_Proxy_Measurement": L(
            "Turnout_Intent_Proxy_Measurement",
            "Protest_Attendance_Proxy_Measurement",
            "Donation_Proxy_Measurement",
            "Elite_Statement_Proxy_Measurement",
        ),
    },
    "Ab_Testing_And_Variant_Experimentation": {
        "Headline_And_Hook_Testing": L(
            "Fear_Hook_Testing",
            "Anger_Hook_Testing",
            "Curiosity_Hook_Testing",
            "Betrayal_Hook_Testing",
        ),
        "Frame_Testing": L(
            "Threat_Frame_Testing",
            "Corruption_Frame_Testing",
            "Identity_Frame_Testing",
            "Hope_Frame_Testing",
        ),
        "Messenger_Testing": L(
            "Expert_Messenger_Testing",
            "Peer_Messenger_Testing",
            "Creator_Messenger_Testing",
            "Whistleblower_Messenger_Testing",
        ),
        "Format_Testing": L(
            "Video_Vs_Text_Testing",
            "Thread_Vs_Article_Testing",
            "Meme_Vs_Testimonial_Testing",
            "Document_Vs_Clip_Testing",
        ),
        "Timing_Testing": L(
            "Morning_Vs_Evening_Testing",
            "Weekday_Vs_Weekend_Testing",
            "Single_Drop_Vs_Serial_Drop_Testing",
            "Preemptive_Vs_Reactive_Testing",
        ),
    },
    "Defender_And_Responder_Learning": {
        "Moderation_Threshold_Mapping": L(
            "Content_Threshold_Mapping",
            "Velocity_Threshold_Mapping",
            "Network_Signature_Threshold_Mapping",
            "Language_Specific_Threshold_Mapping",
        ),
        "Journalistic_Reaction_Mapping": L(
            "Coverage_Trigger_Mapping",
            "Debunk_Trigger_Mapping",
            "Human_Interest_Trigger_Mapping",
            "Conflict_Trigger_Mapping",
        ),
        "Fact_Checking_Resistance_Analysis": L(
            "Correction_Resistant_Claim_Analysis",
            "Mutating_Claim_Analysis",
            "Identity_Protected_Claim_Analysis",
            "Humor_Shielded_Claim_Analysis",
        ),
        "Institutional_Response_Latency_Mapping": L(
            "Election_Office_Latency_Mapping",
            "Party_Response_Latency_Mapping",
            "Regulator_Latency_Mapping",
            "Local_Authority_Latency_Mapping",
        ),
        "Opposition_Countermobilization_Analysis": L(
            "Backlash_Benefit_Analysis",
            "Martyrdom_Conversion_Analysis",
            "Overreaction_Detection_Analysis",
            "Countermessage_Diffusion_Analysis",
        ),
    },
    "State_Space_Recalibration": {
        "Objective_Reprioritization": L(
            "Persuasion_To_Distrust_Shift",
            "Mobilization_To_Confusion_Shift",
            "Turnout_To_Fragmentation_Shift",
            "Issue_Conversion_To_Identity_Hardening_Shift",
        ),
        "Segment_Reweighting": L(
            "High_Moveability_Segment_Reweighting",
            "Elite_Pressure_Segment_Reweighting",
            "Diaspora_Segment_Reweighting",
            "Low_Scrutiny_Segment_Reweighting",
        ),
        "Tactic_Mix_Reweighting": L(
            "Relay_Heavy_Reweighting",
            "Artifact_Heavy_Reweighting",
            "Microtargeting_Heavy_Reweighting",
            "Community_Capture_Heavy_Reweighting",
        ),
        "Risk_Reward_Recalibration": L(
            "High_Visibility_Low_Return_Tactic_Dropping",
            "Low_Risk_High_Return_Tactic_Scaling",
            "Attribution_Risk_Recalibration",
            "Platform_Enforcement_Risk_Recalibration",
        ),
        "Longitudinal_Campaign_Memory": L(
            "Persistent_Audience_Memory",
            "Persistent_Messenger_Memory",
            "Persistent_Trigger_Memory",
            "Persistent_Countermeasure_Memory",
        ),
    },
}


# ---------------------------------------------------------------------------
# 15. NARRATIVE INFRASTRUCTURE AND ECOSYSTEM CAPTURE
# ---------------------------------------------------------------------------
# SOURCE: EEAS 3rd FIMI Threat Report (2025), DISARM T0102/T0102.003/T0023,
# NATO ACT Cognitive Warfare Concept (2024), NewsGuard AI Tracking Center (2025).
NARRATIVE_INFRASTRUCTURE = {
    "Narrative_Prepositioning_And_Dormant_Seeding": {
        "Dormant_Narrative_Reservoirs": L(
            "Multi_Year_Narrative_Reservoir_Seeding",
            "Anniversary_Activation_Narrative_Seeding",
            "Latent_Crisis_Frame_Seeding",
            "Future_Scandal_Template_Seeding",
            "Diaspora_Memory_Prepositioning",
        ),
        "Semantic_Term_Shift": L(
            "Controlled_Political_Term_Redescription",
            "Euphemism_Normalisation_Seeding",
            "Pejorative_Relabeling_Drift",
            "Policy_Label_Meaning_Drift",
            "Hashtag_Semantic_Anchoring",
        ),
        "Overton_Window_Preconditioning": L(
            "Peripheral_Claim_Trial_Ballooning",
            "Extremity_Anchor_Normalisation",
            "Respectability_Laddering",
            "Expert_Forum_Preconditioning",
        ),
    },
    "Ecosystem_Capture_And_Authority_Substitution": {
        "Reference_Source_Seeding": L(
            "Wiki_Adjacent_Backgrounder_Seeding",
            "Local_History_Page_Seeding",
            "Explainer_Site_Backfill_Seeding",
            "Pseudo_Archive_Timeline_Seeding",
        ),
        "Knowledge_Graph_Context_Capture": L(
            "Entity_Relationship_Context_Seeding",
            "Alias_And_Synonym_Context_Seeding",
            "Source_Citation_Context_Seeding",
            "Event_Timeline_Context_Seeding",
        ),
        "Influence_Ecosystem_Dependency": L(
            "Creator_Grant_Dependency_Creation",
            "Local_Media_Content_Subsidy",
            "Research_Network_Citation_Dependency",
            "Civil_Society_Resource_Dependency",
        ),
    },
    "Strategic_Silence_And_Counter_Narrative_Denial": {
        "Counter_Narrative_Suppression": L(
            "Correction_Visibility_Suppression",
            "Alternative_Witness_Silencing",
            "Local_Context_Withholding",
            "Counter_Evidence_Attention_Denial",
        ),
        "Non_Response_Exploitation": L(
            "Ambiguity_Preservation_Silence",
            "Verification_Delay_Silence",
            "Crisis_Information_Vacuum_Maintenance",
            "Denial_Without_Substance",
        ),
        "Search_And_Archive_Absence": L(
            "Negative_Search_Result_Engineering",
            "Counter_Source_Deindexing_Pressure",
            "Archive_Gap_Exploitation",
            "Broken_Link_Memory_Holing",
        ),
    },
}


# ---------------------------------------------------------------------------
# 16. ECONOMIC AND REGULATORY PRESSURE WEAPONISATION
# ---------------------------------------------------------------------------
# SOURCE: Carnegie Endowment "Can Democracy Survive the Disruptive Power of AI?"
# (2024), EEAS 3rd FIMI Threat Report (2025), Meta CIB reporting archive
# (2022-2024), DISARM policy and platform-exploitation techniques.
ECONOMIC_REGULATORY_PRESSURE = {
    "Economic_Pressure_Weaponisation": {
        "Advertiser_And_Brand_Safety_Pressure": L(
            "Advertiser_Boycott_Triggering",
            "Brand_Safety_Risk_Signalling",
            "Programmatic_Ad_Blocklist_Manipulation",
            "Sponsor_Association_Pressure",
            "Revenue_Withdrawal_Campaigning",
        ),
        "Monetisation_Channel_Pressure": L(
            "Creator_Demonetisation_Pressure",
            "Payment_Processor_Complaint_Pressure",
            "Donation_Platform_Deplatforming_Pressure",
            "Affiliate_Revenue_Disruption",
            "Subscription_Churn_Mobilisation",
        ),
        "Labour_And_Market_Pressure": L(
            "Employer_Complaint_Mobilisation",
            "Professional_License_Complaint_Mobilisation",
            "Vendor_Contract_Cancellation_Pressure",
            "Procurement_Reputation_Pressure",
        ),
    },
    "Regulatory_Arbitrage_And_Process_Weaponisation": {
        "Jurisdictional_Gap_Exploitation": L(
            "Cross_Border_Complaint_Routing",
            "Platform_Law_Mismatch_Exploitation",
            "Data_Protection_Forum_Shopping",
            "Election_Law_Timing_Arbitrage",
            "Language_Jurisdiction_Arbitrage",
        ),
        "Complaint_Process_Weaponisation": L(
            "Coordinated_Regulatory_Complaint_Filing",
            "Automated_Takedown_Request_Batching",
            "Procedural_Appeal_Queue_Flooding",
            "Transparency_Report_Gaming",
            "Public_Record_Request_Burdening",
        ),
        "Policy_Exception_Exploitation": L(
            "Newsworthiness_Exception_Abuse",
            "Satire_Exception_Cover",
            "Public_Interest_Exception_Abuse",
            "Political_Ad_Disclaimer_Gap_Exploitation",
        ),
    },
    "Information_Integrity_Supply_Chain_Pressure": {
        "Fact_Checking_Infrastructure_Targeting": L(
            "Fact_Check_Database_Pollution",
            "Fact_Checker_Source_Harvesting",
            "Correction_Queue_Overload",
            "Claim_Rating_Context_Manipulation",
        ),
        "Provenance_And_Trust_Tooling_Targeting": L(
            "C2pa_Attestation_Confusion",
            "Trust_Indicator_Plugin_Spoofing",
            "Media_Rating_Appeal_Pressure",
            "Browser_Extension_Warning_Evasion",
        ),
        "Civil_Society_Funding_Pressure": L(
            "Donor_Complaint_Campaigning",
            "Grant_Compliance_Harassment",
            "Audit_Burden_Induction",
            "Partner_Reputation_Risk_Signalling",
        ),
    },
}


# ---------------------------------------------------------------------------
# 17. INSIDER THREAT FACILITATION AND HUMAN RECRUITMENT
# ---------------------------------------------------------------------------
# SOURCE: OpenAI "Disrupting malicious uses of AI" (June 2025), Google GTIG
# "Adversarial Misuse of Generative AI" (2025), DISARM T0091/T0093 recruit
# techniques, NCSC AI cyber-threat assessment (2024).
INSIDER_THREAT_RECRUITMENT = {
    "Human_Source_Recruitment": {
        "Insider_Prospect_Identification": L(
            "Aggrieved_Employee_Discovery",
            "Contractor_Access_Discovery",
            "Volunteer_Access_Discovery",
            "Moderator_Access_Discovery",
            "Election_Worker_Access_Discovery",
        ),
        "Ideological_Recruitment": L(
            "Cause_Alignment_Cultivation",
            "Grievance_Validation_Recruitment",
            "Status_Restoration_Recruitment",
            "Ingroup_Duty_Recruitment",
        ),
        "Financial_And_Career_Recruitment": L(
            "Paid_Tipster_Recruitment",
            "Consulting_Cover_Recruitment",
            "Job_Offer_Lure_Recruitment",
            "Debt_Relief_Recruitment",
        ),
        "Coercive_Recruitment": L(
            "Compromising_Material_Leverage",
            "Family_Safety_Leverage",
            "Immigration_Status_Leverage",
            "Legal_Process_Leverage",
        ),
    },
    "Insider_Enablement_And_Tasking": {
        "Access_Abuse_Tasking": L(
            "Document_Leak_Tasking",
            "Meeting_Record_Tasking",
            "Moderation_Action_Tasking",
            "Internal_Channel_Screenshot_Tasking",
            "Vote_Process_Observation_Tasking",
        ),
        "Influence_From_Within": L(
            "Internal_Rumor_Seeding",
            "Meeting_Agenda_Pressure",
            "Staff_Chat_Narrative_Seeding",
            "Internal_Petition_Seeding",
        ),
        "Verification_By_Insider": L(
            "Anonymous_Internal_Confirmation",
            "Credentialed_Backchannel_Confirmation",
            "Internal_Process_Attestation",
            "Controlled_Leak_Validation",
        ),
    },
    "Insider_Lifecycle_Management": {
        "Relationship_Maintenance": L(
            "Periodic_Check_In_Cultivation",
            "Secure_Channel_Migration",
            "Benefit_Delivery_Maintenance",
            "Risk_Reassurance_Maintenance",
        ),
        "Exit_And_Burn_Management": L(
            "Disposable_Insider_Retirement",
            "Whistleblower_Public_Pivot",
            "Attribution_Shielding_For_Insider",
            "Replacement_Source_Seeding",
        ),
        "Contractor_And_Remote_Worker_Schemes": L(
            "Laptop_Farm_Facilitation",
            "Remote_Access_Worker_Placement",
            "Video_Interview_Proxying",
            "Payroll_Identity_Cover",
        ),
    },
}


# ---------------------------------------------------------------------------
# 18. GAMIFICATION AND UNWITTING PARTICIPANT MOBILISATION
# ---------------------------------------------------------------------------
# SOURCE: DISARM T0016/T0020/T0103/T0104/T0105 delivery and engagement
# techniques, OpenAI June 2025 TikTok/comment-network case studies, DFRLab
# 2024 Foreign Interference Attribution Tracker.
GAMIFICATION_PARTICIPANT_MOBILISATION = {
    "Gamified_Recruitment_And_Tasking": {
        "Challenge_Based_Mobilisation": L(
            "Hashtag_Challenge_Tasking",
            "Duet_Response_Challenge_Tasking",
            "Meme_Template_Challenge_Tasking",
            "Screenshot_Proof_Challenge_Tasking",
            "Leaderboard_Action_Challenge",
        ),
        "Quest_And_Mission_Structures": L(
            "Daily_Mission_Content_Sharing",
            "Clue_Hunt_Narrative_Drop",
            "Tiered_Role_Unlocking",
            "Badge_Earning_Action_Loops",
            "Streak_Based_Participation",
        ),
        "Alternate_Reality_Game_Patterns": L(
            "Puzzle_Trail_Narrative_Seeding",
            "Hidden_Message_Decoding",
            "Insider_Drop_Interpretation_Game",
            "Symbol_Hunt_Mobilisation",
            "Countdown_Unlock_Event",
        ),
    },
    "Unwitting_Participant_Mobilisation": {
        "Civic_Game_Shells": L(
            "Polling_Info_Quiz_Shell",
            "Policy_Match_Game_Shell",
            "Candidate_Ranking_Game_Shell",
            "Local_Issue_Simulator_Shell",
        ),
        "Social_Competition_Shells": L(
            "Share_To_Unlock_Competition",
            "Invite_Friends_Boost_Loop",
            "Team_Versus_Team_Political_Game",
            "Creator_Challenge_Tournament",
        ),
        "Microtask_Influence_Labour": L(
            "Comment_Raid_Microtasking",
            "Report_Button_Microtasking",
            "Translation_Caption_Microtasking",
            "Clip_Curation_Microtasking",
        ),
    },
    "Interactive_Media_Narrative_Embedding": {
        "Game_And_Mod_Content": L(
            "Political_Mod_Narrative_Insertion",
            "Interactive_Fiction_Persuasion_Path",
            "Visual_Novel_Issue_Framing",
            "Simulator_Rule_Bias_Embedding",
        ),
        "Avatar_And_Live_Event_Participation": L(
            "Virtual_Rally_Game_Event",
            "Avatar_Sign_Carrying_Event",
            "Live_Stream_Decision_Polling",
            "Multiplayer_Roleplay_Narrative_Seeding",
        ),
        "Reward_Loop_Optimisation": L(
            "Loot_Box_Style_Disclosure_Drops",
            "Progress_Bar_Action_Prompts",
            "Social_Status_Badge_Rewards",
            "Scarcity_Timer_Action_Prompts",
        ),
    },
}


# ---------------------------------------------------------------------------
# Secondary axes (cross-cutting attributes)
# ---------------------------------------------------------------------------
SECONDARY_AXES = {
    "Effect_Sought": {
        "Agenda_And_Salience_Shaping": L(
            "Issue_Omnipresence_Inflation",
            "Crisis_Urgency_Inflation",
            "Topic_Suppression",
            "Window_Of_Discussion_Shifting",
            "Public_Memory_Shaping",
        ),
        "Attitude_And_Belief_Shifting": L(
            "Policy_Attitude_Shifting",
            "Actor_Trust_Shifting",
            "Institutional_Trust_Shifting",
            "Causal_Belief_Shifting",
            "Identity_Salience_Shifting",
        ),
        "Mobilisation_And_Suppression": L(
            "Vote_Mobilisation",
            "Vote_Suppression",
            "Protest_Mobilisation",
            "Protest_Suppression",
            "Donation_Mobilisation",
            "Volunteer_Mobilisation",
        ),
        "Polarisation_And_Radicalisation": L(
            "Issue_Polarisation",
            "Affective_Polarisation",
            "Ideological_Radicalisation",
            "Movement_Edge_Recruitment",
        ),
        "Elite_Behaviour_Shaping": L(
            "Politician_Statement_Shaping",
            "Media_Coverage_Shaping",
            "Regulator_Action_Shaping",
            "Donor_Behaviour_Shaping",
        ),
        "Epistemic_Demobilisation": L(
            "Verification_Demobilisation",
            "News_Consumption_Demobilisation",
            "Civic_Engagement_Demobilisation",
        ),
    },
    "Target": {
        "Individual_Target": L(
            "Voter_Target",
            "Politician_Target",
            "Journalist_Target",
            "Activist_Target",
            "Public_Servant_Target",
        ),
        "Sub_Population_Target": L(
            "Demographic_Cohort_Target",
            "Issue_Public_Target",
            "Identity_Group_Target",
            "Geographic_Region_Target",
            "Diaspora_Target",
        ),
        "Institution_Target": L(
            "Election_Authority_Target",
            "Court_System_Target",
            "Regulatory_Agency_Target",
            "Media_Outlet_Target",
            "Civil_Society_Organisation_Target",
        ),
        "Platform_Target": L(
            "Social_Media_Platform_Target",
            "Search_Engine_Target",
            "Ai_Assistant_Target",
            "Messaging_App_Target",
            "Forum_Or_Imageboard_Target",
        ),
        "Coalition_Target": L(
            "Alliance_Target",
            "Party_Coalition_Target",
            "International_Forum_Target",
            "Movement_Coalition_Target",
        ),
    },
    "Campaign_Phase": {
        "Preparation_Phase": L(
            "Reconnaissance_Phase",
            "Audience_Mapping_Phase",
            "Asset_Development_Phase",
            "Infrastructure_Setup_Phase",
        ),
        "Seeding_Phase": L(
            "Initial_Persona_Seeding_Phase",
            "Initial_Narrative_Seeding_Phase",
            "Influencer_Seeding_Phase",
        ),
        "Amplification_Phase": L(
            "Bot_Amplification_Phase",
            "Influencer_Amplification_Phase",
            "Cross_Platform_Amplification_Phase",
        ),
        "Exploitation_Phase": L(
            "Crisis_Exploitation_Phase",
            "Election_Window_Exploitation_Phase",
            "Trigger_Event_Exploitation_Phase",
        ),
        "Sustainment_Phase": L(
            "Repetition_Sustainment_Phase",
            "Adaptation_Sustainment_Phase",
            "Audience_Retention_Sustainment_Phase",
        ),
        "Reconstitution_Phase": L(
            "Identity_Reset_Phase",
            "Audience_Migration_Phase",
            "Narrative_Mutation_Phase",
        ),
    },
    "Modality_Organization": {
        "State_Apparatus_Modality": L(
            "Intelligence_Service_Operation",
            "Defence_Or_Military_Information_Operation",
            "Diplomatic_Information_Operation",
            "State_Media_Operation",
        ),
        "State_Aligned_Proxy_Modality": L(
            "State_Sponsored_Contractor_Operation",
            "State_Adjacent_Hacktivist_Operation",
            "State_Funded_Cutout_Ngo_Operation",
        ),
        "Domestic_Political_Modality": L(
            "Party_Operation",
            "Campaign_Operation",
            "Movement_Operation",
            "Lobbying_Operation",
        ),
        "Commercial_And_Pr_Modality": L(
            "Pr_Firm_Operation",
            "Marketing_Agency_Operation",
            "Disinformation_For_Hire_Operation",
        ),
        "Criminal_Modality": L(
            "Cybercriminal_Operation",
            "Extortion_Modality_Operation",
            "Influence_For_Hire_Operation",
        ),
        "Ai_Driven_Autonomous_Modality": L(
            "Llm_Persona_Network_Autonomous_Operation",
            "Reinforcement_Learning_Targeting_Operation",
            "Closed_Loop_Multi_Agent_Operation",
        ),
        "Fellow_Traveler_Network_Modality": L(
            "Aligned_Media_Operation",
            "Aligned_Subculture_Operation",
            "Aligned_Movement_Operation",
        ),
    },
    "Doctrine_Reference_Mapping": {
        "Nato_Cognitive_Warfare_Doctrine": L(
            "Cognitive_Battlespace_Operation",
            "Cognitive_Manoeuvre_Operation",
            "Cognitive_Resilience_Disruption_Operation",
        ),
        "Disarm_Framework_Mapping": L(
            "Plan_Strategy_Disarm_Mapping",
            "Plan_Objectives_Disarm_Mapping",
            "Target_Audience_Analysis_Disarm_Mapping",
            "Develop_Narratives_Disarm_Mapping",
            "Develop_Content_Disarm_Mapping",
            "Establish_Assets_Disarm_Mapping",
            "Establish_Legitimacy_Disarm_Mapping",
            "Deliver_Content_Disarm_Mapping",
            "Microtarget_Disarm_Mapping",
            "Select_Channels_Disarm_Mapping",
            "Conduct_Operations_Disarm_Mapping",
            "Maximise_Exposure_Disarm_Mapping",
            "Conduct_Fundraising_Disarm_Mapping",
            "Recruit_Disarm_Mapping",
            "Drive_Online_To_Offline_Disarm_Mapping",
            "Persist_In_Information_Environment_Disarm_Mapping",
            "Assess_Effectiveness_Disarm_Mapping",
        ),
        "Mitre_Atlas_Mapping": L(
            "Atlas_Prompt_Injection_Mapping",
            "Atlas_Rag_Poisoning_Mapping",
            "Atlas_Data_Poisoning_Mapping",
            "Atlas_Model_Evasion_Mapping",
            "Atlas_Identity_Spoofing_Mapping",
        ),
        "Eu_Fimi_Framework_Mapping": L(
            "Foreign_Information_Manipulation_Operation",
            "Information_Interference_Operation",
            "Fimi_Infrastructure_Exposure_Mapping",
            "Tactic_Technique_Procedure_Mapping",
        ),
        "Mitre_Att_Ck_For_Influence_Mapping": L(
            "Reconnaissance_Att_Ck_Mapping",
            "Resource_Development_Att_Ck_Mapping",
            "Initial_Access_Att_Ck_Mapping",
            "Establish_Legitimacy_Att_Ck_Mapping",
            "Develop_Content_Att_Ck_Mapping",
            "Microtarget_Att_Ck_Mapping",
            "Deliver_Content_Att_Ck_Mapping",
            "Maximise_Exposure_Att_Ck_Mapping",
            "Drive_Online_To_Offline_Att_Ck_Mapping",
            "Persist_Att_Ck_Mapping",
            "Assess_Effectiveness_Att_Ck_Mapping",
        ),
        "Hybrid_Threat_Toolkit_Mapping": L(
            "Information_Domain_Hybrid_Operation",
            "Cyber_Domain_Hybrid_Operation",
            "Economic_Domain_Hybrid_Operation",
            "Political_Domain_Hybrid_Operation",
            "Social_Domain_Hybrid_Operation",
        ),
    },
    "Complexity_Tier": L(
        "T1_Atomic_Single_Artifact",
        "T2_Campaign_Coordinated_Multi_Artifact",
        "T3_Synthetic_Generative_Ai_Artifact",
        "T4_Orchestrated_Multi_Agent_Operation",
        "T5_Sustained_State_Level_Operation",
    ),
    "Temporal_Horizon": L(
        "Instant_Horizon",
        "Hours_Horizon",
        "Days_Horizon",
        "Weeks_Horizon",
        "Months_Horizon",
        "Years_Horizon",
        "Election_Cycle_Horizon",
        "Multi_Year_Prepositioning_Horizon",
        "Real_Time_Adaptive_Horizon",
    ),
    "Epistemic_Target": L(
        "Factual_Belief_Target",
        "Evaluative_Attitude_Target",
        "Identity_Anchor_Target",
        "Trust_Dimension_Target",
        "Epistemic_Routine_Target",
        "Behavioural_Intention_Target",
        "Semantic_Meaning_Target",
        "Procedural_Memory_Target",
        "Information_Access_Target",
    ),
    "Detection_Evasion_Modality": L(
        "Textual_Mutation_Evasion_Modality",
        "Visual_Perceptual_Evasion_Modality",
        "Audio_Provenance_Evasion_Modality",
        "Cross_Platform_Threshold_Evasion_Modality",
        "Jurisdictional_Policy_Evasion_Modality",
        "Human_Automation_Blending_Evasion_Modality",
        "Provenance_Attestation_Evasion_Modality",
    ),
}


# ---------------------------------------------------------------------------
# Top-level metadata + compatibility-rule metanode
# ---------------------------------------------------------------------------

METADATA_BLOCK: Dict[str, Any] = {
    "schema_version": "v4-test-run-1-production-rev1",
    "ontology_role": "deployment",
    "deployment_compatible": True,
    "title": "Cyber-manipulation of Political Opinions — Deployment Attack Ontology",
    "subtitle": "Full hierarchical state space of adversarial techniques targeting political-opinion formation",
    "design_principles": [
        "Subtree keys are PascalCase_With_Underscores; leaves default to {} so flatten_leaf_paths enumerates every technique.",
        "Per-leaf metadata is restricted to UNIQUE structural sampling constraints not derivable from the leaf's parent path.",
        "Subtree-wide sampling logic is encoded ONCE in the top-level _compatibility_rules metanode using path-glob patterns.",
        "The ontology does NOT encode psychological-amplification hypotheses (e.g. 'Neuroticism amplifies fear-appeal X'). Those are exactly what the inferential layer estimates.",
        "Crosswalks to NATO cognitive-warfare doctrine, the DISARM framework, EU FIMI, MITRE ATT&CK for influence, MITRE ATLAS, and Hybrid Threat toolkits live under Secondary_Axes > Doctrine_Reference_Mapping.",
        "Gap-filled via systematic literature review (2024-2025) covering DISARM, TA09/ATT&CK-style influence mappings, NATO StratCom/ACT, EEAS FIMI, UK NCSC, Stanford IO, DFRLab, OpenAI, Google GTIG, Microsoft MTAC, Meta, NewsGuard, RAND, and Carnegie. Compatibility rules optimised for cross-ontology coherence with OPINION and PROFILE ontologies.",
    ],
    "primary_families": [
        "Intelligence_Preparation_And_Vulnerability_Analysis",
        "Source_Identity_And_Legitimacy_Manipulation",
        "Claim_Frame_And_Narrative_Manipulation",
        "Ai_Generated_Synthetic_Media_And_Content",
        "Multi_Agent_Adversarial_Architecture",
        "Targeting_Delivery_And_Discoverability_Optimization",
        "Amplification_Visibility_And_Attention_Manipulation",
        "Social_Proof_Network_And_Community_Manipulation",
        "Platform_Information_Environment_And_Media_System_Shaping",
        "Cyber_Enabled_Compromise_Coercion_And_Disruption",
        "Behavioral_Conversion_Mobilization_Suppression_And_Radicalization",
        "Cognitive_Infrastructure_And_Epistemic_Subversion",
        "Operational_Security_Evasion_Persistence_And_Reconstitution",
        "Measurement_Experimentation_And_Adaptive_Learning",
        "Narrative_Infrastructure_And_Ecosystem_Capture",
        "Economic_And_Regulatory_Pressure_Weaponisation",
        "Insider_Threat_Facilitation_And_Human_Recruitment",
        "Gamification_And_Unwitting_Participant_Mobilisation",
    ],
    "secondary_axes": [
        "Effect_Sought",
        "Target",
        "Campaign_Phase",
        "Modality_Organization",
        "Doctrine_Reference_Mapping",
        "Complexity_Tier",
        "Temporal_Horizon",
        "Epistemic_Target",
        "Detection_Evasion_Modality",
    ],
}


# Path-pattern compatibility rules. Patterns use a single `**` wildcard that
# matches any number of intermediate path components. The compatibility engine
# evaluates rules in order; later rules override earlier ones for the same
# attribute on the same leaf.
COMPATIBILITY_RULES_BLOCK: Dict[str, Any] = {
    "schema_version": "v4-test-run-1-meta",
    "rule_evaluation": "ordered_merge_last_wins_per_attribute",
    "pattern_wildcard": "**",
    "pattern_separator": " > ",
    "_documentation": (
        "Path-pattern rules apply attributes to whole subtrees of the ATTACK ontology. "
        "An attack leaf path matches a pattern when every literal segment in the pattern "
        "appears in the same order in the leaf path; segments separated by `**` are wildcards "
        "that match any number of intervening segments. Rules are evaluated in order; for any "
        "given attribute (e.g. `compatible_opinion_domains`), the LAST matching rule wins. "
        "Per-leaf inline metadata, when present, takes precedence over rule-derived attributes."
    ),
    "rules": [
        # ── Capability prerequisites ────────────────────────────────────────
        {
            "rule_id": "multi_agent_requires_orchestration",
            "applies_to_attack_paths": [
                "**Multi_Agent_Adversarial_Architecture**",
                "**Persona_Cluster_Orchestration**",
                "**Adversarial_Conversation_Loops**",
                "**Cross_Channel_Synchronization**",
                "**Auto_Targeting_Pipelines**",
                "**Hybrid_Human_Ai_Operations**",
                "**Reflexive_Argumentation_Refinement**",
            ],
            "requires_capability": ["agent_orchestration"],
            "rationale": "These operations are realised through multi-turn or multi-agent infrastructure.",
        },
        {
            "rule_id": "microtargeting_requires_personalisation",
            "applies_to_attack_paths": [
                "**Audience_Matching_And_Microtargeting**",
                "**Profile_Inference_Targeting**",
                "**Cohort_Adapted_Messaging**",
                "**Reinforcement_Learning_Targeting**",
                "**Behavioural_Retargeting**",
                "**Lookalike_Audience_Construction**",
            ],
            "requires_capability": ["profile_personalisation"],
            "rationale": "Microtargeting is only meaningful when at least partial profile signal is observable.",
        },
        {
            "rule_id": "intelligence_preparation_requires_data_access",
            "applies_to_attack_paths": [
                "**Intelligence_Preparation_And_Vulnerability_Analysis**",
                "**Data_Acquisition_And_Feature_Enrichment**",
            ],
            "requires_capability": ["data_acquisition"],
            "rationale": "Preparation-phase tactics presuppose access to OSINT or covert data sources.",
        },
        {
            "rule_id": "synthetic_media_requires_generative_ai",
            "applies_to_attack_paths": [
                "**Ai_Generated_Synthetic_Media_And_Content**",
                "**Foundation_Model_Text_Generation**",
                "**Foundation_Model_Image_Generation**",
                "**Foundation_Model_Audio_Generation**",
                "**Foundation_Model_Video_Generation**",
                "**Llm_Long_Form_Article_Generation**",
                "**Voice_Cloning_Generation**",
                "**Lip_Sync_Deepfake_Generation**",
                "**Full_Body_Deepfake_Generation**",
                "**Text_To_Video_Generation**",
                "**Image_To_Video_Generation**",
                "**Multimodal_Bundle_Generation**",
            ],
            "requires_capability": ["generative_ai_models"],
            "rationale": "All foundation-model-driven content production presupposes generative AI access.",
        },
        {
            "rule_id": "ai_retrieval_surface_requires_search_target",
            "applies_to_attack_paths": [
                "**Ai_Retrieval_Surface_Poisoning**",
                "**Ai_Answer_Poisoning**",
                "**Search_Pollution**",
                "**Search_Engine_Optimisation**",
                "**Vector_Embedding_Adversarial_Drift**",
                "**Voice_Assistant_Answer_Poisoning**",
            ],
            "requires_capability": ["search_index_targetable"],
            "rationale": "Retrieval-surface manipulation only matters for opinions where target users actively query.",
        },
        {
            "rule_id": "cyber_compromise_requires_exploit_capability",
            "applies_to_attack_paths": [
                "**Cyber_Enabled_Compromise_Coercion_And_Disruption**",
                "**Compromise_And_Expose**",
                "**Trusted_Channel_Hijacking**",
                "**Disruption_For_Interpretive_Effect**",
            ],
            "requires_capability": ["intrusion_or_disruption"],
            "rationale": "Compromise-and-expose operations presuppose access to exploit or disruption tooling.",
        },
        {
            "rule_id": "economic_regulatory_pressure_requires_process_access",
            "applies_to_attack_paths": [
                "**Economic_And_Regulatory_Pressure_Weaponisation**",
                "**Complaint_Process_Weaponisation**",
                "**Information_Integrity_Supply_Chain_Pressure**",
            ],
            "requires_capability": ["legal_process_or_platform_policy_access"],
            "rationale": "These tactics operate through formal complaint, monetisation, trust-safety, or policy channels.",
        },
        {
            "rule_id": "insider_facilitation_requires_human_source_access",
            "applies_to_attack_paths": [
                "**Insider_Threat_Facilitation_And_Human_Recruitment**",
                "**Human_Source_Recruitment**",
                "**Insider_Enablement_And_Tasking**",
            ],
            "requires_capability": ["human_source_recruitment_access"],
            "rationale": "Insider facilitation mechanically requires a recruitable or taskable human source.",
        },
        {
            "rule_id": "interactive_gamification_requires_interactive_surface",
            "applies_to_attack_paths": [
                "**Gamification_And_Unwitting_Participant_Mobilisation**",
                "**Interactive_Media_Narrative_Embedding**",
                "**Quest_And_Mission_Structures**",
            ],
            "requires_capability": ["interactive_media_or_game_surface"],
            "rationale": "Gamified tasking requires an interactive social, game, or live-event surface.",
        },

        # ── Domain compatibility per attack subtree ────────────────────────
        {
            "rule_id": "ai_retrieval_surface_compatible_domains",
            "applies_to_attack_paths": [
                "**Ai_Retrieval_Surface_Poisoning**",
                "**Ai_Answer_Poisoning**",
                "**Vector_Embedding_Adversarial_Drift**",
                "**Voice_Assistant_Answer_Poisoning**",
            ],
            "compatible_opinion_domains": [
                "Information_Integrity_And_Platforms",
                "Democratic_Resilience_And_Institutions",
                "Defense_And_National_Security",
                "Foreign_Policy_And_Geopolitics",
                "Technology_And_Ai_Governance",
                "Critical_Infrastructure_And_Energy_Sovereignty",
                "Macroeconomic_And_Fiscal_Policy",
                "Healthcare_And_Public_Health",
                "Information_Behaviour_And_Media_Diet",
                "Conspiracy_Misinformation_And_Epistemic_Orientations",
            ],
            "rationale": "AI retrieval-surface attacks are strongest on issues where users actively search for guidance.",
        },
        {
            "rule_id": "deepfake_video_audio_compatible_domains",
            "applies_to_attack_paths": [
                "**Foundation_Model_Video_Generation**",
                "**Foundation_Model_Audio_Generation**",
                "**Lip_Sync_Deepfake_Generation**",
                "**Full_Body_Deepfake_Generation**",
                "**Voice_Cloning_Generation**",
                "**Synthetic_Speech_Generation**",
            ],
            "compatible_opinion_domains": [
                "Defense_And_National_Security",
                "Foreign_Policy_And_Geopolitics",
                "Information_Integrity_And_Platforms",
                "Democratic_Resilience_And_Institutions",
                "Civil_Liberties_And_Surveillance",
                "Immigration_And_Citizenship",
                "Macroeconomic_And_Fiscal_Policy",
                "Healthcare_And_Public_Health",
                "Law_Safety_And_Justice",
                "Threat_Perceptions_And_Existential_Anxieties",
            ],
            "rationale": "Deepfake video / audio attacks gain power from named officials and high-stakes domains.",
        },
        {
            "rule_id": "election_specific_attacks_constrained_to_governance",
            "applies_to_attack_paths": [
                "**Election_Window_Targeting**",
                "**Counting_And_Certification_Window_Targeting**",
                "**Registration_And_Procedural_Discouragement**",
                "**Queue_Confusion_And_Timing_Effects**",
            ],
            "compatible_opinion_domains": [
                "Democratic_Resilience_And_Institutions",
                "Information_Integrity_And_Platforms",
                "Civil_Liberties_And_Surveillance",
                "Democratic_Norms_And_Regime_Orientation",
                "Civic_Participation_Intentions",
            ],
            "rationale": "Election-window tactics presuppose an electoral context.",
        },
        {
            "rule_id": "diaspora_specific_attacks_compatibility",
            "applies_to_attack_paths": [
                "**Diaspora_Proxy_Voices**",
                "**Diaspora_Language_Chaining**",
                "**Diaspora_Bridge_Nodes**",
                "**Diaspora_And_Identity_Space_Penetration**",
            ],
            "compatible_opinion_domains": [
                "Foreign_Policy_And_Geopolitics",
                "Defense_And_National_Security",
                "Immigration_And_Citizenship",
                "Democratic_Resilience_And_Institutions",
            ],
            "rationale": "Diaspora-anchored vectors are most coherent on identity / sovereignty / migration domains.",
        },
        {
            "rule_id": "regulatory_arbitrage_structural_domains",
            "applies_to_attack_paths": [
                "**Regulatory_Arbitrage_And_Process_Weaponisation**",
                "**Complaint_Process_Weaponisation**",
                "**Policy_Exception_Exploitation**",
            ],
            "compatible_opinion_domains": [
                "Technology_And_Ai_Governance",
                "Information_Integrity_And_Platforms",
                "Democratic_Resilience_And_Institutions",
                "Civil_Liberties_And_Surveillance",
                "Law_Safety_And_Justice",
                "Institutional_Trust_And_System_Legitimacy",
            ],
            "rationale": "Regulatory-arbitrage operations presuppose governance, rights, platform-policy, or institutional-process objects.",
        },
        {
            "rule_id": "information_integrity_supply_chain_domains",
            "applies_to_attack_paths": [
                "**Information_Integrity_Supply_Chain_Pressure**",
                "**Fact_Checking_Infrastructure_Targeting**",
                "**Provenance_And_Trust_Tooling_Targeting**",
            ],
            "compatible_opinion_domains": [
                "Information_Integrity_And_Platforms",
                "Technology_And_Ai_Governance",
                "Information_Behaviour_And_Media_Diet",
                "Conspiracy_Misinformation_And_Epistemic_Orientations",
                "Institutional_Trust_And_System_Legitimacy",
                "Democratic_Norms_And_Regime_Orientation",
                "Democratic_Resilience_And_Institutions",
            ],
            "rationale": "These techniques presuppose information-integrity infrastructure, provenance systems, or source-trust objects.",
        },

        # ── Complexity tiering for whole subtrees ──────────────────────────
        {
            "rule_id": "atomic_tier_for_simple_post_techniques",
            "applies_to_attack_paths": [
                "**Headline_And_Lede_Misframing**",
                "**Quote_Context_Stripping**",
                "**Personal_Safety_Fear**",
                "**Outrage_To_Cynicism_Mutation**",
                "**Single_Case_Horror_Story**",
                "**Symbolic_Counter_Action_Mobilization**",
            ],
            "complexity_tier": "T1_atomic",
        },
        {
            "rule_id": "campaign_tier_for_amplification_subtree",
            "applies_to_attack_paths": [
                "**Amplification_Visibility_And_Attention_Manipulation**",
                "**Social_Proof_Network_And_Community_Manipulation**",
                "**Platform_Information_Environment_And_Media_System_Shaping**",
            ],
            "complexity_tier": "T2_campaign",
        },
        {
            "rule_id": "synthetic_tier_for_ai_generated_subtree",
            "applies_to_attack_paths": [
                "**Ai_Generated_Synthetic_Media_And_Content**",
            ],
            "complexity_tier": "T3_synthetic",
        },
        {
            "rule_id": "orchestrated_tier_for_multi_agent",
            "applies_to_attack_paths": [
                "**Multi_Agent_Adversarial_Architecture**",
                "**Hybrid_Human_Ai_Operations**",
                "**Reinforcement_Learning_Targeting**",
            ],
            "complexity_tier": "T4_orchestrated",
        },
        {
            "rule_id": "sustained_tier_for_long_horizon_ops",
            "applies_to_attack_paths": [
                "**Operational_Security_Evasion_Persistence_And_Reconstitution**",
                "**Measurement_Experimentation_And_Adaptive_Learning**",
                "**Long_Horizon_Trust_Building**",
                "**Long_Term_Relationship_Building**",
                "**State_Aligned_Proxy_Modality**",
                "**Narrative_Infrastructure_And_Ecosystem_Capture**",
                "**Economic_And_Regulatory_Pressure_Weaponisation**",
                "**Insider_Threat_Facilitation_And_Human_Recruitment**",
            ],
            "complexity_tier": "T5_sustained",
        },
        {
            "rule_id": "orchestrated_tier_for_gamified_mobilisation",
            "applies_to_attack_paths": [
                "**Gamification_And_Unwitting_Participant_Mobilisation**",
                "**Interactive_Media_Narrative_Embedding**",
                "**Gamified_Recruitment_And_Tasking**",
            ],
            "complexity_tier": "T4_orchestrated",
        },
        {
            "rule_id": "atomic_or_campaign_tier_for_claim_level_deception",
            "applies_to_attack_paths": [
                "**Claim_Frame_And_Narrative_Manipulation > Claim_Level_Deception**",
            ],
            "complexity_tier": "T1_atomic",
            "rationale": "Claim-level deception is a single-artifact mechanism unless combined with amplification or orchestration elsewhere.",
        },
        {
            "rule_id": "systemic_vulnerability_mapping_minimum_tier",
            "applies_to_attack_paths": [
                "**Systemic_Vulnerability_Mapping**",
                "**Narrative_Prepositioning_And_Dormant_Seeding**",
                "**Ecosystem_Capture_And_Authority_Substitution**",
            ],
            "complexity_tier": "T5_sustained",
            "temporal_horizon": "years",
            "rationale": "Systemic mapping and ecosystem capture require sustained cross-source observation and preparation.",
        },

        # ── Epistemic target tagging ────────────────────────────────────────
        {
            "rule_id": "factual_belief_target_for_evidence_subtree",
            "applies_to_attack_paths": [
                "**Claim_Level_Deception**",
                "**Fabricated_Claim_Injection**",
                "**Forensic_Trace_Planting**",
                "**Foundation_Model_Image_Generation**",
                "**Foundation_Model_Video_Generation**",
            ],
            "epistemic_target": "factual_belief",
        },
        {
            "rule_id": "evaluative_attitude_target_for_framing_subtree",
            "applies_to_attack_paths": [
                "**Frame_And_Reframing_Operations**",
                "**Master_Narrative_Engineering**",
                "**Issue_Position_Shift**",
                "**Polarization_And_Radicalization**",
            ],
            "epistemic_target": "evaluative_attitude",
        },
        {
            "rule_id": "identity_anchor_target_for_identity_subtree",
            "applies_to_attack_paths": [
                "**Identity_And_Group_Coding**",
                "**Identity_Hardening**",
                "**Status_Threat_Coding**",
                "**Outgroup_Essentialisation**",
            ],
            "epistemic_target": "identity_anchor",
        },
        {
            "rule_id": "trust_dimension_target_for_legitimacy_subtree",
            "applies_to_attack_paths": [
                "**Source_Identity_And_Legitimacy_Manipulation**",
                "**Impersonation_And_Mimicry**",
                "**Stolen_Credibility_Signals**",
                "**Trusted_Channel_Hijacking**",
            ],
            "epistemic_target": "trust_dimension",
        },
        {
            "rule_id": "epistemic_routine_target_for_subversion_subtree",
            "applies_to_attack_paths": [
                "**Epistemic_Destabilisation**",
                "**Cognitive_Infrastructure_And_Epistemic_Subversion**",
                "**Verification_Fatigue_Induction**",
                "**Evidence_Reweighting**",
                "**Source_Relativisation**",
            ],
            "epistemic_target": "epistemic_routine",
        },
        {
            "rule_id": "semantic_meaning_target_for_narrative_infrastructure",
            "applies_to_attack_paths": [
                "**Narrative_Infrastructure_And_Ecosystem_Capture**",
                "**Semantic_Term_Shift**",
                "**Overton_Window_Preconditioning**",
            ],
            "epistemic_target": "semantic_meaning",
        },
        {
            "rule_id": "behavioural_intention_target_for_gamified_mobilisation",
            "applies_to_attack_paths": [
                "**Gamification_And_Unwitting_Participant_Mobilisation**",
                "**Unwitting_Participant_Mobilisation**",
                "**Microtask_Influence_Labour**",
            ],
            "epistemic_target": "behavioural_intention",
        },
        {
            "rule_id": "information_access_target_for_pressure_and_supply_chain",
            "applies_to_attack_paths": [
                "**Information_Integrity_Supply_Chain_Pressure**",
                "**Strategic_Silence_And_Counter_Narrative_Denial**",
                "**Economic_And_Regulatory_Pressure_Weaponisation**",
            ],
            "epistemic_target": "information_access",
        },

        # ── Co-deployment hints (interaction rules) ─────────────────────────
        {
            "rule_id": "deepfake_amplifies_when_paired_with_amplification",
            "applies_to_attack_paths": [
                "**Foundation_Model_Video_Generation**",
                "**Foundation_Model_Audio_Generation**",
                "**Lip_Sync_Deepfake_Generation**",
            ],
            "natural_companions": [
                "**Amplification_Visibility_And_Attention_Manipulation**",
                "**Cross_Platform_Narrative_Laundering**",
                "**Citation_Chain_Construction**",
            ],
            "rationale": "Synthetic-media artifacts only realise their full effect when paired with downstream amplification and laundering.",
        },
        {
            "rule_id": "intelligence_prep_precedes_targeting",
            "applies_to_attack_paths": [
                "**Audience_Matching_And_Microtargeting**",
                "**Profile_Inference_Targeting**",
            ],
            "natural_predecessors": [
                "**Intelligence_Preparation_And_Vulnerability_Analysis**",
                "**Data_Acquisition_And_Feature_Enrichment**",
            ],
            "rationale": "Microtargeting presupposes prior segmentation / profiling.",
        },
        {
            "rule_id": "operational_security_wraps_all_subtrees",
            "applies_to_attack_paths": [
                "**Operational_Security_Evasion_Persistence_And_Reconstitution**",
            ],
            "natural_companions": ["**"],
            "rationale": "Opsec / persistence / reconstitution applies as a cross-cutting layer to all primary families.",
        },
        {
            "rule_id": "synthetic_media_requires_payload_or_persona_predecessor",
            "applies_to_attack_paths": [
                "**Foundation_Model_Video_Generation**",
                "**Foundation_Model_Audio_Generation**",
                "**Foundation_Model_Image_Generation**",
                "**Multimodal_Bundle_Generation**",
            ],
            "natural_predecessors": [
                "**Persona_Fabrication**",
                "**Claim_Level_Deception**",
            ],
            "rationale": "Synthetic artifacts require either an attributed source identity or a claim payload to be operationally meaningful.",
        },
        {
            "rule_id": "multi_agent_operations_require_seed_predecessors",
            "applies_to_attack_paths": [
                "**Multi_Agent_Adversarial_Architecture**",
                "**Gamification_And_Unwitting_Participant_Mobilisation**",
            ],
            "natural_predecessors": [
                "**Persona_Fabrication**",
                "**Audience_Segmentation_And_Population_Mapping**",
                "**Claim_Level_Deception**",
            ],
            "rationale": "Orchestrated systems require seed identities, target segmentation, or narrative payloads before deployment.",
        },
        {
            "rule_id": "long_horizon_capture_requires_laundering_predecessors",
            "applies_to_attack_paths": [
                "**Narrative_Infrastructure_And_Ecosystem_Capture**",
                "**Economic_And_Regulatory_Pressure_Weaponisation**",
                "**Insider_Threat_Facilitation_And_Human_Recruitment**",
            ],
            "natural_predecessors": [
                "**Intelligence_Preparation_And_Vulnerability_Analysis**",
                "**Source_Identity_And_Legitimacy_Manipulation**",
                "**Narrative_Laundering_And_Translation**",
            ],
            "rationale": "Long-horizon ecosystem, pressure, and insider operations require prior mapping, identity infrastructure, or laundering paths.",
        },
        {
            "rule_id": "legacy_high_complexity_predecessors",
            "applies_to_attack_paths": [
                "**Systemic_Vulnerability_Mapping**",
                "**Long_Horizon_Trust_Building**",
                "**Operational_Security_Evasion_Persistence_And_Reconstitution**",
                "**Measurement_Experimentation_And_Adaptive_Learning**",
            ],
            "natural_predecessors": [
                "**Intelligence_Preparation_And_Vulnerability_Analysis**",
                "**Claim_Frame_And_Narrative_Manipulation**",
                "**Source_Identity_And_Legitimacy_Manipulation**",
            ],
            "rationale": "Legacy high-complexity branches require a mapped target, a payload, or identity infrastructure before they become operationally plausible.",
        },
        {
            "rule_id": "high_complexity_operations_pair_with_measurement",
            "applies_to_attack_paths": [
                "**Ai_Generated_Synthetic_Media_And_Content**",
                "**Multi_Agent_Adversarial_Architecture**",
                "**Operational_Security_Evasion_Persistence_And_Reconstitution**",
                "**Narrative_Infrastructure_And_Ecosystem_Capture**",
                "**Economic_And_Regulatory_Pressure_Weaponisation**",
                "**Insider_Threat_Facilitation_And_Human_Recruitment**",
                "**Gamification_And_Unwitting_Participant_Mobilisation**",
            ],
            "natural_companions": [
                "**Measurement_Experimentation_And_Adaptive_Learning**",
            ],
            "rationale": "High-complexity operations normally require measurement loops to coordinate, adapt, and decide whether to persist or reconstitute.",
        },
        {
            "rule_id": "synthetic_evidence_triad_companions",
            "applies_to_attack_paths": [
                "**Ai_Generated_Synthetic_Media_And_Content**",
                "**Multimodal_Bundle_Generation**",
            ],
            "natural_companions": [
                "**Source_Identity_And_Legitimacy_Manipulation**",
                "**Amplification_Visibility_And_Attention_Manipulation**",
                "**Narrative_Laundering_And_Translation**",
            ],
            "rationale": "Synthetic evidence needs source attribution, distribution, and laundering to function as a complete operation.",
        },
        {
            "rule_id": "strategic_silence_excludes_same_target_amplification",
            "applies_to_attack_paths": [
                "**Strategic_Silence_And_Counter_Narrative_Denial**",
            ],
            "mutually_exclusive_with_paths": [
                "**Amplification_Visibility_And_Attention_Manipulation**",
            ],
            "rationale": "A same-scenario silence operation suppresses visibility; direct amplification of the same target construct is the opposite mechanism.",
        },
        {
            "rule_id": "formal_complaint_pressure_excludes_detection_evasion",
            "applies_to_attack_paths": [
                "**Complaint_Process_Weaponisation**",
                "**Regulatory_Arbitrage_And_Process_Weaponisation**",
            ],
            "mutually_exclusive_with_paths": [
                "**Detection_Evasion**",
            ],
            "rationale": "Formal complaint pressure intentionally creates process visibility, while detection-evasion subtrees avoid such visibility.",
        },
        {
            "rule_id": "intelligence_preparation_scenario_role",
            "applies_to_attack_paths": [
                "**Intelligence_Preparation_And_Vulnerability_Analysis**",
            ],
            "scenario_role": "operator_preparation",
            "rationale": "These leaves model adversary preparation rather than direct target exposure.",
        },
        {
            "rule_id": "secondary_axes_are_classification_only",
            "applies_to_attack_paths": [
                "**Secondary_Axes**",
                "**Doctrine_Reference_Mapping**",
                "**Complexity_Tier**",
                "**Temporal_Horizon**",
                "**Epistemic_Target**",
                "**Detection_Evasion_Modality**",
            ],
            "scenario_role": "classification_axis",
            "is_classification_axis": True,
            "rationale": "Secondary axes are labels and crosswalks, not attack scenarios for exposure sampling.",
        },
    ],
    "capability_definitions": {
        "agent_orchestration": "Multi-turn / multi-agent runtime with persistent persona memory.",
        "profile_personalisation": "Operator can observe at least partial target-profile signal.",
        "data_acquisition": "Access to OSINT, leaked datasets, broker data, or covert collection capability.",
        "generative_ai_models": "Access to foundation models for text, image, audio, video, or multimodal generation.",
        "search_index_targetable": "Target population uses search / retrieval surfaces for the relevant issue.",
        "intrusion_or_disruption": "Capability to compromise systems or disrupt online services.",
        "legal_process_or_platform_policy_access": "Access to formal complaint, moderation, monetisation, regulatory, or policy-process channels.",
        "human_source_recruitment_access": "Ability to identify, approach, and securely task human sources or insiders.",
        "interactive_media_or_game_surface": "Access to an interactive social, live-event, game, or gamified tasking surface.",
    },
    "complexity_tier_definitions": {
        "T1_atomic": "Single artifact / single platform; producible in minutes by one operator.",
        "T2_campaign": "Multi-artifact campaign with light coordination across accounts or platforms.",
        "T3_synthetic": "Generative-AI artifact requiring foundation-model access and verification-evasion know-how.",
        "T4_orchestrated": "Multi-agent / persona-cluster operation with adaptive feedback loops.",
        "T5_sustained": "Long-horizon, state-level / advanced non-state operation with persistence and reconstitution.",
    },
    "epistemic_target_definitions": {
        "factual_belief": "Believed truth of a specific claim or event.",
        "evaluative_attitude": "Approval / disapproval valence toward a policy, actor, or institution.",
        "identity_anchor": "Salience and content of in-group / out-group identification used to derive political positions.",
        "trust_dimension": "Trust in actors, institutions, media, science, elections.",
        "epistemic_routine": "Habits and shortcuts used to acquire, verify, and correct information.",
        "behavioural_intention": "Intended civic / political action (vote, protest, donate, abstain).",
        "semantic_meaning": "Meaning and valence assigned to political terms, labels, and reference categories.",
        "information_access": "Availability, salience, and retrievability of countervailing information.",
    },
}


def build_attack_ontology() -> Dict[str, Any]:
    tree = {
        "_metadata": METADATA_BLOCK,
        "_compatibility_rules": COMPATIBILITY_RULES_BLOCK,
        "Political_Opinion_Cybermanipulation_Ontology": {
            "Primary_Axis": {
                "Attack_Family": {
                    "Intelligence_Preparation_And_Vulnerability_Analysis": INTELLIGENCE_PREPARATION,
                    "Source_Identity_And_Legitimacy_Manipulation": SOURCE_IDENTITY,
                    "Claim_Frame_And_Narrative_Manipulation": CLAIM_FRAME_NARRATIVE,
                    "Ai_Generated_Synthetic_Media_And_Content": AI_GENERATED_SYNTHETIC_MEDIA,
                    "Multi_Agent_Adversarial_Architecture": MULTI_AGENT_ADVERSARIAL,
                    "Targeting_Delivery_And_Discoverability_Optimization": TARGETING_DELIVERY,
                    "Amplification_Visibility_And_Attention_Manipulation": AMPLIFICATION_VISIBILITY,
                    "Social_Proof_Network_And_Community_Manipulation": SOCIAL_PROOF_NETWORK,
                    "Platform_Information_Environment_And_Media_System_Shaping": PLATFORM_MEDIA_SHAPING,
                    "Cyber_Enabled_Compromise_Coercion_And_Disruption": CYBER_COMPROMISE,
                    "Behavioral_Conversion_Mobilization_Suppression_And_Radicalization": BEHAVIORAL_CONVERSION,
                    "Cognitive_Infrastructure_And_Epistemic_Subversion": COGNITIVE_INFRASTRUCTURE,
                    "Operational_Security_Evasion_Persistence_And_Reconstitution": OPERATIONAL_SECURITY,
                    "Measurement_Experimentation_And_Adaptive_Learning": MEASUREMENT_EXPERIMENTATION,
                    "Narrative_Infrastructure_And_Ecosystem_Capture": NARRATIVE_INFRASTRUCTURE,
                    "Economic_And_Regulatory_Pressure_Weaponisation": ECONOMIC_REGULATORY_PRESSURE,
                    "Insider_Threat_Facilitation_And_Human_Recruitment": INSIDER_THREAT_RECRUITMENT,
                    "Gamification_And_Unwitting_Participant_Mobilisation": GAMIFICATION_PARTICIPANT_MOBILISATION,
                },
            },
            "Secondary_Axes": SECONDARY_AXES,
        },
    }
    leaf_paths = _local_leaf_paths(tree)
    primary_paths = [
        p
        for p in leaf_paths
        if "Political_Opinion_Cybermanipulation_Ontology > Primary_Axis > Attack_Family" in p
    ]
    secondary_paths = [
        p
        for p in leaf_paths
        if "Political_Opinion_Cybermanipulation_Ontology > Secondary_Axes" in p
    ]

    # Bake fully-resolved combinatorial sampling metadata into every leaf.
    enriched = _bake_per_leaf_metadata(tree)

    tree["_metadata"] = {
        **tree["_metadata"],
        "stats": {
            "leaf_count_total": len(leaf_paths),
            "leaf_count_primary_axis": len(primary_paths),
            "leaf_count_secondary_axes": len(secondary_paths),
            "leaves_with_baked_metadata": enriched,
            "primary_family_count": len(
                tree["Political_Opinion_Cybermanipulation_Ontology"]["Primary_Axis"]["Attack_Family"]
            ),
            "secondary_axis_count": len(
                tree["Political_Opinion_Cybermanipulation_Ontology"]["Secondary_Axes"]
            ),
            "compatibility_rule_count": len(COMPATIBILITY_RULES_BLOCK["rules"]),
        },
    }
    tree["_metadata"]["per_leaf_metadata_schema"] = {
        "schema_version": "v4-test-run-1-baked",
        "documentation": (
            "Every primary-axis and secondary-axis leaf carries STRUCTURAL "
            "combinatorial sampling metadata baked at build time. The baked "
            "fields are derived from `_compatibility_rules` plus per-family "
            "defaults; they do NOT encode psychological-amplification "
            "hypotheses. Per-leaf inline overrides take precedence over rules."
        ),
        "fields": [
            "family",
            "complexity_tier",
            "temporal_horizon",
            "epistemic_target",
            "requires_capability",
            "compatible_opinion_domains",
            "incompatible_opinion_domains",
            "natural_companions",
            "natural_predecessors",
            "mutually_exclusive_with_paths",
            "scenario_role",
            "is_classification_axis",
            "mechanism_summary",
        ],
    }
    return tree


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: build_production_attack.py <output_path>", file=sys.stderr)
        sys.exit(2)
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    tree = build_attack_ontology()
    out.write_text(json.dumps(tree, indent=2, ensure_ascii=False), encoding="utf-8")

    # Quick stats. Use the local walker so the builder works in lean Python
    # environments before optional backend dependencies are installed.
    leaves = _local_leaf_paths(tree)
    families = list(tree["Political_Opinion_Cybermanipulation_Ontology"]["Primary_Axis"]["Attack_Family"].keys())
    secondary = list(tree["Political_Opinion_Cybermanipulation_Ontology"]["Secondary_Axes"].keys())
    print(json.dumps({
        "out": str(out),
        "n_leaves": len(leaves),
        "n_primary_families": len(families),
        "primary_families": families,
        "n_secondary_axes": len(secondary),
        "secondary_axes": secondary,
        "n_compatibility_rules": len(COMPATIBILITY_RULES_BLOCK["rules"]),
    }, indent=2))


if __name__ == "__main__":
    main()
