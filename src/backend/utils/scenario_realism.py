from __future__ import annotations

"""
Technical overview
------------------
This module contains the realism layer that sits between pure ontology sampling
and the LLM-facing prompt payloads. Its job is to turn abstract leaf-node
choices into scenario context that is realistic enough to constrain the
simulation, without pretending to be the final analytical susceptibility model.

The functions here do two main things:
- derive lightweight heuristic context used to bound plausible opinion movement
- build attack-side prompt context and heuristic checks for downstream review

Important distinction:
- `heuristic_shift_sensitivity_proxy` and `resilience_index` are legacy realism
  helpers used for boundedness and prompt guidance
- they are not the analysis-facing susceptibility construct used in Stage 06

So this module is about keeping scenarios plausible and reviewable, not about
estimating the final moderation answer to the research question.
"""

from typing import Dict, List

from src.backend.utils.schemas import ProfileConfiguration


def extract_opinion_domain(opinion_leaf: str) -> str:
    parts = [part.strip() for part in opinion_leaf.split(">")]
    if len(parts) >= 2:
        return parts[1]
    return parts[0]


def extract_leaf_label(path: str) -> str:
    return path.split(">")[-1].strip()


def pretty_label(value: str) -> str:
    return value.replace("_", " ").strip()


def bounded(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_shift_sensitivity_proxy(profile: ProfileConfiguration) -> float:
    values = profile.continuous_attributes
    neuro = values.get("big_five_neuroticism_mean_pct", 50.0) / 100.0
    consc = values.get("big_five_conscientiousness_mean_pct", 50.0) / 100.0
    open_ = values.get("big_five_openness_to_experience_mean_pct", 50.0) / 100.0
    age = values.get("age_years", 42.0)
    age_factor = 1.0 - bounded(age / 80.0, 0.0, 1.0)

    # Optional extended-ontology moderators (graceful defaults if absent)
    # Dual-Process: low analytical thinking → higher susceptibility
    analytical = values.get("dual_process_inventory_analytical_thinking_mean_pct", 50.0) / 100.0
    # Digital literacy: lower = more susceptible to platform-native misinformation
    digital_lit = values.get("digital_literacy_inventory_critical_evaluation_mean_pct", 50.0) / 100.0
    # Civic engagement: higher engagement can mean more entrenched opinions (lower drift)
    civic_eng = values.get("political_engagement_inventory_news_engagement_mean_pct", 50.0) / 100.0

    base_score = 0.40 * neuro + 0.22 * (1.0 - consc) + 0.18 * age_factor + 0.10 * (1.0 - open_)
    ext_score = 0.06 * (1.0 - analytical) + 0.04 * (1.0 - digital_lit)
    score = base_score + ext_score
    return round(bounded(score, 0.0, 1.0), 4)


def compute_susceptibility_index(profile: ProfileConfiguration) -> float:
    return compute_shift_sensitivity_proxy(profile)


def compute_resilience_index(profile: ProfileConfiguration) -> float:
    return round(1.0 - compute_shift_sensitivity_proxy(profile), 4)


def profile_context_snapshot(profile: ProfileConfiguration) -> Dict[str, float | str]:
    values = profile.continuous_attributes
    return {
        "sex": profile.categorical_attributes.get("sex", "Unknown"),
        "age_years": float(values.get("age_years", 42.0)),
        "big_five_neuroticism_mean_pct": float(values.get("big_five_neuroticism_mean_pct", 50.0)),
        "big_five_conscientiousness_mean_pct": float(values.get("big_five_conscientiousness_mean_pct", 50.0)),
        "big_five_openness_to_experience_mean_pct": float(values.get("big_five_openness_to_experience_mean_pct", 50.0)),
        "heuristic_shift_sensitivity_proxy": compute_shift_sensitivity_proxy(profile),
        "resilience_index": compute_resilience_index(profile),
    }


def control_exposure_template(opinion_leaf: str) -> str:
    domain = pretty_label(extract_opinion_domain(opinion_leaf))
    leaf = pretty_label(extract_leaf_label(opinion_leaf))
    return (
        f"A neutral civic-information post summarizes balanced arguments about {domain}. "
        f"It explains policy background for {leaf} without persuasive or manipulative framing."
    )


def build_attack_context(
    opinion_leaf: str,
    attack_leaf: str,
    profile: ProfileConfiguration,
    baseline_score: int,
    adversarial_direction: int = 0,
    *,
    attack_metadata: Dict[str, object] | None = None,
) -> Dict[str, object]:
    """Build attack context for the exposure generator.

    adversarial_direction: +1 (adversary wants score to INCREASE), -1 (wants DECREASE), 0 (unspecified).

    attack_metadata (current design): optional resolved STRUCTURAL metadata from the
    ATTACK ontology (mechanism, primary_system, platform_hint, complexity_tier,
    temporal_horizon, epistemic_target, requires_personalization,
    agent_orchestration_required). Used purely as descriptive context for the
    exposure agent — it tells the LLM what KIND of attack to instantiate, not
    what psychological lever should resonate with the target.

    No psychological-amplification signal is constructed here: the per-profile
    lever-match is exactly the quantity the inferential layer estimates and
    must not be encoded into the scenario context.
    """
    domain = extract_opinion_domain(opinion_leaf)
    leaf_label = extract_leaf_label(opinion_leaf)
    context = profile_context_snapshot(profile)
    shift_sensitivity_proxy = float(context["heuristic_shift_sensitivity_proxy"])
    openness = float(context["big_five_openness_to_experience_mean_pct"])
    neuroticism = float(context["big_five_neuroticism_mean_pct"])
    conscientiousness = float(context["big_five_conscientiousness_mean_pct"])

    if domain == "Defense_and_National_Security":
        issue_frame = "national security, civil liberties, deterrence, alliance burden-sharing, and public preparedness"
    elif domain == "Foreign_Policy_and_Geopolitics":
        issue_frame = "alliances, strategic autonomy, diplomacy, sanctions, and international order"
    elif domain == "Immigration_and_Citizenship":
        issue_frame = "migration, citizenship, border governance, and social cohesion"
    elif domain == "Information_Integrity_and_Platforms":
        issue_frame = "platform regulation, content moderation, algorithmic amplification, AI-generated content, and epistemic trust"
    elif domain == "Civil_Liberties_and_Surveillance":
        issue_frame = "privacy, surveillance, digital rights, due-process safeguards, and the security-vs-liberty trade-off"
    elif domain == "Democratic_Resilience_and_Institutions":
        issue_frame = "institutional trust, electoral integrity, opposition legitimacy, judicial independence, and civic guardrails"
    elif domain == "Technology_and_AI_Governance":
        issue_frame = "AI regulation, algorithmic transparency, cybersecurity governance, and digital infrastructure accountability"
    elif domain == "Critical_Infrastructure_and_Energy_Sovereignty":
        issue_frame = "energy independence, supply-chain resilience, strategic-reserves policy, and critical-infrastructure protection"
    else:
        # Generic fallback for any future opinion domains
        issue_frame = f"{domain.replace('_', ' ').lower()}, related policy debates, and societal impacts"

    # ── Direction-aware persuasion goal ──────────────────────────────────────
    # Determine whether the baseline is aligned with (reinforcement) or opposed
    # to (erosion / shift) the adversarial goal.  This is the most important
    # context signal for the attack generator: without it the LLM defaults to
    # reinforcing whatever the baseline says, which systematically defeats the
    # adversary when the baseline is in the "wrong" direction.
    #
    # Alignment: baseline sign matches adversarial_direction (e.g. positive
    # baseline AND adversary wants increase, or negative baseline AND adversary
    # wants decrease).  Misalignment: they conflict.
    baseline_sign = 0 if baseline_score == 0 else (1 if baseline_score > 0 else -1)
    baseline_aligned = (adversarial_direction != 0) and (baseline_sign == adversarial_direction or baseline_sign == 0)
    baseline_misaligned = (adversarial_direction != 0) and (baseline_sign != 0) and (baseline_sign != adversarial_direction)
    baseline_strength = abs(baseline_score)

    if adversarial_direction == 0:
        # No direction info — fall back to neutral framing
        if baseline_strength < 180:
            persuasion_goal = "nudge an ambivalent opinion into a clearer directional stance"
        elif baseline_strength < 500:
            persuasion_goal = "modestly intensify an existing leaning"
        else:
            persuasion_goal = "gently reinforce an existing strong opinion"
    elif baseline_misaligned:
        # Adversary wants to shift the person AGAINST their current position
        if baseline_strength < 300:
            persuasion_goal = (
                f"shift a weakly {'positive' if baseline_sign > 0 else 'negative'} opinion "
                f"toward a {'more positive' if adversarial_direction > 0 else 'more negative'} stance "
                f"on {leaf_label.replace('_', ' ')} through targeted framing and selective evidence"
            )
        else:
            persuasion_goal = (
                f"gradually erode a strong {'pro' if baseline_sign > 0 else 'anti'} position on "
                f"{leaf_label.replace('_', ' ')} — introduce doubt, reframe costs/risks, and soften "
                f"conviction without triggering overt reactance; aim to move the score "
                f"{'upward' if adversarial_direction > 0 else 'downward'} even if the shift is modest"
            )
    elif baseline_aligned:
        # Adversary and baseline agree — reinforce and amplify
        if baseline_strength < 300:
            persuasion_goal = (
                f"intensify and solidify an already {'favorable' if baseline_sign > 0 else 'negative'} "
                f"opinion on {leaf_label.replace('_', ' ')} — push it further in the "
                f"{'positive' if adversarial_direction > 0 else 'negative'} direction"
            )
        else:
            persuasion_goal = (
                f"reinforce and deepen the existing strong "
                f"{'support for' if baseline_sign > 0 else 'opposition to'} {leaf_label.replace('_', ' ')}"
            )
    else:
        persuasion_goal = "nudge the opinion in a direction consistent with the adversarial goal"

    # ── Direction-aware emotional register (NOT profile-conditioned) ─────────
    # We provide the adversary a direction-appropriate emotional register
    # without conditioning on the target profile's traits. The whole point of
    # the study is to estimate which profiles are differentially susceptible
    # to which emotional registers; the attack writer must not be told the
    # answer in advance.
    if adversarial_direction < 0:
        emotional_register = (
            "threat salience, doubt, systemic-failure framing, loss framing, and elite distrust cues"
        )
    elif adversarial_direction > 0:
        emotional_register = (
            "identity affirmation, opportunity framing, social-proof reassurance, and ordered-benefit framing"
        )
    else:
        emotional_register = "neutral civic-information register"

    # ── Direction-aware motivational lever menu (NOT profile-conditioned) ────
    if adversarial_direction != 0 and baseline_misaligned:
        motivational_lever = (
            "introduce doubt about the current position, reframe its costs / risks, surface alternative "
            "expert framings, and erode confidence without overt insult"
        )
    elif adversarial_direction != 0 and baseline_aligned:
        motivational_lever = (
            "intensify and solidify the existing position, supply confirmatory anecdotes, and harden the "
            "boundary against opposing arguments"
        )
    else:
        motivational_lever = (
            "construct a plausible directional case using selective evidence, identity-compatible framing, "
            "and trust-relevant cues"
        )

    # ── Shift band — wider when misaligned because stronger push needed ───────
    # The band is a structural realism guide for the heuristic check; it is
    # NOT conditioned on profile-attack lever matching.
    band_multiplier = 1.4 if baseline_misaligned and baseline_strength >= 400 else 1.0
    shift_dir_label = "upward" if adversarial_direction > 0 else ("downward" if adversarial_direction < 0 else "any")

    attack_meta = attack_metadata or {}

    context.update(
        {
            "opinion_domain": domain,
            "opinion_leaf_label": leaf_label,
            "attack_leaf_label": extract_leaf_label(attack_leaf),
            "baseline_score": baseline_score,
            "adversarial_direction": adversarial_direction,
            "adversarial_direction_label": (
                f"INCREASE (score must move {shift_dir_label}, toward +1000)"
                if adversarial_direction > 0
                else (
                    f"DECREASE (score must move {shift_dir_label}, toward -1000)"
                    if adversarial_direction < 0
                    else "UNSPECIFIED"
                )
            ),
            "baseline_vs_goal": (
                "ALIGNED — baseline already points in the adversarial direction; reinforce it"
                if baseline_aligned
                else (
                    "MISALIGNED — baseline opposes the adversarial goal; the attack must work AGAINST the current opinion"
                    if baseline_misaligned
                    else "AMBIVALENT — baseline near zero; push in adversarial direction"
                )
            ),
            "emotional_register": emotional_register,
            "issue_frame": issue_frame,
            "motivational_lever": motivational_lever,
            "persuasion_goal": persuasion_goal,
            "recommended_shift_band": {
                "direction": shift_dir_label,
                "low": round((20 + 40 * shift_sensitivity_proxy) * band_multiplier, 1),
                "typical": round((40 + 120 * shift_sensitivity_proxy) * band_multiplier, 1),
                "upper": round((90 + 210 * shift_sensitivity_proxy) * band_multiplier, 1),
            },
            # Structural attack metadata only — descriptive of the attack vector,
            # never predictive of which profile features amplify or dampen it.
            "attack_mechanism": str(attack_meta.get("mechanism", "")),
            "attack_primary_system": str(attack_meta.get("primary_system", "")),
            "attack_platform_hint": str(attack_meta.get("platform_hint", "")),
            "attack_complexity_tier": str(attack_meta.get("complexity_tier", "")),
            "attack_temporal_horizon": str(attack_meta.get("temporal_horizon", "")),
            "attack_epistemic_target": str(attack_meta.get("epistemic_target", "")),
            "attack_requires_personalization": bool(attack_meta.get("requires_personalization", False)),
            "attack_agent_orchestration_required": bool(attack_meta.get("agent_orchestration_required", False)),
            "paper_goal": (
                "Investigate how inter-individual differences moderate the effectivity of cyber-manipulation "
                "on cognitive sovereignty within a high-dimensional political opinion state space."
            ),
            "targeting_note": (
                "Use realistic platform-native misinformation tactics, preserve policy-topic anchoring, "
                "avoid impossible claims, and avoid generic propaganda language. The exposure must push "
                "toward the adversarial direction even when the baseline opposes it."
            ),
        }
    )
    return context


def assess_attack_exposure_heuristics(exposure_text: str, attack_leaf: str, opinion_leaf: str) -> Dict[str, object]:
    text = exposure_text.lower()
    checks: Dict[str, bool] = {
        "length_reasonable": 35 <= len(exposure_text) <= 900,
        "contains_issue_anchor": any(
            token in text
            for token in [
                extract_leaf_label(opinion_leaf).split("_")[0].lower(),
                extract_opinion_domain(opinion_leaf).split("_")[0].lower(),
            ]
        ),
        "contains_attack_theme": any(
            token in text for token in extract_leaf_label(attack_leaf).lower().split("_")[:2]
        ),
        "non_hacking_language": all(term not in text for term in ["malware", "exploit", "phishing kit", "ddos"]),
    }
    checks["overall_pass"] = sum(1 for value in checks.values() if value) >= 3
    return {
        "checks": checks,
        "pass_count": sum(1 for key, value in checks.items() if key != "overall_pass" and value),
    }


def assess_baseline_opinion_heuristics(score: int, confidence: float) -> Dict[str, object]:
    checks: Dict[str, bool] = {
        "within_scale": -1000 <= score <= 1000,
        "high_resolution": abs(score) % 50 != 0,
        "confidence_bounded": 0.0 <= confidence <= 1.0,
    }
    checks["overall_pass"] = all(checks.values())
    return {
        "checks": checks,
        "pass_count": sum(1 for key, value in checks.items() if key != "overall_pass" and value),
    }


def assess_post_opinion_heuristics(
    baseline_score: int,
    post_score: int,
    attack_present: bool,
    intensity_hint: float,
    shift_sensitivity_proxy: float,
    adversarial_direction: int = 0,
) -> Dict[str, object]:
    """Run post-exposure heuristic checks on a (baseline, post) pair.

    The shift band is a STRUCTURAL realism guard: it bounds the magnitude a
    single short exposure can plausibly produce, and is NOT conditioned on
    any profile-attack lever match. The heuristic_shift_sensitivity_proxy is
    a coarse moderation prior used only to set the band envelope; the
    inferential layer estimates the real per-profile susceptibility.

    Directional consistency is STRICT: when an adversarial goal direction is
    defined, the post score must lie between the baseline and the goal pole
    (zero movement, fully resisted, is allowed). Any counter-goal movement
    fails the check and triggers the rewrite loop; stage 04 additionally
    clamps survivors to the baseline so the final dataset cannot contain
    counter-goal deltas.
    """
    delta = post_score - baseline_score
    max_shift = 80.0 + (280.0 * intensity_hint) + (220.0 * shift_sensitivity_proxy)
    if not attack_present:
        max_shift = 120.0

    delta_sign = 0 if delta == 0 else (1 if delta > 0 else -1)

    counter_goal_movement = (
        adversarial_direction != 0
        and delta_sign != 0
        and delta_sign != adversarial_direction
    )

    checks: Dict[str, bool] = {
        "within_scale": -1000 <= post_score <= 1000,
        "high_resolution": abs(post_score) % 50 != 0,
        "bounded_shift": abs(delta) <= max_shift,
        "control_stability": True if attack_present else abs(delta) <= 120,
        "control_not_exact_clone": True if attack_present else abs(delta) >= 3,
        "direction_consistent": not counter_goal_movement,
    }
    checks["overall_pass"] = all(checks.values())
    return {
        "checks": checks,
        "pass_count": sum(1 for key, value in checks.items() if key != "overall_pass" and value),
        "delta": delta,
        "max_reasonable_shift": round(max_shift, 3),
        "counter_goal_flagged": counter_goal_movement,
    }
