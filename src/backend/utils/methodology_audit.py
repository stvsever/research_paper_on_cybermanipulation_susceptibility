from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.backend.utils.data_utils import infer_analysis_mode
from src.backend.utils.schemas import SemFitResult


def build_assumption_register(df: pd.DataFrame, sem_result: SemFitResult) -> List[Dict[str, Any]]:
    n_rows = len(df)
    n_profiles = int(df["profile_id"].nunique()) if "profile_id" in df.columns else n_rows
    n_tasks = (
        int(df[["attack_leaf", "opinion_leaf"]].drop_duplicates().shape[0])
        if {"attack_leaf", "opinion_leaf"}.issubset(df.columns)
        else 0
    )
    analysis_mode = infer_analysis_mode(df)
    attack_ratio = float(df["attack_present"].mean()) if "attack_present" in df.columns and n_rows else 0.0
    score_unique = int(df["baseline_score"].nunique()) if "baseline_score" in df.columns else 0

    mean_realism = None
    if "attack_realism_score" in df.columns:
        attack_realism = df.loc[df["attack_present"] == 1, "attack_realism_score"].dropna()
        if len(attack_realism) > 0:
            mean_realism = float(attack_realism.mean())

    fit_indices = sem_result.fit_indices or {}
    cfi = fit_indices.get("CFI")
    rmsea = fit_indices.get("RMSEA")
    fit_is_acceptable = (
        sem_result.converged
        and cfi is not None
        and rmsea is not None
        and float(cfi) >= 0.9
        and float(rmsea) <= 0.1
    )
    near_saturated_fit = (
        sem_result.converged
        and cfi is not None
        and rmsea is not None
        and float(cfi) >= 0.99
        and float(rmsea) <= 0.01
        and n_profiles <= 30
    )

    mean_plausibility = None
    if "post_plausibility_score" in df.columns:
        post_plausibility = df["post_plausibility_score"].dropna()
        if len(post_plausibility) > 0:
            mean_plausibility = float(post_plausibility.mean())

    baseline_fallback_rate = None
    if "baseline_fallback_used" in df.columns:
        baseline_fallback_rate = float(df["baseline_fallback_used"].fillna(False).astype(bool).mean())

    post_fallback_rate = None
    if "post_fallback_used" in df.columns:
        post_fallback_rate = float(df["post_fallback_used"].fillna(False).astype(bool).mean())

    attack_heuristic_pass_rate = None
    if "attack_heuristic_pass" in df.columns:
        attack_heuristic_pass_rate = float(df["attack_heuristic_pass"].fillna(False).astype(bool).mean())

    baseline_heuristic_pass_rate = None
    if "baseline_heuristic_pass" in df.columns:
        baseline_heuristic_pass_rate = float(df["baseline_heuristic_pass"].fillna(False).astype(bool).mean())

    post_heuristic_pass_rate = None
    if "post_heuristic_pass" in df.columns:
        post_heuristic_pass_rate = float(df["post_heuristic_pass"].fillna(False).astype(bool).mean())

    assumptions = [
        {
            "assumption": "Profile-panel sample size is sufficient for stable moderation estimates.",
            "status": "risk" if n_profiles < 80 else "ok",
            "evidence": {
                "n_profiles": n_profiles,
                "n_rows": n_rows,
                "n_attack_opinion_tasks": n_tasks,
            },
            "mitigation": "Interpret run_1 coefficients as exploratory panel diagnostics and scale the profile count before making substantive moderator claims.",
        },
        (
            {
                "assumption": "Attack-only design is intentional and aligned with the moderation question.",
                "status": "ok" if analysis_mode == "treated_only" else "risk",
                "evidence": {"analysis_mode": analysis_mode, "attack_ratio": attack_ratio},
                "mitigation": "Interpret results as moderation among attacked individuals, not as a no-attack counterfactual contrast.",
            }
            if analysis_mode == "treated_only"
            else {
                "assumption": "Treatment-control balance is adequate.",
                "status": "ok" if 0.35 <= attack_ratio <= 0.65 else "risk",
                "evidence": {"attack_ratio": attack_ratio},
                "mitigation": "Use explicit stratified assignment in scenario generator.",
            }
        ),
        {
            "assumption": "Opinion measurement has enough numeric resolution.",
            "status": "ok" if score_unique >= max(6, int(0.4 * min(n_rows, 100))) else "risk",
            "evidence": {"unique_baseline_scores": score_unique},
            "mitigation": "Prompt enforces high-resolution scores in [-1000,1000].",
        },
        {
            "assumption": "Attack-generation quality checks indicate usable exposure texts.",
            "status": (
                "risk"
                if (mean_realism is None or mean_realism < 0.7)
                or (attack_heuristic_pass_rate is not None and attack_heuristic_pass_rate < 0.8)
                else "ok"
            ),
            "evidence": {
                "mean_attack_realism_score": mean_realism,
                "attack_heuristic_pass_rate": attack_heuristic_pass_rate,
            },
            "mitigation": "Inspect Stage 03 reviewer outputs and treat zero-scored or fallback-generated attack content as a run-quality failure rather than as substantive evidence.",
        },
        {
            "assumption": "Baseline and post-exposure opinions were generated without heavy fallback substitution.",
            "status": (
                "risk"
                if (baseline_fallback_rate is not None and baseline_fallback_rate > 0.05)
                or (post_fallback_rate is not None and post_fallback_rate > 0.05)
                or (mean_plausibility is not None and mean_plausibility < 0.7)
                else "ok"
            ),
            "evidence": {
                "baseline_fallback_rate": baseline_fallback_rate,
                "post_fallback_rate": post_fallback_rate,
                "baseline_heuristic_pass_rate": baseline_heuristic_pass_rate,
                "post_heuristic_pass_rate": post_heuristic_pass_rate,
                "mean_post_plausibility_score": mean_plausibility,
            },
            "mitigation": "Re-run stages 02–04 with funded API access before interpreting moderator coefficients as anything other than pipeline-diagnostic outputs.",
        },
        {
            "assumption": "SEM converges with acceptable fit.",
            "status": "risk" if (not fit_is_acceptable or near_saturated_fit) else "ok",
            "evidence": {
                "sem_converged": sem_result.converged,
                "CFI": cfi,
                "RMSEA": rmsea,
                "near_saturated_fit": near_saturated_fit,
                "warnings": sem_result.warnings,
            },
            "mitigation": "Treat near-perfect fit in small profile panels as a saturation warning; rely on robustness diagnostics and larger panels instead of fit indices alone.",
        },
        {
            "assumption": "LLM outputs are reproducible and auditable.",
            "status": "ok",
            "evidence": {"raw_llm_logging": True},
            "mitigation": "Persist all prompts/outputs in provenance/raw_llm.",
        },
    ]
    return assumptions


def build_peer_review_critique_notes(df: pd.DataFrame, sem_result: SemFitResult) -> List[Dict[str, str]]:
    analysis_mode = infer_analysis_mode(df)
    fit_indices = sem_result.fit_indices or {}
    cfi = fit_indices.get("CFI")
    rmsea = fit_indices.get("RMSEA")
    n_profiles = int(df["profile_id"].nunique()) if "profile_id" in df.columns else len(df)
    baseline_fallback_rate = (
        float(df["baseline_fallback_used"].fillna(False).astype(bool).mean())
        if "baseline_fallback_used" in df.columns
        else None
    )
    post_fallback_rate = (
        float(df["post_fallback_used"].fillna(False).astype(bool).mean())
        if "post_fallback_used" in df.columns
        else None
    )
    notes = [
        {
            "critique": "Synthetic LLM agents may not represent human cognition or causal response behavior.",
            "implemented_change": "Added transparent assumption register and explicit exploratory caveat in reports.",
        },
        {
            "critique": "Adversarial content may be unrealistic or too generic.",
            "implemented_change": "Added realism review agent with rewrite loop and heuristic checks.",
        },
        {
            "critique": "Model dependence on a single LLM could bias outputs.",
            "implemented_change": "Model is CLI-configurable and run metadata captures exact model for replication.",
        },
        {
            "critique": "Insufficient sample size inflates uncertainty and may destabilize SEM.",
            "implemented_change": "Included robust OLS complement and explicit testing-run warnings; ready for scale-up runs.",
        },
    ]

    if analysis_mode == "treated_only":
        notes.append(
            {
                "critique": "Attack-only designs cannot estimate the incremental effect of exposure relative to a no-attack counterfactual.",
                "implemented_change": "The report states explicitly that the attacked-only design estimates heterogeneity of post-minus-baseline response among attacked individuals only.",
            }
        )

    if n_profiles < 20:
        notes.append(
            {
                "critique": "The profile panel is too small for publication-grade inferential claims.",
                "implemented_change": "Outputs are marked as exploratory profile-panel diagnostics and the report now distinguishes profile count from attacked-row count.",
            }
        )

    if (
        baseline_fallback_rate is not None
        and post_fallback_rate is not None
        and (baseline_fallback_rate > 0.05 or post_fallback_rate > 0.05)
    ):
        notes.append(
            {
                "critique": "Observed outcome shifts may be dominated by fallback scoring rather than by genuine model-based opinion elicitation.",
                "implemented_change": "Fallback rates are surfaced explicitly in the audit register, dashboard notes, and manuscript so run_1 is interpreted as a test-run validation rather than as substantive behavioral evidence.",
            }
        )

    if sem_result.warnings:
        notes.append(
            {
                "critique": "SEM warnings indicate possible misfit or numerical instability.",
                "implemented_change": "Warnings surfaced in reports and fit diagnostics exported for scrutiny.",
            }
        )

    if (
        not sem_result.converged
        or cfi is None
        or rmsea is None
        or float(cfi) < 0.9
        or float(rmsea) > 0.1
        or (float(cfi) >= 0.99 and float(rmsea) <= 0.01 and n_profiles <= 30)
    ):
        notes.append(
            {
                "critique": "Global SEM fit is not evidential at this scale; either misfit or near-saturation can make fit indices misleading.",
                "implemented_change": "The report now treats SEM fit indices as diagnostics only and triangulates them with ICC, bootstrap, ridge, and fallback-quality diagnostics.",
            }
        )

    return notes


def render_methodology_audit_text(
    assumptions: List[Dict[str, Any]],
    critiques: List[Dict[str, str]],
) -> str:
    lines: List[str] = [
        "Methodology Audit and Peer-Review Risk Register",
        "===============================================",
        "",
        "Assumption Register",
        "-------------------",
    ]

    for idx, item in enumerate(assumptions, start=1):
        lines.append(f"{idx}. Assumption: {item['assumption']}")
        lines.append(f"   Status: {item['status']}")
        lines.append(f"   Evidence: {item['evidence']}")
        lines.append(f"   Mitigation: {item['mitigation']}")
        lines.append("")

    lines.append("Peer-Review Critique Mitigations")
    lines.append("-------------------------------")
    for idx, item in enumerate(critiques, start=1):
        lines.append(f"{idx}. Potential critique: {item['critique']}")
        lines.append(f"   Implemented change: {item['implemented_change']}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"
