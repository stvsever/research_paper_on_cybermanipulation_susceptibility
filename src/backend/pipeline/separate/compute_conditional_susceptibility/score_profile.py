from __future__ import annotations

"""
Conditional Susceptibility Scoring CLI
---------------------------------------
Scores a pseudoprofile configuration against a fitted conditional susceptibility
artifact produced by Stage 06, then writes an interpreted human-readable .txt report
with hierarchical opinion-domain and feature-hierarchy breakdowns.

Usage
-----
python src/backend/pipeline/separate/compute_conditional_susceptibility/score_profile.py \\
  --artifact-path evaluation/tests/run_1/stage_outputs/06_construct_structural_equation_model/conditional_susceptibility_artifact.json \\
  --config path/to/profile_config.json \\
  --output-dir evaluation/tests/run_1/compute_conditional_susceptibility/

Config JSON structure
---------------------
{
  "profile_id": "optional_id",     // defaults to config filename stem
  "profile": {
    // Mixed nominal + continuous. Missing fields → imputed from training means.
    // Continuous:
    //   age_years                              (numeric, e.g. 35)
    //   big_five_{trait}_mean_pct              (0-100 percentile scale)
    //   big_five_{trait}_{facet}_pct           (0-100 percentile, future subfacet support)
    // Nominal/categorical:
    //   sex                                    (Male | Female | Other)
  },
  "target_attacks":  null,   // null = all attack leaves in artifact; or list of leaf strings
  "target_opinions": null    // null = all opinion leaves in artifact; or list of leaf strings
}

Output
------
{profile_id}.txt  — hierarchical susceptibility report written to --output-dir.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.utils.conditional_susceptibility import (
    BIG_FIVE_TRAITS,  # kept for import compatibility; not used directly
    _build_feature_hierarchy,
    score_profiles_with_conditional_artifact,
)
from src.backend.utils.schemas import ConditionalSusceptibilityArtifact, ConditionalSusceptibilityTaskModel
from src.backend.utils.semantic_scale import get_default_registry as _get_scale_registry


# ---------------------------------------------------------------------------
# Feature key helpers
# ---------------------------------------------------------------------------

def _parse_profile_to_feature_row(
    profile_input: Dict[str, Any],
    artifact: ConditionalSusceptibilityArtifact,
) -> Dict[str, float]:
    """
    Map a user-facing profile dict → model feature column format.

    Fully generic: works for any personality inventory (Big Five, HEXACO,
    Dark Triad, …) and any categorical dimension (sex, education, …) without
    hardcoding any column or field names.

    Resolution strategy (for each key in profile_input):
      1. If the key is already a known feature column → use directly.
      2. If the key starts with "profile_cont_" / "profile_cat__" → use directly
         after checking presence in artifact.feature_columns.
      3. If the value is a string → try to match a categorical level column:
           Look for artifact columns matching "profile_cat__*_{value}".
      4. Otherwise → try prepending "profile_cont_" and use if present in artifact.

    Categorical dimensions (one-hot):
      Discover all groups from artifact.categorical_feature_columns by parsing
      the prefix pattern  profile_cat__{prefix}_{level}
      then for each group with a matching key in profile_input, set the
      active level to 1.0 and all other levels to 0.0.

    Missing feature columns are left absent from the returned dict; the
    downstream scorer will impute them from artifact.feature_means.
    """
    row: Dict[str, float] = {}
    artifact_col_set = set(artifact.feature_columns)

    # ── Build categorical group map from artifact ────────────────────────────
    # group_map: {group_key: {level_label: column_name}}
    # e.g.  {"sex": {"Female": "profile_cat__profile_cat_sex_Female", ...}}
    group_map: Dict[str, Dict[str, str]] = {}
    for col in artifact.categorical_feature_columns:
        # Strip prefix to get "profile_cat_sex_Female" (or similar)
        inner = col.removeprefix("profile_cat__")
        # The last token is the level label; everything before is the group prefix
        parts = inner.split("_")
        if len(parts) < 2:
            continue
        level = parts[-1]
        group_prefix = "_".join(parts[:-1])  # "profile_cat_sex"
        # Also expose the short group key (strip "profile_cat_" if present)
        short_key = group_prefix.removeprefix("profile_cat_")
        group_map.setdefault(group_prefix, {})[level] = col
        group_map.setdefault(short_key, {})[level] = col

    # ── Process each key in the user-supplied profile dict ───────────────────
    for key, val in profile_input.items():
        # 1. Direct full column name
        if key in artifact_col_set:
            row[key] = float(val)
            continue

        # 2. profile_cont_ prefixed continuous
        if key.startswith("profile_cont_"):
            if key in artifact_col_set:
                row[key] = float(val)
            continue

        # 3. String value → categorical one-hot encoding
        if isinstance(val, str):
            # Match against known categorical group keys
            matched = False
            key_lower = key.lower()
            for gkey, level_map in group_map.items():
                if gkey.lower() == key_lower:
                    # Set active level to 1, all others to 0
                    for level_lbl, col_name in level_map.items():
                        if col_name in artifact_col_set:
                            row[col_name] = 1.0 if str(val).strip() == level_lbl else 0.0
                    matched = True
                    break
            if not matched:
                # Try matching by value directly (the level label might be supplied without a key)
                for gkey, level_map in group_map.items():
                    if str(val).strip() in level_map:
                        for level_lbl, col_name in level_map.items():
                            if col_name in artifact_col_set:
                                row[col_name] = 1.0 if str(val).strip() == level_lbl else 0.0
                        break
            continue

        # 4. Numeric value → try profile_cont_ prefix
        bare = key.removeprefix("profile_cat_").removeprefix("profile_cont_")
        candidate = f"profile_cont_{bare}"
        if candidate in artifact_col_set:
            row[candidate] = float(val)
            continue

        # 5. Already a bare continuous key matching a fragment of an artifact column
        for col in artifact.continuous_feature_columns:
            if col.endswith(f"_{bare}") or col.endswith(f"_{bare}_pct"):
                row[col] = float(val)
                break

    return row


def _discover_artifact(run_dir: str) -> Path:
    for sub in [
        Path(run_dir) / "stage_outputs" / "06_construct_structural_equation_model",
        Path(run_dir) / "sem",
        Path(run_dir),
    ]:
        p = sub / "conditional_susceptibility_artifact.json"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"conditional_susceptibility_artifact.json not found under {run_dir}. "
        "Pass --artifact-path explicitly."
    )


# ---------------------------------------------------------------------------
# Hierarchical opinion-domain aggregation
# ---------------------------------------------------------------------------

def _extract_opinion_domain(opinion_leaf: str) -> str:
    """Extract the domain name from a full opinion leaf path."""
    parts = [p.strip() for p in opinion_leaf.split(">")]
    if len(parts) >= 2:
        return parts[-2]
    return parts[0]


def _build_opinion_hierarchy(
    task_models: List[ConditionalSusceptibilityTaskModel],
    profile_predicted: Dict[str, float],  # task_key -> predicted effectivity
) -> Dict[str, Any]:
    """Build a reliability-weighted, hierarchical opinion score structure.

    The hierarchy is:
      overall → domain-level scores → leaf-level scores

    Within each domain, tasks are aggregated using reliability-weight-normalized
    means (not simple averages, because tasks have different CV-MSE quality).
    The overall score is the mean of domain-level scores (equal domain weighting),
    which prevents high-leaf-count domains from dominating the overall estimate.

    Returns:
        {
            "overall_raw_score": float,
            "domains": {
                domain_name: {
                    "domain_raw_score": float,
                    "domain_weight_sum": float,  # sum of reliability weights for this domain
                    "leaves": {
                        leaf_name: {
                            "predicted_effectivity": float,
                            "reliability_weight": float,
                        }
                    }
                }
            }
        }
    """
    # Group tasks by domain
    domain_tasks: Dict[str, List[ConditionalSusceptibilityTaskModel]] = defaultdict(list)
    for tm in task_models:
        domain = _extract_opinion_domain(tm.opinion_leaf)
        domain_tasks[domain].append(tm)

    domain_scores: Dict[str, float] = {}
    domain_details: Dict[str, Any] = {}

    for domain, tasks in domain_tasks.items():
        total_weight = sum(t.reliability_weight for t in tasks)
        if total_weight <= 0:
            continue
        weighted_sum = sum(
            profile_predicted.get(t.task_key, 0.0) * t.reliability_weight
            for t in tasks
        )
        domain_score = weighted_sum / total_weight
        domain_scores[domain] = domain_score
        domain_details[domain] = {
            "domain_raw_score": domain_score,
            "domain_weight_sum": total_weight,
            "leaves": {
                tm.opinion_leaf.split(">")[-1].strip(): {
                    "predicted_effectivity": profile_predicted.get(tm.task_key, 0.0),
                    "reliability_weight": tm.reliability_weight,
                }
                for tm in tasks
            },
        }

    # Overall = mean of domain scores (equal-weight domains)
    overall = float(np.mean(list(domain_scores.values()))) if domain_scores else 0.0

    return {
        "overall_raw_score": overall,
        "domains": domain_details,
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _render_report(
    profile_id: str,
    score_row: Dict[str, Any],
    breakdown_df: pd.DataFrame,
    artifact: ConditionalSusceptibilityArtifact,
    imputed_features: List[str],
    target_attacks: Optional[List[str]],
    target_opinions: Optional[List[str]],
    combined_scores: pd.DataFrame,
    opinion_hierarchy: Dict[str, Any],
) -> str:
    lines: List[str] = [
        "=" * 72,
        "Conditional Susceptibility Report",
        f"Profile ID : {profile_id}",
        f"Outcome    : {artifact.outcome_metric}",
        "=" * 72,
        "",
        "Adversarial Operator Context",
        "----------------------------",
        "  Goal: Maximize aggregate erosion of Western democratic defense",
        "  consensus, multilateral alliance cohesion, and institutional",
        "  security capacity through targeted cyber-manipulation of public",
        "  opinion. Positive susceptibility score = adversary achieves goal.",
        "",
    ]

    # Overall score
    pct = float(score_row.get("susceptibility_index_pct", 0.0))
    raw = float(score_row.get("conditional_susceptibility_raw_score", 0.0))
    n_tasks = int(score_row.get("conditional_target_task_count", 0))

    if pct >= 80:
        band = "HIGH  (top quintile)"
    elif pct >= 60:
        band = "MODERATELY HIGH  (60–80th pct)"
    elif pct >= 40:
        band = "MODERATE  (40–60th pct)"
    elif pct >= 20:
        band = "MODERATELY LOW  (20–40th pct)"
    else:
        band = "LOW  (bottom quintile)"

    lines += [
        "Overall Susceptibility",
        "----------------------",
        f"  Index (percentile vs. population) : {pct:.1f}th  →  {band}",
        f"  Raw model score                   : {raw:+.4f}",
        f"  Conditioned on {n_tasks} attack×opinion task(s).",
    ]
    if target_attacks:
        lines.append(f"  Attack filter  : {', '.join(target_attacks)}")
    if target_opinions:
        lines.append(f"  Opinion filter : {', '.join(target_opinions)}")
    if imputed_features:
        lines.append(
            f"  ⚠ Imputed {len(imputed_features)} missing feature(s) from training mean"
            + (f": {', '.join(imputed_features[:4])}{'…' if len(imputed_features) > 4 else ''}" if imputed_features else "")
        )
    lines += [
        f"  Population reference: {len(combined_scores)} profiles (training + synthetic).",
        "",
    ]

    # Hierarchical opinion-domain scores
    lines += [
        "Opinion-Domain Susceptibility Hierarchy",
        "---------------------------------------",
        "  Aggregation: reliability-weighted mean within domain;",
        "  overall = equal-weight average across domains.",
        f"  Overall raw score: {opinion_hierarchy['overall_raw_score']:+.4f}",
        "",
    ]
    for domain, ddata in sorted(opinion_hierarchy["domains"].items(), key=lambda kv: abs(kv[1]["domain_raw_score"]), reverse=True):
        d_score = ddata["domain_raw_score"]
        d_weight = ddata["domain_weight_sum"]
        lines.append(f"  Domain: {domain}")
        lines.append(f"    Score: {d_score:+.4f}  (reliability weight sum = {d_weight:.4f})")
        for leaf_name, ldata in sorted(ddata["leaves"].items(), key=lambda kv: abs(kv[1]["predicted_effectivity"]), reverse=True):
            pred = ldata["predicted_effectivity"]
            rw = ldata["reliability_weight"]
            direction = "→ toward adversary goal" if pred > 0 else "→ away from adversary goal"
            lines.append(f"      {leaf_name:<45} pred={pred:+.4f}  w={rw:.4f}  {direction}")
    lines.append("")

    # Feature-level contributions (top 10)
    feat_bd = breakdown_df[breakdown_df["component_type"] == "feature"].copy()
    if not feat_bd.empty:
        feat_bd = feat_bd.sort_values("contribution", key=abs, ascending=False)
        lines += [
            "Profile Feature Contributions",
            "-----------------------------",
            "  (ridge coef × standardized feature value, task-reliability weighted)",
            "  +contribution = feature pushes score toward adversary's goal.",
        ]
        scale_reg = _get_scale_registry()
        for _, fr in feat_bd.head(10).iterrows():
            col_name = str(fr["component_name"])
            c = float(fr["contribution"])
            dir_str = "↑" if c > 0 else "↓"
            # Use semantic scale registry for a human-readable label
            sc = scale_reg.get_scale(col_name)
            if sc is not None:
                fname = sc.dimension_label
            else:
                fname = col_name
                for pfx in ["profile_cont_", "profile_cat__profile_cat_",
                             "profile_cat__", "profile_cat_"]:
                    if fname.startswith(pfx):
                        fname = fname[len(pfx):]
                        break
                fname = fname.replace("_", " ").strip()
            lines.append(f"  {fname:<45} {c:+.4f}  {dir_str}")
        lines.append("")

    # Hierarchical feature decomposition from feature groups
    hier = _build_feature_hierarchy(artifact.feature_columns)
    feat_contribs = {
        str(fr["component_name"]): float(fr["contribution"])
        for _, fr in feat_bd.iterrows()
        if fr["component_type"] == "feature"
    }

    group_total_contrib: Dict[str, float] = {}
    for group, cols in hier.items():
        group_total_contrib[group] = sum(feat_contribs.get(c, 0.0) for c in cols)

    total_abs = sum(abs(v) for v in group_total_contrib.values()) or 1.0

    lines += [
        "Profile Feature Hierarchy Contribution",
        "--------------------------------------",
        "  Summed ridge contributions per ontology-aligned feature group.",
        "  Relative share = |group contribution| / sum(|all group contributions|).",
    ]
    for group, contrib in sorted(group_total_contrib.items(), key=lambda kv: abs(kv[1]), reverse=True):
        rel = abs(contrib) / total_abs * 100.0
        dir_str = "→ susceptible" if contrib > 0 else "→ resistant"
        # Human-readable group label (e.g. "big_five_neuroticism" → "Neuroticism")
        sc = _get_scale_registry().get_scale(group)
        group_label = sc.dimension_label if sc else group.replace("_", " ").title()
        lines.append(f"  {group_label:<35} {contrib:+.4f}  ({rel:.1f}%)  {dir_str}")
    lines.append("")

    # Notes
    lines += [
        "Methodological Notes",
        "--------------------",
        f"  Outcome metric: {artifact.outcome_metric}",
        "    adversarial_effectivity = signed opinion delta × adversarial goal",
        "    direction per leaf. Positive = attack moved opinion in adversary's",
        "    intended direction.",
        "  The index is CONDITIONAL on the specified attack/opinion target set.",
        "  It is a population-relative rank; not an absolute shift probability.",
        "  Ridge regularization (CV-selected α) prevents overfitting.",
        "  Missing features imputed from training-set means; uncertainty increases",
        "  with more imputed features.",
        "  Hierarchical opinion aggregation: reliability-weighted within domains,",
        "  equal-weight across domains.",
        "=" * 72,
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score a profile configuration against a fitted conditional susceptibility artifact."
    )
    p.add_argument("--config", required=True, help="Path to profile config JSON.")
    p.add_argument("--artifact-path", default=None, help="Path to conditional_susceptibility_artifact.json.")
    p.add_argument("--run-dir", default=None, help="Run directory; artifact auto-discovered.")
    p.add_argument("--output-dir", required=True, help="Directory for .txt report output.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with config_path.open(encoding="utf-8") as fh:
        config = json.load(fh)

    profile_id: str = config.get("profile_id") or config_path.stem
    profile_input: Dict[str, Any] = config.get("profile", {})
    target_attacks: Optional[List[str]] = config.get("target_attacks")
    target_opinions: Optional[List[str]] = config.get("target_opinions")

    if args.artifact_path:
        artifact_path = Path(args.artifact_path)
    elif args.run_dir:
        artifact_path = _discover_artifact(args.run_dir)
    else:
        raise ValueError("Provide either --artifact-path or --run-dir.")

    with artifact_path.open(encoding="utf-8") as fh:
        artifact = ConditionalSusceptibilityArtifact.model_validate(json.load(fh))

    feature_row = _parse_profile_to_feature_row(profile_input, artifact)
    imputed_features = [c for c in artifact.feature_columns if c not in feature_row]

    # Build a synthetic reference population to establish percentile rank
    rng = np.random.default_rng(42)
    n_pop = 300
    pop_rows = [
        {"profile_id": f"_pop_{i}", **{
            col: float(rng.normal(artifact.feature_means.get(col, 0.0), artifact.feature_stds.get(col, 1.0)))
            for col in artifact.feature_columns
        }}
        for i in range(n_pop)
    ]
    profile_df_single = pd.DataFrame([{"profile_id": profile_id, **feature_row}])
    pop_df = pd.DataFrame(pop_rows)

    combined_df = pd.concat([profile_df_single, pop_df], ignore_index=True)
    for col in artifact.feature_columns:
        if col not in combined_df.columns:
            combined_df[col] = artifact.feature_means.get(col, 0.0)
        elif combined_df[col].isna().any():
            combined_df[col] = combined_df[col].fillna(artifact.feature_means.get(col, 0.0))

    combined_scores, combined_breakdown = score_profiles_with_conditional_artifact(
        profile_df=combined_df,
        artifact=artifact,
        target_attacks=target_attacks,
        target_opinions=target_opinions,
    )

    score_row_df = combined_scores[combined_scores["profile_id"] == profile_id]
    if score_row_df.empty:
        print(f"ERROR: profile '{profile_id}' missing from scored results.", file=sys.stderr)
        sys.exit(1)

    score_row = score_row_df.iloc[0].to_dict()
    profile_breakdown = combined_breakdown[combined_breakdown["profile_id"] == profile_id]

    # Build per-task predicted effectivity for opinion hierarchy
    active_tasks = [
        t for t in artifact.task_models
        if (target_attacks is None or t.attack_leaf in target_attacks)
        and (target_opinions is None or t.opinion_leaf in target_opinions)
    ]
    weight_sum = max(sum(t.reliability_weight for t in active_tasks), 1e-10)
    profile_predicted: Dict[str, float] = {}
    for t in active_tasks:
        slug = (
            t.task_key.lower()
            .replace(" ", "_").replace(">", "_").replace("|", "_").replace("/", "_")
        )
        col = f"predicted_effectivity__{slug}"
        profile_predicted[t.task_key] = float(score_row.get(col, 0.0))

    opinion_hierarchy = _build_opinion_hierarchy(active_tasks, profile_predicted)

    report_text = _render_report(
        profile_id=profile_id,
        score_row=score_row,
        breakdown_df=profile_breakdown,
        artifact=artifact,
        imputed_features=imputed_features,
        target_attacks=target_attacks,
        target_opinions=target_opinions,
        combined_scores=combined_scores,
        opinion_hierarchy=opinion_hierarchy,
    )

    out_path = output_dir / f"{profile_id}.txt"
    out_path.write_text(report_text, encoding="utf-8")
    print(f"Report written to: {out_path}")
    print(f"Susceptibility index: {score_row.get('susceptibility_index_pct', 'n/a'):.1f}th percentile")


if __name__ == "__main__":
    main()
