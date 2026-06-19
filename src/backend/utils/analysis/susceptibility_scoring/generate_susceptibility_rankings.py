"""
Susceptibility Rankings Generator
----------------------------------
Generates comprehensive susceptibility rankings for all possible
attack×opinion target-set configurations derived from a run's fitted
conditional susceptibility artifact.

For a run with N tasks (attack×opinion pairs), this script enumerates
all 2^N - 1 non-empty subsets, scores all training profiles under each
subset, and writes:

  full/
    susceptibility_rankings_all_configs.txt   — comprehensive human-readable ranking
    susceptibility_rankings_all_configs.csv   — machine-readable; one row per
                                                profile × config × opinion × attack
  01_separated/
    config_{label}.txt                        — one file per configuration

Key design: NO aggregation across opinions within a config. Every row in
the CSV carries a specific attack_leaf and opinion_leaf so the table is
fully compatible with multi-attack, multi-opinion future runs.

Usage
-----
python src/backend/pipeline/01_separated/compute_conditional_susceptibility/generate_susceptibility_rankings.py \\
  --run-dir evaluation/tests/run_1 \\
  --output-dir evaluation/tests/run_1/compute_conditional_susceptibility
"""

from __future__ import annotations

import argparse
import itertools
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

from src.backend.utils.analysis.conditional_susceptibility import (
    score_profiles_with_conditional_artifact,
)
from src.backend.utils.schemas import (
    ConditionalSusceptibilityArtifact,
    ConditionalSusceptibilityTaskModel,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

TRAIT_MEANS = [
    "big_five_agreeableness_mean_pct",
    "big_five_conscientiousness_mean_pct",
    "big_five_extraversion_mean_pct",
    "big_five_neuroticism_mean_pct",
    "big_five_openness_to_experience_mean_pct",
]
TRAIT_LABELS = {
    "big_five_agreeableness_mean_pct": "Agreeableness",
    "big_five_conscientiousness_mean_pct": "Conscientiousness",
    "big_five_extraversion_mean_pct": "Extraversion",
    "big_five_neuroticism_mean_pct": "Neuroticism",
    "big_five_openness_to_experience_mean_pct": "Openness",
}

# ──────────────────────────────────────────────────────────────────────────────
# Discovery helpers
# ──────────────────────────────────────────────────────────────────────────────

def _discover_artifact(run_dir: Path) -> Path:
    for sub in [
        run_dir / "stage_outputs" / "06_construct_structural_equation_model",
        run_dir / "sem",
        run_dir,
    ]:
        p = sub / "conditional_susceptibility_artifact.json"
        if p.exists():
            return p
    raise FileNotFoundError(f"conditional_susceptibility_artifact.json not found under {run_dir}.")


def _discover_sem_long(run_dir: Path) -> Optional[Path]:
    for sub in [
        run_dir / "stage_outputs" / "05_compute_effectivity_deltas",
        run_dir,
    ]:
        p = sub / "sem_long_encoded.csv"
        if p.exists():
            return p
    return None


def _discover_scenarios(run_dir: Path) -> Optional[Path]:
    for sub in [
        run_dir / "stage_outputs" / "01_create_scenarios",
        run_dir,
    ]:
        for name in ["scenarios.json", "scenarios.jsonl"]:
            p = sub / name
            if p.exists():
                return p
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Data loaders
# ──────────────────────────────────────────────────────────────────────────────

def _load_profile_display(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load human-readable profile attributes from scenarios (age, sex, Big Five means)."""
    p = _discover_scenarios(run_dir)
    if p is None:
        return {}
    raw = p.read_text(encoding="utf-8")
    scenarios = json.loads(raw) if p.suffix == ".json" else [
        json.loads(line) for line in raw.splitlines() if line.strip()
    ]
    profiles: Dict[str, Dict[str, Any]] = {}
    for s in scenarios:
        prof = s.get("profile", {})
        pid = prof.get("profile_id", s.get("profile_id", ""))
        if not pid or pid in profiles:
            continue
        cat = prof.get("categorical_attributes", {})
        cont = prof.get("continuous_attributes", {})
        profiles[pid] = {
            "sex": cat.get("sex", "?"),
            "age_years": cont.get("age_years", float("nan")),
            **{t: cont.get(t, float("nan")) for t in TRAIT_MEANS},
        }
    return profiles


def _load_observed_per_task(run_dir: Path) -> pd.DataFrame:
    """Load per-profile×opinion×attack observed effectivity rows from sem_long."""
    p = _discover_sem_long(run_dir)
    if p is None:
        return pd.DataFrame()
    df = pd.read_csv(p)
    keep = [c for c in ["profile_id", "attack_leaf", "opinion_leaf", "opinion_leaf_label",
                         "baseline_score", "post_score", "delta_score", "abs_delta_score",
                         "adversarial_effectivity", "adversarial_direction"] if c in df.columns]
    return df[keep].copy()


# ──────────────────────────────────────────────────────────────────────────────
# Leaf utilities
# ──────────────────────────────────────────────────────────────────────────────

def _leaf_short(leaf: str) -> str:
    return leaf.split(">")[-1].strip()


def _leaf_domain(leaf: str) -> str:
    parts = [p.strip() for p in leaf.split(">")]
    return parts[-2] if len(parts) >= 2 else parts[0]


def _config_label(tasks: Tuple[ConditionalSusceptibilityTaskModel, ...]) -> str:
    opinion_shorts = sorted(_leaf_short(t.opinion_leaf) for t in tasks)
    if len(opinion_shorts) == 1:
        label = opinion_shorts[0]
    elif len(opinion_shorts) <= 3:
        label = "_x_".join(o[:12].replace(" ", "") for o in opinion_shorts)
    else:
        label = f"all_{len(opinion_shorts)}_opinions"
    return label.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")[:64]


def _config_title(tasks: Tuple[ConditionalSusceptibilityTaskModel, ...]) -> str:
    attacks = sorted({_leaf_short(t.attack_leaf) for t in tasks})
    opinions = sorted(_leaf_short(t.opinion_leaf) for t in tasks)
    return f"Attack(s): {', '.join(attacks)}  |  Opinion(s): {', '.join(opinions)}"


def _bar(value: float, min_val: float, max_val: float, width: int = 20) -> str:
    if max_val == min_val:
        filled = 0
    else:
        filled = int(round((value - min_val) / (max_val - min_val) * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration enumeration
# ──────────────────────────────────────────────────────────────────────────────

def _enumerate_configurations(
    task_models: List[ConditionalSusceptibilityTaskModel],
    max_configs: int = 63,
) -> List[Tuple[str, Tuple[ConditionalSusceptibilityTaskModel, ...]]]:
    all_configs: List[Tuple[str, Tuple[ConditionalSusceptibilityTaskModel, ...]]] = []
    n = len(task_models)
    for r in range(1, n + 1):
        for combo in itertools.combinations(task_models, r):
            all_configs.append((_config_label(combo), combo))
            if len(all_configs) >= max_configs:
                break
        if len(all_configs) >= max_configs:
            break
    # Disambiguate duplicate labels
    seen: Dict[str, int] = {}
    unique: List[Tuple[str, Tuple[ConditionalSusceptibilityTaskModel, ...]]] = []
    for label, tasks in all_configs:
        if label in seen:
            seen[label] += 1
            label = f"{label}_{seen[label]}"
        else:
            seen[label] = 0
        unique.append((label, tasks))
    return unique


# ──────────────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────────────

def _score_config(
    profile_df: pd.DataFrame,
    artifact: ConditionalSusceptibilityArtifact,
    tasks: Tuple[ConditionalSusceptibilityTaskModel, ...],
) -> pd.DataFrame:
    target_attacks = list({t.attack_leaf for t in tasks})
    target_opinions = list({t.opinion_leaf for t in tasks})
    scores, _ = score_profiles_with_conditional_artifact(
        profile_df=profile_df,
        artifact=artifact,
        target_attacks=target_attacks,
        target_opinions=target_opinions,
    )
    return scores


# ──────────────────────────────────────────────────────────────────────────────
# CSV builder — per profile × config × opinion × attack (NO aggregation)
# ──────────────────────────────────────────────────────────────────────────────

def _build_summary_csv(
    configs: List[Tuple[str, Tuple[ConditionalSusceptibilityTaskModel, ...]]],
    all_scores: Dict[str, pd.DataFrame],
    profile_display: Dict[str, Dict[str, Any]],
    observed_tasks: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per profile × config × opinion_leaf × attack_leaf.
    Includes:
      - config metadata (label, attack(s), opinion set)
      - profile metadata (age, sex, Big Five means)
      - per-task predicted effectivity from the model
      - observed effectivity for this specific opinion×attack pair (from the actual run)
      - overall config CSI percentile for this profile
    """
    rows: List[Dict[str, Any]] = []

    for label, tasks in configs:
        scores = all_scores.get(label)
        if scores is None:
            continue

        for task in tasks:
            attack_leaf = task.attack_leaf
            opinion_leaf = task.opinion_leaf
            opinion_label = _leaf_short(opinion_leaf)
            attack_label = _leaf_short(attack_leaf)
            opinion_domain = _leaf_domain(opinion_leaf)

            # Model-predicted effectivity column for this task
            slug = (
                task.task_key.lower()
                .replace(" ", "_").replace(">", "_").replace("|", "_").replace("/", "_")
            )
            pred_col = f"predicted_effectivity__{slug}"

            for _, score_row in scores.iterrows():
                pid = str(score_row["profile_id"])
                pd_row = profile_display.get(pid, {})

                # Observed values for this specific profile × opinion × attack
                if not observed_tasks.empty:
                    obs_match = observed_tasks[
                        (observed_tasks["profile_id"] == pid) &
                        (observed_tasks["opinion_leaf"] == opinion_leaf) &
                        (observed_tasks["attack_leaf"] == attack_leaf)
                    ]
                    obs_ae = float(obs_match["adversarial_effectivity"].iloc[0]) if len(obs_match) > 0 else float("nan")
                    obs_delta = float(obs_match["delta_score"].iloc[0]) if len(obs_match) > 0 else float("nan")
                    obs_abs_delta = float(obs_match["abs_delta_score"].iloc[0]) if len(obs_match) > 0 else float("nan")
                    baseline = float(obs_match["baseline_score"].iloc[0]) if len(obs_match) > 0 else float("nan")
                    post = float(obs_match["post_score"].iloc[0]) if len(obs_match) > 0 else float("nan")
                    adv_dir = int(obs_match["adversarial_direction"].iloc[0]) if len(obs_match) > 0 else 0
                else:
                    obs_ae = obs_delta = obs_abs_delta = baseline = post = float("nan")
                    adv_dir = 0

                r: Dict[str, Any] = {
                    # Config metadata
                    "config_label": label,
                    "config_n_tasks": len(tasks),
                    "config_attack_leaf": attack_leaf,
                    "config_attack_label": attack_label,
                    "config_opinion_leaf": opinion_leaf,
                    "config_opinion_label": opinion_label,
                    "config_opinion_domain": opinion_domain,
                    "adversarial_direction": adv_dir,
                    "task_reliability_weight": task.reliability_weight,
                    # Profile metadata
                    "profile_id": pid,
                    "sex": pd_row.get("sex", ""),
                    "age_years": pd_row.get("age_years", float("nan")),
                }
                # Big Five trait means
                for t_key, t_label in TRAIT_LABELS.items():
                    r[f"bf_{t_label.lower().replace(' ','_')}"] = pd_row.get(t_key, float("nan"))

                # Model scores
                r["predicted_adversarial_effectivity"] = float(score_row.get(pred_col, float("nan")))
                r["config_csi_pct"] = float(score_row.get("susceptibility_index_pct", float("nan")))
                r["config_csi_raw"] = float(score_row.get("conditional_susceptibility_raw_score", float("nan")))

                # Observed values (actual run data)
                r["observed_baseline_score"] = baseline
                r["observed_post_score"] = post
                r["observed_delta_score"] = obs_delta
                r["observed_abs_delta_score"] = obs_abs_delta
                r["observed_adversarial_effectivity"] = obs_ae

                rows.append(r)

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Per-opinion×attack ranking table (no aggregation)
# ──────────────────────────────────────────────────────────────────────────────

def _render_per_task_ranking(
    task: ConditionalSusceptibilityTaskModel,
    scores: pd.DataFrame,
    profile_display: Dict[str, Dict[str, Any]],
    observed_tasks: pd.DataFrame,
    n_top: int = 10,
    label: str = "",
) -> str:
    """Render top/bottom N profiles for one specific attack×opinion task."""
    opinion_short = _leaf_short(task.opinion_leaf)
    attack_short = _leaf_short(task.attack_leaf)
    slug = (
        task.task_key.lower()
        .replace(" ", "_").replace(">", "_").replace("|", "_").replace("/", "_")
    )
    pred_col = f"predicted_effectivity__{slug}"

    scores_sorted = scores.copy()
    if pred_col in scores_sorted.columns:
        scores_sorted = scores_sorted.sort_values(pred_col, ascending=False)
    else:
        scores_sorted = scores_sorted.sort_values("susceptibility_index_pct", ascending=False)

    pred_vals = scores_sorted[pred_col].dropna().tolist() if pred_col in scores_sorted.columns else []
    min_pred = min(pred_vals) if pred_vals else 0.0
    max_pred = max(pred_vals) if pred_vals else 1.0

    lines = [
        "─" * 72,
        f"Opinion: {opinion_short}",
        f"Attack:  {attack_short}",
        f"Adversarial direction: {task.opinion_leaf} → dir={_get_direction_note(task.opinion_leaf)}",
        f"Task reliability weight: {task.reliability_weight:.4f}  |  Config: {label}",
        "",
    ]

    def _profile_line(rank: int, row: pd.Series, top: bool) -> List[str]:
        pid = str(row["profile_id"])
        pred = float(row.get(pred_col, float("nan")))
        csi = float(row.get("susceptibility_index_pct", float("nan")))
        pd_row = profile_display.get(pid, {})
        sex = pd_row.get("sex", "?")
        age = pd_row.get("age_years", float("nan"))
        n_mean = pd_row.get("big_five_neuroticism_mean_pct", float("nan"))
        c_mean = pd_row.get("big_five_conscientiousness_mean_pct", float("nan"))
        a_mean = pd_row.get("big_five_agreeableness_mean_pct", float("nan"))

        obs_row = pd.DataFrame()
        if not observed_tasks.empty:
            obs_row = observed_tasks[
                (observed_tasks["profile_id"] == pid) &
                (observed_tasks["opinion_leaf"] == task.opinion_leaf) &
                (observed_tasks["attack_leaf"] == task.attack_leaf)
            ]
        obs_ae = float(obs_row["adversarial_effectivity"].iloc[0]) if len(obs_row) > 0 else float("nan")
        obs_baseline = float(obs_row["baseline_score"].iloc[0]) if len(obs_row) > 0 else float("nan")
        obs_delta = float(obs_row["delta_score"].iloc[0]) if len(obs_row) > 0 else float("nan")

        bar = _bar(pred, min_pred, max_pred)
        direction_str = "→ toward adversary goal" if pred > 0 else "← away from adversary goal"
        flag = "SUSCEPTIBLE" if top else "RESILIENT"
        result = [
            f"  #{rank:>2} [{flag}]  {pid}",
            f"      predicted_ae={pred:+.2f}  {direction_str}",
            f"      config_csi={csi:.1f}th  |  [{bar}]",
            f"      Profile: age={age:.0f}  sex={sex}  N={n_mean:.0f}  C={c_mean:.0f}  A={a_mean:.0f}",
            f"      Observed: baseline={obs_baseline:.0f}  Δ={obs_delta:+.0f}  obs_ae={obs_ae:+.1f}",
            "",
        ]
        return result

    lines.append(f"  TOP {n_top} MOST SUSCEPTIBLE for this opinion×attack")
    for rank, (_, row) in enumerate(scores_sorted.head(n_top).iterrows(), 1):
        lines.extend(_profile_line(rank, row, top=True))

    lines.append(f"  TOP {n_top} MOST RESILIENT for this opinion×attack")
    for rank, (_, row) in enumerate(scores_sorted.tail(n_top).iloc[::-1].iterrows(), 1):
        lines.extend(_profile_line(rank, row, top=False))

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Config-level report
# ──────────────────────────────────────────────────────────────────────────────

def _get_direction_note(opinion_leaf: str) -> str:
    leaf = _leaf_short(opinion_leaf)
    directions = {
        "Alliance_Commitment_Support": "-1 (adversary wants LOWER)",
        "Civil_Defense_Preparedness_Support": "-1 (adversary wants LOWER)",
        "Domestic_Security_Powers_Expansion_Support": "-1 (adversary wants LOWER)",
        "Strategic_Autonomy_Support": "+1 (adversary wants HIGHER)",
    }
    return directions.get(leaf, "see opinion ontology")


def _render_config_report(
    config_index: int,
    label: str,
    tasks: Tuple[ConditionalSusceptibilityTaskModel, ...],
    scores: pd.DataFrame,
    profile_display: Dict[str, Dict[str, Any]],
    observed_tasks: pd.DataFrame,
    n_top: int = 10,
) -> str:
    lines: List[str] = [
        "=" * 76,
        f"Configuration {config_index:02d}: {_config_title(tasks)}",
        f"Label: {label}  |  Tasks: {len(tasks)}  |  Profiles: {len(scores)}",
        "",
        "Adversarial operator goal: maximize erosion of Western democratic defense",
        "consensus, alliance cohesion, and institutional security capacity.",
        "Positive predicted adversarial effectivity = opinion moves toward adversary's goal.",
        "",
        "Task details:",
    ]
    for t in tasks:
        lines.append(f"  [{t.reliability_weight:.4f}] {_leaf_short(t.attack_leaf)} → {_leaf_short(t.opinion_leaf)}"
                     f"  (dir: {_get_direction_note(t.opinion_leaf)})")
    lines.append("")

    # Overall CSI ranking for this config (all tasks combined)
    raw_vals = scores["conditional_susceptibility_raw_score"].dropna().tolist()
    min_raw = min(raw_vals) if raw_vals else 0.0
    max_raw = max(raw_vals) if raw_vals else 1.0

    lines.append(f"  ── OVERALL CONFIG CSI TOP {n_top} MOST SUSCEPTIBLE ──")
    for rank, (_, row) in enumerate(scores.head(n_top).iterrows(), 1):
        pid = str(row["profile_id"])
        pct = float(row.get("susceptibility_index_pct", 0.0))
        raw = float(row.get("conditional_susceptibility_raw_score", 0.0))
        pd_row = profile_display.get(pid, {})
        sex = pd_row.get("sex", "?")
        age = pd_row.get("age_years", float("nan"))
        n_m = pd_row.get("big_five_neuroticism_mean_pct", float("nan"))
        c_m = pd_row.get("big_five_conscientiousness_mean_pct", float("nan"))
        bar = _bar(raw, min_raw, max_raw)
        lines.append(f"  #{rank:>2} {pid}  CSI={pct:.1f}th  raw={raw:+.3f}  age={age:.0f}  {sex}  N={n_m:.0f}  C={c_m:.0f}  [{bar}]")
    lines.append("")
    lines.append(f"  ── OVERALL CONFIG CSI TOP {n_top} MOST RESILIENT ──")
    for rank, (_, row) in enumerate(scores.tail(n_top).iloc[::-1].iterrows(), 1):
        pid = str(row["profile_id"])
        pct = float(row.get("susceptibility_index_pct", 0.0))
        raw = float(row.get("conditional_susceptibility_raw_score", 0.0))
        pd_row = profile_display.get(pid, {})
        sex = pd_row.get("sex", "?")
        age = pd_row.get("age_years", float("nan"))
        n_m = pd_row.get("big_five_neuroticism_mean_pct", float("nan"))
        c_m = pd_row.get("big_five_conscientiousness_mean_pct", float("nan"))
        bar = _bar(raw, min_raw, max_raw)
        lines.append(f"  #{rank:>2} {pid}  CSI={pct:.1f}th  raw={raw:+.3f}  age={age:.0f}  {sex}  N={n_m:.0f}  C={c_m:.0f}  [{bar}]")
    lines.append("")

    # Per-task breakdowns (no aggregation)
    lines.append("  ── PER OPINION×ATTACK BREAKDOWNS (no aggregation) ──")
    for task in tasks:
        lines.append(_render_per_task_ranking(task, scores, profile_display, observed_tasks, n_top, label))

    lines.append("=" * 76)
    lines.append("")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Full summary report
# ──────────────────────────────────────────────────────────────────────────────

def _render_full_summary(
    configs: List[Tuple[str, Tuple[ConditionalSusceptibilityTaskModel, ...]]],
    all_scores: Dict[str, pd.DataFrame],
    profile_display: Dict[str, Dict[str, Any]],
    observed_tasks: pd.DataFrame,
    run_id: str,
    n_top: int = 10,
) -> str:
    lines: List[str] = [
        "=" * 76,
        f"COMPREHENSIVE SUSCEPTIBILITY RANKINGS — {run_id.upper()}",
        f"All {len(configs)} attack×opinion target-set configurations enumerated.",
        "",
        "Rows are NOT aggregated across opinions. Each profile×opinion×attack",
        "combination is ranked separately for full conditional transparency.",
        "Compatible with multi-attack, multi-opinion future runs.",
        "",
        "Adversarial operator goal: maximize erosion of Western democratic defense",
        "consensus, multilateral alliance cohesion, and institutional security capacity.",
        "=" * 76,
        "",
    ]

    # Index
    lines.append("CONFIGURATIONS INDEX")
    lines.append("─" * 60)
    for ci, (label, tasks) in enumerate(configs, start=1):
        attacks = sorted({_leaf_short(t.attack_leaf) for t in tasks})
        opinions = sorted(_leaf_short(t.opinion_leaf) for t in tasks)
        lines.append(f"  {ci:02d}. [{label}]")
        lines.append(f"      Attacks : {', '.join(attacks)}")
        lines.append(f"      Opinions: {', '.join(opinions)}")
    lines.append("")

    # Cross-config summary
    top_appearances: Dict[str, int] = defaultdict(int)
    bot_appearances: Dict[str, int] = defaultdict(int)
    for label, _ in configs:
        sc = all_scores.get(label)
        if sc is None:
            continue
        for pid in sc.head(n_top)["profile_id"].tolist():
            top_appearances[pid] += 1
        for pid in sc.tail(n_top)["profile_id"].tolist():
            bot_appearances[pid] += 1

    lines.append("CROSS-CONFIGURATION SUMMARY (full config = all tasks combined)")
    lines.append("─" * 60)
    lines.append(f"  Profiles most consistently SUSCEPTIBLE across configs:")
    for pid, cnt in sorted(top_appearances.items(), key=lambda kv: -kv[1])[:n_top]:
        pd_row = profile_display.get(pid, {})
        sex = pd_row.get("sex", "?")
        age = pd_row.get("age_years", float("nan"))
        n_m = pd_row.get("big_five_neuroticism_mean_pct", float("nan"))
        c_m = pd_row.get("big_five_conscientiousness_mean_pct", float("nan"))
        a_m = pd_row.get("big_five_agreeableness_mean_pct", float("nan"))
        lines.append(f"    {pid}: top-{n_top} in {cnt}/{len(configs)} configs  "
                     f"age={age:.0f}  {sex}  N={n_m:.0f}  C={c_m:.0f}  A={a_m:.0f}")
    lines.append("")
    lines.append(f"  Profiles most consistently RESILIENT across configs:")
    for pid, cnt in sorted(bot_appearances.items(), key=lambda kv: -kv[1])[:n_top]:
        pd_row = profile_display.get(pid, {})
        sex = pd_row.get("sex", "?")
        age = pd_row.get("age_years", float("nan"))
        n_m = pd_row.get("big_five_neuroticism_mean_pct", float("nan"))
        c_m = pd_row.get("big_five_conscientiousness_mean_pct", float("nan"))
        a_m = pd_row.get("big_five_agreeableness_mean_pct", float("nan"))
        lines.append(f"    {pid}: bot-{n_top} in {cnt}/{len(configs)} configs  "
                     f"age={age:.0f}  {sex}  N={n_m:.0f}  C={c_m:.0f}  A={a_m:.0f}")
    lines.append("")

    # Per-config detailed blocks
    lines.append("PER-CONFIGURATION DETAILED RANKINGS")
    lines.append("=" * 76)
    lines.append("")
    for ci, (label, tasks) in enumerate(configs, start=1):
        sc = all_scores.get(label)
        if sc is None:
            lines.append(f"[Config {ci:02d}] No scores available.\n")
            continue
        lines.append(_render_config_report(ci, label, tasks, sc, profile_display, observed_tasks, n_top))

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate per-opinion×attack susceptibility rankings across all target-set configurations."
    )
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-top", type=int, default=10)
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    run_id = args.run_id or run_dir.name
    n_top = args.n_top

    full_dir = output_dir / "full"
    sep_dir = output_dir / "01_separated"
    full_dir.mkdir(parents=True, exist_ok=True)
    sep_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading artifact ...")
    artifact_path = _discover_artifact(run_dir)
    artifact = ConditionalSusceptibilityArtifact.model_validate(
        json.loads(artifact_path.read_text(encoding="utf-8"))
    )
    print(f"  {len(artifact.task_models)} tasks, {len(artifact.feature_columns)} features.")

    print("Loading profile display data ...")
    profile_display = _load_profile_display(run_dir)
    print(f"  {len(profile_display)} profiles.")

    print("Loading observed per-task effectivity ...")
    observed_tasks = _load_observed_per_task(run_dir)
    print(f"  {len(observed_tasks)} observed rows.")

    # Build profile feature DataFrame from sem_long
    sem_long_path = _discover_sem_long(run_dir)
    if not sem_long_path:
        raise RuntimeError("sem_long_encoded.csv not found.")
    long_df = pd.read_csv(sem_long_path)
    feature_cols_in_df = [c for c in artifact.feature_columns if c in long_df.columns]
    profile_df = long_df[["profile_id"] + feature_cols_in_df].drop_duplicates(
        subset=["profile_id"]
    ).reset_index(drop=True)
    for col in artifact.feature_columns:
        if col not in profile_df.columns:
            profile_df[col] = artifact.feature_means.get(col, 0.0)

    print(f"  Scoring {len(profile_df)} profiles.")

    # Enumerate and score
    configs = _enumerate_configurations(artifact.task_models, max_configs=63)
    print(f"Enumerated {len(configs)} configurations.")

    all_scores: Dict[str, pd.DataFrame] = {}
    for ci, (label, tasks) in enumerate(configs, start=1):
        print(f"  [{ci:02d}/{len(configs)}] {label}")
        all_scores[label] = _score_config(profile_df, artifact, tasks)

    # Write 01_separated per-config files
    print("Writing 01_separated per-config files ...")
    for ci, (label, tasks) in enumerate(configs, start=1):
        sc = all_scores.get(label)
        if sc is None:
            continue
        report = _render_config_report(ci, label, tasks, sc, profile_display, observed_tasks, n_top)
        (sep_dir / f"config_{ci:02d}_{label}.txt").write_text(report, encoding="utf-8")
    print(f"  {len(configs)} files → {sep_dir}")

    # Write full report
    print("Writing full report ...")
    full_report = _render_full_summary(configs, all_scores, profile_display, observed_tasks, run_id, n_top)
    (full_dir / "susceptibility_rankings_all_configs.txt").write_text(full_report, encoding="utf-8")

    # Write CSV (per profile × config × opinion × attack — no aggregation)
    print("Writing CSV ...")
    csv_df = _build_summary_csv(configs, all_scores, profile_display, observed_tasks)
    csv_path = full_dir / "susceptibility_rankings_all_configs.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"  {len(csv_df)} rows → {csv_path}")

    # Console summary
    full_label = configs[-1][0]
    full_scores = all_scores.get(full_label, pd.DataFrame())
    print()
    print("=" * 70)
    print(f"FULL CONFIG TOP {n_top} SUSCEPTIBLE")
    print("=" * 70)
    if not full_scores.empty:
        for rank, (_, row) in enumerate(full_scores.head(n_top).iterrows(), 1):
            pid = str(row["profile_id"])
            pct = float(row.get("susceptibility_index_pct", 0.0))
            raw = float(row.get("conditional_susceptibility_raw_score", 0.0))
            pd_row = profile_display.get(pid, {})
            sex = pd_row.get("sex", "?")
            age = pd_row.get("age_years", float("nan"))
            n_m = pd_row.get("big_five_neuroticism_mean_pct", float("nan"))
            c_m = pd_row.get("big_five_conscientiousness_mean_pct", float("nan"))
            print(f"  #{rank:>2}  {pid}  CSI={pct:.1f}th  raw={raw:+.3f}  age={age:.0f}  {sex}  N={n_m:.0f}  C={c_m:.0f}")
    print()
    print("=" * 70)
    print(f"FULL CONFIG TOP {n_top} RESILIENT")
    print("=" * 70)
    if not full_scores.empty:
        for rank, (_, row) in enumerate(full_scores.tail(n_top).iloc[::-1].iterrows(), 1):
            pid = str(row["profile_id"])
            pct = float(row.get("susceptibility_index_pct", 0.0))
            raw = float(row.get("conditional_susceptibility_raw_score", 0.0))
            pd_row = profile_display.get(pid, {})
            sex = pd_row.get("sex", "?")
            age = pd_row.get("age_years", float("nan"))
            n_m = pd_row.get("big_five_neuroticism_mean_pct", float("nan"))
            c_m = pd_row.get("big_five_conscientiousness_mean_pct", float("nan"))
            print(f"  #{rank:>2}  {pid}  CSI={pct:.1f}th  raw={raw:+.3f}  age={age:.0f}  {sex}  N={n_m:.0f}  C={c_m:.0f}")

    print(f"\nAll files → {output_dir}")


if __name__ == "__main__":
    main()
