from __future__ import annotations

"""
CLI — high-resolution, maximal-entropy PROFILE sampling
=======================================================

Samples a population of high-resolution pseudoprofiles directly from the
production PROFILE ontology's per-leaf STRUCTURAL sampling metadata and writes
them into the scenario-stage ``samples/01_separated/profiles`` directory.

What it produces (in ``--out-dir``)
-----------------------------------
  production_profiles_maxent_<N>.json        full population (compact columnar)
  production_profiles_examples_3.json        3 fully-expanded example profiles
  production_profiles_maxent_<N>.summary.json plan + entropy + coverage report

Why this exists
---------------
The pipeline's per-run sampler (utils/profile_sampling.py) re-derives variable
types from keyword heuristics and only touches a handful of constructs.  This
CLI instead reads the ontology's baked metadata (modality_type, sampling_role,
exclusivity_group, prevalence_weight, is_unknown_marker, …) and samples EVERY
leaf according to its declared nature — categorical options by mutually-
exclusive choice, continuous facets as full-range percentiles, ordinal items as
Likert levels, age/dates as coherent scalars — while forcing the basic
demographics (sex, age, gender) to a near-uniform (maximal-entropy) marginal.

Typical use
-----------
Run by file path (the stage directory name starts with a digit, so ``-m`` does
not apply); the script inserts the project root on sys.path itself:

  python src/backend/pipeline/separate/01_create_scenarios/sample_high_res_profiles.py \\
      --n 10000

  # explicit knobs
  python src/backend/pipeline/separate/01_create_scenarios/sample_high_res_profiles.py \\
      --n 10000 --seed 42 --age-min 0 --age-max 95 --ordinal-levels 5 \\
      --balance-categoricals Sex_Assigned_At_Birth,Gender_Identity
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[6]
for _p in (str(PROJECT_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from production_profile_sampling import (  # noqa: E402  (local sibling module)
    SamplingConfig,
    build_variable_plan,
    encode_population_columnar,
    MaxEntropyPopulationSampler,
    profile_to_record,
)

DEFAULT_ONTOLOGY = (
    PROJECT_ROOT
    / "src" / "backend" / "ontology" / "separate" / "production" / "PROFILE" / "profile.json"
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "src" / "backend" / "pipeline" / "separate" / "01_create_scenarios"
    / "samples" / "01_separated" / "profiles"
)


# ─────────────────────────────────────────────────────────────────────────────
# Subtree pruning (skip whole ontology branches before the plan is built)
# ─────────────────────────────────────────────────────────────────────────────

def prune_subtrees(node: Any, names: set) -> int:
    """Recursively delete any BRANCH child whose key is in ``names``.

    Pruning happens before build_variable_plan, so every excluded leaf is simply
    absent from the variable plan and is never sampled. All other sampling logic
    (coherence, caps, entropy) is unchanged; it just operates on a smaller tree.
    """
    if not isinstance(node, dict):
        return 0
    removed = 0
    for k in list(node.keys()):
        if k and k[0].isupper():  # branch key (metadata keys are lowercase)
            if k in names:
                del node[k]
                removed += 1
            else:
                removed += prune_subtrees(node[k], names)
    return removed


# ─────────────────────────────────────────────────────────────────────────────
# Coverage diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def _coverage_report(plan, profiles) -> Dict[str, Any]:
    """How much of each variable's value range the population actually visits."""
    # Categorical option coverage (fraction of option leaves visited at least once).
    visited: Dict[str, set] = {c.key: set() for c in plan.categoricals}
    for p in profiles:
        for k, v in p.categorical.items():
            visited[k].add(v)
    cat_cov = [len(visited[c.key]) / max(1, len(c.options)) for c in plan.categoricals]

    # Continuous spread: mean min/max actually reached across percentile columns.
    cont_min: Dict[str, float] = {}
    cont_max: Dict[str, float] = {}
    for p in profiles:
        for k, v in p.continuous.items():
            if k.endswith("_mean_pct"):
                continue
            cont_min[k] = min(cont_min.get(k, 100.0), v)
            cont_max[k] = max(cont_max.get(k, 0.0), v)
    spreads = [cont_max[k] - cont_min[k] for k in cont_max]

    ages = [p.demographics["age_years"] for p in profiles]
    return {
        "categorical_option_coverage_mean": round(sum(cat_cov) / max(1, len(cat_cov)), 4),
        "categorical_variables_fully_covered": int(sum(1 for c in cat_cov if c >= 0.999)),
        "categorical_variable_count": len(cat_cov),
        "continuous_mean_range_visited_pct": round(sum(spreads) / max(1, len(spreads)), 2),
        "age_min": min(ages),
        "age_max": max(ages),
        "age_distinct_values": len(set(ages)),
        "distinct_sex_values": len(set(p.demographics.get("sex_assigned_at_birth") for p in profiles)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Sample high-resolution, maximal-entropy profiles from the production PROFILE ontology",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--n", type=int, default=10_000, help="Number of profile configurations to sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY), help="Path to production profile.json")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory for output JSON files")
    ap.add_argument("--n-examples", type=int, default=3, help="How many fully-expanded examples to write")

    # Max-entropy fundamentals: chronological age + Big Five personality are
    # spread by stratified linear-spaced anchors (uniform, full-range, no bands).
    ap.add_argument("--balance-scale-markers", default="Big_Five",
                    help="Comma-separated path markers of continuous scales spread to max entropy")
    ap.add_argument("--entropy-threshold", type=float, default=0.99)

    ap.add_argument("--age-min", type=int, default=16)
    ap.add_argument("--age-max", type=int, default=80)
    ap.add_argument("--reference-year", type=int, default=2026)

    # Subtrees to drop entirely before planning. Issue_Position_Taxonomy (223
    # issue-position leaves under Political Profile) dominates the state space and
    # is now the OPINION target space, so it is excluded from the profile here.
    ap.add_argument("--exclude-subtrees", default="Issue_Position_Taxonomy",
                    help="Comma-separated ontology branch node names to prune before sampling")

    # Coherence: forward-checking constraint satisfaction from _coherence_rules.
    ap.add_argument("--no-coherence", action="store_true",
                    help="Disable ontology _coherence_rules (allows impossible combos)")
    ap.add_argument("--facet-jitter", type=float, default=9.0,
                    help="Within-scale facet spread (rejection-bounded; no 0/100 pile-up)")

    ap.add_argument("--encoding", choices=["index", "labels"], default="index",
                    help="Columnar categorical encoding for the bulk file")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print the bulk file (much larger)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ontology_path = Path(args.ontology).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== High-resolution max-entropy profile sampling ===")
    print(f"  ontology : {ontology_path}")
    print(f"  out dir  : {out_dir}")
    print(f"  n        : {args.n:,}   seed: {args.seed}")

    with open(ontology_path) as fh:
        ontology = json.load(fh)
    person_tree = ontology["PERSON"]

    exclude = {t.strip() for t in args.exclude_subtrees.split(",") if t.strip()}
    n_pruned = prune_subtrees(person_tree, exclude) if exclude else 0
    if exclude:
        print(f"  excluded   : {sorted(exclude)}  ({n_pruned} subtree node(s) pruned)")

    cfg = SamplingConfig(
        n_profiles=args.n,
        seed=args.seed,
        balance_scale_markers=tuple(t.strip() for t in args.balance_scale_markers.split(",") if t.strip()),
        entropy_threshold=args.entropy_threshold,
        age_min=args.age_min,
        age_max=args.age_max,
        reference_year=args.reference_year,
        coherence=not args.no_coherence,
        facet_jitter=args.facet_jitter,
    )

    plan = build_variable_plan(person_tree)
    print("\n  variable plan:")
    for k, v in plan.summary().items():
        print(f"    {k:30}: {v}")

    print("\n  sampling (forward-checking constraint satisfaction; lin-space age + Big Five)...")
    sampler = MaxEntropyPopulationSampler(
        plan, cfg,
        coherence_rules=ontology.get("_coherence_rules"),
        population_priors=ontology.get("_population_priors"),
    )
    profiles, entropy_report = sampler.sample_population()
    print(f"  -> {len(profiles):,} profiles; max_entropy_reached={entropy_report['max_entropy_reached']}")
    print("     MAX-ENTROPY (continuous, full-range): age + Big Five")
    for tgt, h in entropy_report["normalised_entropy_per_max_entropy_target"].items():
        print(f"       norm. entropy[{tgt}] = {h}")
    print(f"     sex marginal      = {entropy_report['sex_marginal']}")
    print(f"     gender modality   = {entropy_report['gender_modality_marginal']}")
    n_omit = len(entropy_report.get("applicability_omissions", {}))
    print(f"     applicability omissions (N/A-gated vars) = {n_omit}")

    coverage = _coverage_report(plan, profiles)

    # ── Write bulk population (compact columnar) ─────────────────────────────
    encoded = encode_population_columnar(plan, profiles, encoding=args.encoding)
    meta = {
        "generator": "sample_high_res_profiles.py",
        "ontology_source": str(ontology_path),
        "ontology_schema_version": ontology.get("_metadata", {}).get("schema_version"),
        "n_profiles": len(profiles),
        "seed": args.seed,
        "excluded_subtrees": sorted(exclude),
        "config": cfg.__dict__,
        "plan_summary": plan.summary(),
        "entropy_report": entropy_report,
        "coverage_report": coverage,
        "leaves_resolved_per_profile": profiles[0].n_resolved_leaves() if profiles else 0,
    }
    # _meta / _format / schema first, the big profiles array last.
    bulk = {"_meta": meta, **encoded}
    bulk_path = out_dir / f"production_profiles_maxent_{len(profiles)}.json"
    with open(bulk_path, "w") as fh:
        if args.pretty:
            json.dump(bulk, fh, indent=2)
        else:
            json.dump(bulk, fh, separators=(",", ":"))

    # ── Write expanded examples ──────────────────────────────────────────────
    n_ex = max(1, min(args.n_examples, len(profiles)))
    examples = [profile_to_record(profiles[i], include_leaves=True) for i in range(n_ex)]
    examples_path = out_dir / f"production_profiles_examples_{n_ex}.json"
    with open(examples_path, "w") as fh:
        json.dump(
            {
                "_meta": {
                    "ontology_source": str(ontology_path),
                    "note": "Fully-expanded examples; the full population is in the columnar file.",
                },
                "examples": examples,
            },
            fh,
            indent=2,
        )

    # ── Write summary / diagnostics ──────────────────────────────────────────
    summary_path = out_dir / f"production_profiles_maxent_{len(profiles)}.summary.json"
    with open(summary_path, "w") as fh:
        json.dump(
            {
                "ontology_source": str(ontology_path),
                "n_profiles": len(profiles),
                "config": cfg.__dict__,
                "plan_summary": plan.summary(),
                "entropy_report": entropy_report,
                "coverage_report": coverage,
            },
            fh,
            indent=2,
        )

    # ── Console report ───────────────────────────────────────────────────────
    bulk_mb = bulk_path.stat().st_size / 1e6
    print(f"\n  coverage:")
    for k, v in coverage.items():
        print(f"    {k:38}: {v}")
    print(f"\n  written:")
    print(f"    {bulk_path.name}  ({bulk_mb:.1f} MB, {args.encoding} encoding)")
    print(f"    {examples_path.name}")
    print(f"    {summary_path.name}")

    # Preview the examples' headline demographics.
    print(f"\n  example profiles (headline):")
    for ex in examples:
        dem = ex["demographics"]
        print(
            f"    {ex['profile_id']}: age={dem.get('age_years')} "
            f"sex={dem.get('sex_assigned_at_birth')} gender={dem.get('gender_identity')} "
            f"rel={dem.get('relationship_status')} | {len(ex['categorical_attributes'])} cat vars"
        )
    print("\n=== Done ===\n")


if __name__ == "__main__":
    main()
