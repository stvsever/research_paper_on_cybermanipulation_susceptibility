from __future__ import annotations

"""
CLI — two-layer maximal-entropy OPINION-target sampling
=======================================================
Samples N opinion targets from the production OPINION ontology, restricted to
directionally-encoded leaves (``adversarial_direction != 0``), clustered by
parent node, with two-layer (cluster + within-cluster) maximal-entropy
allocation.  Writes three JSON files into the opinions samples directory.

Run by file path (stage dir name starts with a digit):
  python src/backend/pipeline/separate/01_create_scenarios/utils/sample_high_res_opinions.py --n 1000
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[6]
for _p in (str(PROJECT_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from opinion_sampling import sample_opinion_panel  # noqa: E402  (local sibling)

DEFAULT_ONTOLOGY = (
    PROJECT_ROOT / "src" / "backend" / "ontology" / "separate" / "production" / "OPINION" / "opinion.json"
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT / "src" / "backend" / "pipeline" / "separate" / "01_create_scenarios"
    / "samples" / "01_separated" / "opinions"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Sample maximal-entropy opinion targets clustered by parent node",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--n", type=int, default=1000, help="Number of opinion targets to sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--n-example-clusters", type=int, default=3)
    # Restrict the opinion target space to one top-level subtree. Opinion targets
    # are now the issue positions (the stances an adversary tries to shift), so we
    # sample only from Issue_Position_Taxonomy and cluster by its issue domains.
    ap.add_argument("--root-subtree", default="Issue_Position_Taxonomy",
                    help="Top-level opinion family to restrict sampling to (empty = whole ontology)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ontology_path = Path(args.ontology).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(ontology_path) as fh:
        opinion_tree = json.load(fh)

    # Restrict the sampling source to one top-level subtree, keeping the metadata
    # keys so the convention/schema are still recorded. Leaf paths (hence family,
    # parent cluster) are derived from this restricted tree by the sampler.
    if args.root_subtree:
        if args.root_subtree not in opinion_tree:
            raise SystemExit(f"--root-subtree {args.root_subtree!r} not in ontology top level")
        restricted = {args.root_subtree: opinion_tree[args.root_subtree]}
        for mk in ("_metadata", "_direction_rules"):
            if mk in opinion_tree:
                restricted[mk] = opinion_tree[mk]
        opinion_tree = restricted

    print(f"\n=== Opinion-target sampling (2-layer max entropy) ===")
    print(f"  ontology : {ontology_path}")
    print(f"  subtree  : {args.root_subtree or '(whole ontology)'}")
    print(f"  out dir  : {out_dir}")
    print(f"  n        : {args.n:,}   seed: {args.seed}")

    bulk, summary = sample_opinion_panel(opinion_tree, n=args.n, seed=args.seed)
    bulk["_meta"] = {
        "generator": "sample_high_res_opinions.py",
        "ontology_source": str(ontology_path),
        "ontology_schema_version": opinion_tree.get("_metadata", {}).get("schema_version"),
        "root_subtree": args.root_subtree or None,
        "seed": args.seed,
        "convention": opinion_tree.get("_direction_rules", {}).get("convention"),
        "summary": summary,
    }
    # reorder: _meta first
    bulk = {"_meta": bulk.pop("_meta"), **bulk}

    n_real = summary["n_sampled_total"]
    bulk_path = out_dir / f"opinion_targets_maxent_{n_real}.json"
    summary_path = out_dir / f"opinion_targets_maxent_{n_real}.summary.json"
    examples_path = out_dir / "opinion_clusters_examples_3.json"

    with open(bulk_path, "w") as fh:
        json.dump(bulk, fh, indent=2, ensure_ascii=False)
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    # 3 example clusters (largest allocations, fully expanded).
    clusters = bulk["clusters"]
    top = sorted(clusters.items(), key=lambda kv: -kv[1]["n_sampled"])[: max(1, args.n_example_clusters)]
    with open(examples_path, "w") as fh:
        json.dump({
            "_meta": {"note": "Three example parent clusters from the sampled opinion panel."},
            "example_clusters": {k: v for k, v in top},
        }, fh, indent=2, ensure_ascii=False)

    print(f"\n  directional leaves (non-zero) : {summary['n_directional_leaves']}")
    print(f"  parent clusters               : {summary['n_clusters']}  across {summary['n_families']} families")
    print(f"  two-layer entropy             : cluster={summary['two_layer_max_entropy']['cluster_layer_normalised_entropy']}"
          f"  leaf={summary['two_layer_max_entropy']['leaf_layer_normalised_entropy']}")
    print(f"  direction balance (-1/+1)     : {summary['direction_balance']}")
    print(f"\n  written:")
    for p in (bulk_path, summary_path, examples_path):
        print(f"    {p.name}")
    print("\n  example clusters:")
    for k, v in top:
        leaves = ", ".join(f"{e['leaf']}({e['adversarial_direction']:+d})" for e in v["leaves"][:4])
        print(f"    {v['family']} > {v['parent_name']}  [{v['n_sampled']} draws / {v['n_leaves']} leaves]: {leaves} …")
    print("\n=== Done ===\n")


if __name__ == "__main__":
    main()
