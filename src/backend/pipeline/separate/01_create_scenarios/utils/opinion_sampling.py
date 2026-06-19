from __future__ import annotations

"""
Opinion-target sampling engine
==============================
Samples a balanced panel of *opinion targets* from the production OPINION
ontology for cyber-manipulation simulation.

Design
------
* Only **directionally-encoded** leaves (``adversarial_direction != 0``) are
  eligible — these are the constructs a general adversary would actually try to
  shift (erode trust/norms/efficacy; amplify polarization/conspiracy/threat/…).
  Direction-neutral leaves (policy-preference diversity) are excluded.
* Leaves are **clustered by their parent node** (the psychometric construct /
  scale they belong to).
* Allocation is **two-layer maximal entropy** (hierarchical uniform):
  the N draws are split equally across parent clusters, then equally across the
  leaves within each cluster — so P(leaf) = (1/#clusters) · (1/#leaves_in_cluster).
  Both layers are uniform, which is the maximal-entropy hierarchical design.
"""

import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[6]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.backend.utils.ontology_utils import flatten_leaf_paths, get_leaf_metadata  # noqa: E402


@dataclass
class OpinionLeaf:
    path: str
    family: str        # top-level opinion family
    parent_path: str   # cluster key (immediate parent path)
    parent_name: str   # cluster label
    leaf_name: str
    direction: int     # adversarial_direction ∈ {-1, +1}


def load_directional_leaves(opinion_tree: Dict[str, Any]) -> List[OpinionLeaf]:
    """All opinion leaves whose baked adversarial_direction is non-zero."""
    out: List[OpinionLeaf] = []
    for lp in flatten_leaf_paths(opinion_tree):
        meta = get_leaf_metadata(opinion_tree, lp)
        try:
            direction = int(meta.get("adversarial_direction", 0) or 0)
        except (TypeError, ValueError):
            direction = 0
        if direction == 0:
            continue
        segs = [s.strip() for s in lp.split(">")]
        parent_path = " > ".join(segs[:-1])
        out.append(OpinionLeaf(
            path=lp,
            family=segs[0],
            parent_path=parent_path,
            parent_name=segs[-2] if len(segs) >= 2 else segs[0],
            leaf_name=segs[-1],
            direction=direction,
        ))
    return out


def cluster_by_parent(leaves: List[OpinionLeaf]) -> Dict[str, List[OpinionLeaf]]:
    clusters: Dict[str, List[OpinionLeaf]] = defaultdict(list)
    for lf in leaves:
        clusters[lf.parent_path].append(lf)
    return dict(sorted(clusters.items()))


def _even_split(total: int, k: int, rng) -> List[int]:
    """Split `total` into k near-equal integer parts (max entropy), order shuffled."""
    if k <= 0:
        return []
    base, rem = divmod(total, k)
    parts = [base + (1 if i < rem else 0) for i in range(k)]
    rng.shuffle(parts)
    return parts


def normalised_entropy(counts: List[int]) -> float:
    total = sum(counts)
    k = len(counts)
    if total == 0 or k <= 1:
        return 1.0 if k <= 1 else 0.0
    h = -sum((c / total) * math.log(c / total) for c in counts if c > 0)
    return h / math.log(k)


def sample_opinion_panel(
    opinion_tree: Dict[str, Any], n: int, seed: int = 42
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Two-layer max-entropy opinion-target panel of size `n`, clustered by parent."""
    import random
    rng = random.Random(seed)

    leaves = load_directional_leaves(opinion_tree)
    clusters = cluster_by_parent(leaves)
    cluster_keys = list(clusters.keys())

    # Layer 1: equal allocation across clusters.
    cluster_alloc = dict(zip(cluster_keys, _even_split(n, len(cluster_keys), rng)))

    # Layer 2: equal allocation across leaves within each cluster.
    out_clusters: Dict[str, Any] = {}
    direction_counts: Counter = Counter()
    leaf_counts_for_entropy: List[int] = []
    total_assigned = 0
    for ckey in cluster_keys:
        members = clusters[ckey]
        per_leaf = _even_split(cluster_alloc[ckey], len(members), rng)
        leaf_entries = []
        for lf, cnt in zip(members, per_leaf):
            if cnt > 0:
                direction_counts[lf.direction] += cnt
            leaf_counts_for_entropy.append(cnt)
            total_assigned += cnt
            leaf_entries.append({
                "leaf": lf.leaf_name,
                "path": lf.path,
                "adversarial_direction": lf.direction,
                "count": cnt,
            })
        leaf_entries.sort(key=lambda e: (-e["count"], e["leaf"]))
        out_clusters[ckey] = {
            "parent_name": members[0].parent_name,
            "family": members[0].family,
            "n_leaves": len(members),
            "n_sampled": cluster_alloc[ckey],
            "leaves": leaf_entries,
        }

    cluster_counts = [out_clusters[c]["n_sampled"] for c in cluster_keys]
    summary = {
        "n_target": n,
        "n_sampled_total": total_assigned,
        "n_directional_leaves": len(leaves),
        "n_clusters": len(cluster_keys),
        "n_families": len({lf.family for lf in leaves}),
        "two_layer_max_entropy": {
            "cluster_layer_normalised_entropy": round(normalised_entropy(cluster_counts), 6),
            "leaf_layer_normalised_entropy": round(normalised_entropy(leaf_counts_for_entropy), 6),
        },
        "direction_balance": {str(k): v for k, v in sorted(direction_counts.items())},
        "leaves_per_cluster": {"min": min(len(v) for v in clusters.values()),
                               "max": max(len(v) for v in clusters.values())},
        "family_leaf_counts": dict(Counter(lf.family for lf in leaves)),
    }
    bulk = {"clusters": out_clusters}
    return bulk, summary
