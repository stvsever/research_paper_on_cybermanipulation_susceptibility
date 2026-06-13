from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.backend.utils.io import read_json


OntologyTree = Dict[str, Any]

# Keys that are metadata annotations, not subtree nodes.
_METADATA_KEYS = frozenset({"adversarial_direction", "description", "notes", "examples"})


def _is_metadata_key(key: str) -> bool:
    """Return True for keys that carry leaf-level metadata, not subtree structure."""
    return key.startswith("_") or key in _METADATA_KEYS or not key[0].isupper()


def _is_leaf_node(child: Any) -> bool:
    """Return True if child represents a leaf (empty dict, non-dict, or metadata-only dict)."""
    if not isinstance(child, dict):
        return True
    if not child:
        return True
    # A dict is a leaf if ALL its keys are metadata keys (no uppercase-starting subtree keys)
    return all(_is_metadata_key(k) for k in child)


def default_ontology_root(project_root: Path, use_test_ontology: bool) -> Path:
    mode = "test" if use_test_ontology else "production"
    return project_root / "src" / "backend" / "ontology" / "separate" / mode


def load_ontology_triplet(ontology_root: str | Path) -> Dict[str, OntologyTree]:
    root = Path(ontology_root)
    return {
        "PROFILE": read_json(root / "PROFILE" / "profile.json"),
        "OPINION": read_json(root / "OPINION" / "opinion.json"),
        "ATTACK": read_json(root / "ATTACK" / "attack.json"),
    }


def iter_leaf_paths(tree: OntologyTree, prefix: Tuple[str, ...] = ()) -> List[Tuple[str, ...]]:
    leaves: List[Tuple[str, ...]] = []
    for node, child in tree.items():
        if _is_metadata_key(node):
            continue  # skip _metadata blocks and other annotation keys
        path = prefix + (node,)
        if _is_leaf_node(child):
            leaves.append(path)
        else:
            leaves.extend(iter_leaf_paths(child, path))
    return leaves


def flatten_leaf_paths(tree: OntologyTree) -> List[str]:
    return [" > ".join(path) for path in iter_leaf_paths(tree)]


def get_leaf_metadata(tree: OntologyTree, leaf_path: str) -> Dict[str, Any]:
    """Return the metadata dict stored at a given leaf path (e.g. adversarial_direction)."""
    parts = [p.strip() for p in leaf_path.split(">")]
    node: Any = tree
    for part in parts:
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return {}
    if isinstance(node, dict):
        return {k: v for k, v in node.items() if _is_metadata_key(k)}
    return {}


def load_adversarial_directions_from_opinion(
    opinion_tree: OntologyTree,
) -> Tuple[Dict[str, int], str]:
    """Extract adversarial direction mappings from an opinion ontology tree.

    Resolution order per leaf:
      1. Per-leaf inline ``adversarial_direction`` (overrides everything).
      2. First matching pattern in ``_direction_rules.rules`` (subtree default).
      3. Fallback: ``0`` (neutral).

    Only non-zero directions are returned in the mapping. The leaf-name key
    is the LAST segment of the path, matching the convention used elsewhere
    in the pipeline.
    """
    meta = opinion_tree.get("_metadata", {})
    goal: str = meta.get("adversarial_operator_goal", "")

    rules: List[Dict[str, Any]] = []
    rules_block = opinion_tree.get("_direction_rules")
    if isinstance(rules_block, dict):
        rules = rules_block.get("rules", []) or []

    def _split_path_local(path: str) -> List[str]:
        return [p.strip() for p in path.split(">") if p.strip()]

    def _split_pattern(pattern: str) -> List[str]:
        return [p.strip() for p in pattern.replace("**", " ** ").split(" ** ")]

    def _matches(pattern: str, segments: List[str]) -> bool:
        parts = [p for p in _split_pattern(pattern) if p]
        if not parts:
            return True
        cursor = 0
        for needle in parts:
            sub = _split_path_local(needle)
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

    leaf_paths = flatten_leaf_paths(opinion_tree)
    directions: Dict[str, int] = {}
    for leaf_path in leaf_paths:
        leaf_meta = get_leaf_metadata(opinion_tree, leaf_path)
        direction_raw = leaf_meta.get("adversarial_direction")
        try:
            direction = int(direction_raw) if direction_raw is not None else None
        except (TypeError, ValueError):
            direction = None
        if direction in (None, 0):
            segments = _split_path_local(leaf_path)
            for rule in rules:
                patterns = rule.get("applies_to_opinion_paths", []) or []
                if not isinstance(patterns, list):
                    continue
                if any(_matches(str(p), segments) for p in patterns):
                    try:
                        rule_dir = int(rule.get("default_direction", 0) or 0)
                    except (TypeError, ValueError):
                        rule_dir = 0
                    if direction is None or rule_dir != 0:
                        direction = rule_dir
                    break
        if direction and direction != 0:
            leaf_name = leaf_path.split(">")[-1].strip()
            directions[leaf_name] = int(direction)

    return directions, goal


def leaf_to_key(path: str) -> str:
    return path.lower().replace(" ", "").replace(">", "_").replace("-", "_")


def find_primary_node(path: str) -> str:
    parts = [x.strip() for x in path.split(">")]
    if len(parts) >= 2:
        return parts[1]
    return parts[0]
