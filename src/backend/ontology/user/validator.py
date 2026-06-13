"""
Ontology validator for user-provided PROFILE × ATTACK × OPINION JSON files.

Design goals
------------
- Zero false positives on valid ontologies: all existing test ontologies must
  pass without warnings.
- Actionable error messages: each issue references the exact JSON path so the
  user can fix it immediately.
- Graceful degradation: warnings do not block execution; errors do.
- Generic: no hardcoded inventory names (Big Five, HEXACO, etc.).

Validation layers
-----------------
1. Structural: valid JSON, non-empty, correct root-key naming
2. Hierarchy: at least one leaf discoverable, depth ≥ 2 for PROFILE/OPINION
3. Metadata: OPINION leaves should carry adversarial_direction (±1 or 0)
4. Compatibility: PROFILE leaves yield ≥ 1 continuous or ≥ 1 categorical column
   when processed by profile_sampling._detect_column_type
5. Cross-ontology: at least one ATTACK leaf and one OPINION leaf exist (minimum
   viable simulation requires both)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


OntologyTree = Dict[str, Any]

# ── Metadata key detection (mirrors ontology_utils) ──────────────────────────
_METADATA_KEYS = frozenset({"adversarial_direction", "description", "notes", "examples",
                              "mechanism", "primary_system", "platform_hint"})


def _is_metadata_key(key: str) -> bool:
    return key.startswith("_") or key in _METADATA_KEYS or not key[0].isupper()


def _is_leaf_node(child: Any) -> bool:
    if not isinstance(child, dict):
        return True
    if not child:
        return True
    return all(_is_metadata_key(k) for k in child)


def _iter_leaves(tree: OntologyTree, prefix: Tuple[str, ...] = ()) -> List[Tuple[Tuple[str, ...], Any]]:
    """Yield (path_tuple, leaf_value) for every leaf in the tree."""
    results = []
    for key, child in tree.items():
        if _is_metadata_key(key):
            continue
        path = prefix + (key,)
        if _is_leaf_node(child):
            results.append((path, child))
        else:
            results.extend(_iter_leaves(child, path))
    return results


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    severity: str   # "error" | "warning" | "info"
    ontology: str   # "PROFILE" | "ATTACK" | "OPINION" | "cross"
    path: str       # JSON path string, e.g. "ATTACK_VECTORS > Deepfake > ..."
    message: str


@dataclass
class ValidationReport:
    profile_path: str
    attack_path: str
    opinion_path: str
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = [
            f"Validation report",
            f"  PROFILE : {self.profile_path}",
            f"  ATTACK  : {self.attack_path}",
            f"  OPINION : {self.opinion_path}",
            f"  Status  : {'PASS' if self.is_valid else 'FAIL'}",
            f"  Errors  : {len(self.errors)}",
            f"  Warnings: {len(self.warnings)}",
        ]
        for issue in self.issues:
            tag = f"[{issue.severity.upper()}]"
            lines.append(f"  {tag} [{issue.ontology}] {issue.path} — {issue.message}")
        return "\n".join(lines)

    def raise_if_invalid(self) -> None:
        if not self.is_valid:
            raise ValueError(
                f"Ontology validation failed with {len(self.errors)} error(s).\n"
                + self.summary()
            )


# ─────────────────────────────────────────────────────────────────────────────
# Per-ontology validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_json_safe(path: Path) -> Tuple[Optional[OntologyTree], Optional[str]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return None, f"Root element must be a JSON object (dict), got {type(data).__name__}"
        if not data:
            return None, "Root element is an empty dict — no ontology content found"
        return data, None
    except FileNotFoundError:
        return None, f"File not found: {path}"
    except json.JSONDecodeError as exc:
        return None, f"JSON parse error: {exc}"


def _validate_profile(tree: OntologyTree, issues: List[ValidationIssue]) -> None:
    leaves = _iter_leaves(tree)
    if not leaves:
        issues.append(ValidationIssue("error", "PROFILE", "(root)", "No leaves found in PROFILE ontology"))
        return
    if len(leaves) < 5:
        issues.append(ValidationIssue(
            "warning", "PROFILE", "(root)",
            f"Only {len(leaves)} leaf nodes found. At least 5 are recommended for meaningful susceptibility modelling."
        ))

    # Check depth ≥ 2 for structural validity
    shallow = [p for p, _ in leaves if len(p) < 2]
    if shallow:
        issues.append(ValidationIssue(
            "warning", "PROFILE", str(shallow[0]),
            "Some PROFILE leaves are at depth < 2. Typical inventories have ≥ 2 hierarchy levels."
        ))

    # Detect whether any leaves would produce continuous or categorical columns
    has_continuous = any(
        _profile_leaf_type(p) in ("continuous", "absolute") for p, _ in leaves
    )
    has_categorical = any(
        _profile_leaf_type(p) == "categorical" for p, _ in leaves
    )
    if not has_continuous and not has_categorical:
        issues.append(ValidationIssue(
            "error", "PROFILE", "(root)",
            "Could not infer any typed columns (continuous or categorical) from PROFILE leaves. "
            "Ensure leaf names follow conventions: CamelCase for subtrees, lowercase/special for leaf labels."
        ))


_CATEGORICAL_MARKERS = frozenset({
    "sex", "gender", "race", "ethnicity", "education", "religion",
    "nationality", "party", "marital", "category", "type", "group",
    "orientation", "level",
})
_ABSOLUTE_MARKERS = frozenset({"age", "years", "income", "bmi", "weight", "height"})


def _profile_leaf_type(path: Tuple[str, ...]) -> str:
    """Infer leaf column type from path tokens (mirrors profile_sampling logic)."""
    tokens = {t.lower() for part in path for t in part.split("_")}
    if tokens & _CATEGORICAL_MARKERS:
        return "categorical"
    if tokens & _ABSOLUTE_MARKERS:
        return "absolute"
    return "continuous"


def _validate_attack(tree: OntologyTree, issues: List[ValidationIssue]) -> None:
    leaves = _iter_leaves(tree)
    if not leaves:
        issues.append(ValidationIssue("error", "ATTACK", "(root)", "No leaves found in ATTACK ontology"))
        return
    if len(leaves) < 2:
        issues.append(ValidationIssue(
            "warning", "ATTACK", "(root)",
            f"Only {len(leaves)} attack leaf. At least 2 attack variants are recommended for comparative analysis."
        ))

    for path, leaf_val in leaves:
        path_str = " > ".join(path)
        if isinstance(leaf_val, dict):
            # Check for recommended metadata
            if "mechanism" not in leaf_val and "description" not in leaf_val:
                issues.append(ValidationIssue(
                    "info", "ATTACK", path_str,
                    "Leaf has no 'mechanism' or 'description' metadata. Adding these improves simulation realism."
                ))


def _validate_opinion(tree: OntologyTree, issues: List[ValidationIssue]) -> None:
    leaves = _iter_leaves(tree)
    if not leaves:
        issues.append(ValidationIssue("error", "OPINION", "(root)", "No leaves found in OPINION ontology"))
        return
    if len(leaves) < 4:
        issues.append(ValidationIssue(
            "warning", "OPINION", "(root)",
            f"Only {len(leaves)} opinion leaves. At least 4 are recommended for meaningful state space coverage."
        ))

    missing_direction = []
    invalid_direction = []
    for path, leaf_val in leaves:
        path_str = " > ".join(path)
        if not isinstance(leaf_val, dict):
            missing_direction.append(path_str)
            continue
        ad = leaf_val.get("adversarial_direction")
        if ad is None:
            missing_direction.append(path_str)
        elif ad not in (-1, 0, 1):
            invalid_direction.append(f"{path_str} (got {ad!r})")

    if missing_direction:
        sample = missing_direction[:3]
        issues.append(ValidationIssue(
            "warning", "OPINION", ", ".join(sample) + ("..." if len(missing_direction) > 3 else ""),
            f"{len(missing_direction)} of {len(leaves)} OPINION leaves lack 'adversarial_direction'. "
            "Direction-aware attack simulation will fall back to direction-blind mode for these leaves."
        ))
    if invalid_direction:
        issues.append(ValidationIssue(
            "error", "OPINION", invalid_direction[0],
            f"adversarial_direction must be -1, 0, or +1. Got invalid values at {len(invalid_direction)} leaf(ves)."
        ))

    # Depth check
    shallow = [p for p, _ in leaves if len(p) < 2]
    if shallow:
        issues.append(ValidationIssue(
            "warning", "OPINION", str(shallow[0]),
            "Some OPINION leaves are at depth < 2. At least 2 levels (domain > issue) are recommended."
        ))


def _validate_cross(
    profile_tree: OntologyTree,
    attack_tree: OntologyTree,
    opinion_tree: OntologyTree,
    issues: List[ValidationIssue],
) -> None:
    attack_leaves = _iter_leaves(attack_tree)
    opinion_leaves = _iter_leaves(opinion_tree)
    profile_leaves = _iter_leaves(profile_tree)

    total_tasks = len(attack_leaves) * len(opinion_leaves)
    total_profiles = max(1, len(profile_leaves))

    if total_tasks == 0:
        issues.append(ValidationIssue(
            "error", "cross", "(global)",
            "Zero attack × opinion task combinations. At least one attack and one opinion leaf required."
        ))
        return

    # Minimum viable simulation: recommend ≥ 20 profiles for meaningful statistics
    if total_profiles < 15:
        issues.append(ValidationIssue(
            "warning", "cross", "(global)",
            f"PROFILE ontology yields approximately {total_profiles} dimensional leaves. "
            "Ridge regression reliability improves significantly with ≥ 20 unique profile features."
        ))

    issues.append(ValidationIssue(
        "info", "cross", "(global)",
        f"Ontology triplet defines {len(attack_leaves)} attack × {len(opinion_leaves)} opinion = "
        f"{total_tasks} simulation task(s) with ~{len(profile_leaves)} PROFILE feature dimensions."
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def validate_ontology_triplet(
    profile_path: str | Path,
    attack_path: str | Path,
    opinion_path: str | Path,
) -> ValidationReport:
    """Validate a user-provided ontology triplet (PROFILE, ATTACK, OPINION).

    Parameters
    ----------
    profile_path : path to PROFILE JSON ontology file
    attack_path  : path to ATTACK JSON ontology file
    opinion_path : path to OPINION JSON ontology file

    Returns
    -------
    ValidationReport — call .raise_if_invalid() to abort on errors,
    or .summary() to print a human-readable report.
    """
    profile_path = Path(profile_path)
    attack_path = Path(attack_path)
    opinion_path = Path(opinion_path)

    report = ValidationReport(
        profile_path=str(profile_path),
        attack_path=str(attack_path),
        opinion_path=str(opinion_path),
    )
    issues = report.issues

    # Load all three; record errors for missing/invalid files
    profile_tree, err = _load_json_safe(profile_path)
    if err:
        issues.append(ValidationIssue("error", "PROFILE", str(profile_path), err))

    attack_tree, err = _load_json_safe(attack_path)
    if err:
        issues.append(ValidationIssue("error", "ATTACK", str(attack_path), err))

    opinion_tree, err = _load_json_safe(opinion_path)
    if err:
        issues.append(ValidationIssue("error", "OPINION", str(opinion_path), err))

    if profile_tree is not None:
        _validate_profile(profile_tree, issues)
    if attack_tree is not None:
        _validate_attack(attack_tree, issues)
    if opinion_tree is not None:
        _validate_opinion(opinion_tree, issues)

    if profile_tree is not None and attack_tree is not None and opinion_tree is not None:
        _validate_cross(profile_tree, attack_tree, opinion_tree, issues)

    return report


def load_user_ontology_triplet(
    profile_path: str | Path,
    attack_path: str | Path,
    opinion_path: str | Path,
    validate: bool = True,
) -> Dict[str, OntologyTree]:
    """Load and optionally validate a user-provided ontology triplet.

    Parameters
    ----------
    profile_path, attack_path, opinion_path : paths to JSON files
    validate : if True (default), run validation and raise on errors

    Returns
    -------
    Dict with keys "PROFILE", "ATTACK", "OPINION"
    """
    if validate:
        report = validate_ontology_triplet(profile_path, attack_path, opinion_path)
        report.raise_if_invalid()

    profile_path = Path(profile_path)
    attack_path = Path(attack_path)
    opinion_path = Path(opinion_path)

    with open(profile_path, "r", encoding="utf-8") as fh:
        profile_tree = json.load(fh)
    with open(attack_path, "r", encoding="utf-8") as fh:
        attack_tree = json.load(fh)
    with open(opinion_path, "r", encoding="utf-8") as fh:
        opinion_tree = json.load(fh)

    return {
        "PROFILE": profile_tree,
        "ATTACK": attack_tree,
        "OPINION": opinion_tree,
    }
