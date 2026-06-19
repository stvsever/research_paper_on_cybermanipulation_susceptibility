from __future__ import annotations

"""
High-resolution, metadata-driven PROFILE sampling engine
========================================================

Purpose
-------
Sample coherent, *high-resolution* pseudoprofiles directly from the **structural
combinatorial sampling metadata** baked into the production PROFILE ontology
(`src/backend/ontology/separate/production/PROFILE/profile.json`).

Unlike :mod:`src.backend.utils.profile_sampling` — which re-derives variable
types from a small set of keyword heuristics and therefore only ever touches a
handful of constructs (sex, age, a few personality traits) — this engine reads
the per-leaf metadata that the ontology actually carries and samples *every*
leaf according to its declared nature:

  ============== ================= ==========================================
  modality_type  sampling_role     how it is sampled here
  ============== ================= ==========================================
  categorical    option_value      one mutually-exclusive option per variable,
                                    uniform over the *feasible* set (forward
                                    checking against _coherence_rules)
  continuous     scale_facet       real-valued percentile, rejection-bounded
                                    around its parent-scale anchor (no 0/100
                                    pile-up)
  continuous     scale_dimension   real-valued percentile (full-range)
  ordinal        scale_dimension   continuous position on a 0..100 support
                                    scale (no discrete Likert levels)
  continuous     trait_state       numeric scalar (age in years/months/days,
                                    ISO dates) with internal unit-coherence
  identifier     identifier        freshly generated ID string
  ============== ================= ==========================================

Why this reduces the "range-restriction problem"
------------------------------------------------
Range restriction (statistical and state-space) is what you get when the sampler
only explores a narrow band of each variable.  This engine attacks both:

  * **value restriction** — the fundamentals (chronological age, Big Five
    personality) are spread by *stratified linear-spaced anchors*, giving exactly
    uniform, full-range marginals — no Gaussian central tendency, no pile-up at
    the 0/100 boundaries, real-valued (non-round) samples throughout.
  * **state-space restriction** — *every* sampleable leaf contributes a value, so
    a profile is a full assignment over the deployment state space.

Maximal entropy subject to coherence
------------------------------------
Sex and gender are sampled at maximal entropy (uniform; gender uniform over the
sex-feasible options).  Selection is **forward-checking constraint satisfaction**
driven by the ontology's ``_coherence_rules``: each categorical is drawn
uniformly over only the options still feasible given the partial assignment
(a male-assigned person can never draw a cisgender-woman identity), and
female-biology variables (pregnancy, menstruation, …) are *omitted* as
not-applicable for male-assigned profiles rather than forced to a sentinel.
"""

import math
import random
import sys
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# This module lives under a stage directory whose name starts with a digit
# (not an importable package), so bootstrap the project root for src.backend.*.
_PROJECT_ROOT = Path(__file__).resolve().parents[6]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

OntologyNode = Dict[str, Any]

# ─────────────────────────────────────────────────────────────────────────────
# Structural metadata vocabulary (mirrors profile.json `_sampling_rules`)
# ─────────────────────────────────────────────────────────────────────────────

OPTION_ROLES = frozenset({"option_value", "unknown_marker"})
PCT_MIN, PCT_MAX = 0.0, 100.0

# Per-leaf metadata keys never start with an uppercase letter; subtree (branch)
# keys always do.  This is the same convention used by ontology_utils.
def _is_branch_key(key: str) -> bool:
    return bool(key) and key[0].isupper()


def _upper_children(node: Any) -> Dict[str, Any]:
    if not isinstance(node, dict):
        return {}
    return {k: v for k, v in node.items() if _is_branch_key(k)}


def _is_leaf(node: Any) -> bool:
    """A node is a leaf when it carries no subtree (branch) children."""
    if not isinstance(node, dict):
        return True
    return not any(_is_branch_key(k) for k in node)


def _meta(node: Any) -> Dict[str, Any]:
    if not isinstance(node, dict):
        return {}
    return {k: v for k, v in node.items() if not _is_branch_key(k)}


def _all_leaves_are_options(node: Any) -> bool:
    """True iff every terminal leaf under ``node`` is a categorical option."""
    if _is_leaf(node):
        return _meta(node).get("sampling_role") in OPTION_ROLES
    return all(_all_leaves_are_options(v) for v in _upper_children(node).values())


# ─────────────────────────────────────────────────────────────────────────────
# Variable specifications discovered from the ontology
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OptionLeaf:
    path: str           # full " > " joined leaf path
    label: str          # last path segment
    weight: float       # prevalence_weight (unknowns are down-weighted)
    is_unknown: bool


@dataclass
class CategoricalVar:
    """A single mutually-exclusive categorical variable (one option is chosen).

    Hierarchical categoricals (e.g. Gender_Identity, whose Woman/Man/Nonbinary
    sub-groups are alternatives) are *flattened*: every terminal option leaf in
    the choice subtree becomes a candidate, so the sampler picks exactly one
    high-resolution leaf across the whole state space of the variable.
    """
    name: str
    path: str
    construct_type: str
    options: List[OptionLeaf]
    key: str = ""        # unique, path-derived output key (assigned post-build)

    @property
    def substantive_options(self) -> List[OptionLeaf]:
        real = [o for o in self.options if not o.is_unknown]
        return real or list(self.options)


@dataclass
class ContinuousScaleVar:
    """A psychometric / continuous scale: percentile facets sharing an anchor."""
    name: str
    path: str
    construct_type: str
    leaves: List[Tuple[str, str]]   # (column_key, leaf_path)


@dataclass
class OrdinalVar:
    """A single ordinal item, sampled as a continuous position on a 0..100 scale."""
    column_key: str
    path: str
    construct_type: str


@dataclass
class TraitScalarVar:
    """An absolute numeric scalar / date (age years, dates, perinatal fields)."""
    column_key: str
    path: str
    label: str
    value_format: str   # int | iso_date | float


@dataclass
class IdentifierVar:
    column_key: str
    path: str
    label: str


@dataclass
class VariablePlan:
    categoricals: List[CategoricalVar] = field(default_factory=list)
    continuous_scales: List[ContinuousScaleVar] = field(default_factory=list)
    ordinals: List[OrdinalVar] = field(default_factory=list)
    trait_scalars: List[TraitScalarVar] = field(default_factory=list)
    identifiers: List[IdentifierVar] = field(default_factory=list)

    def summary(self) -> Dict[str, int]:
        return {
            "categorical_variables": len(self.categoricals),
            "categorical_options_total": sum(len(c.options) for c in self.categoricals),
            "continuous_scale_groups": len(self.continuous_scales),
            "continuous_scale_leaves": sum(len(s.leaves) for s in self.continuous_scales),
            "ordinal_variables": len(self.ordinals),
            "trait_scalar_variables": len(self.trait_scalars),
            "identifier_variables": len(self.identifiers),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Plan extraction (recursive CHOICE / CONTAINER / SCALE descent)
# ─────────────────────────────────────────────────────────────────────────────

def _column_key(path: str) -> str:
    """Stable, compact column key for a leaf path (last 2-3 informative parts)."""
    parts = [p.strip() for p in path.split(">") if p.strip()]
    tail = parts[-3:] if len(parts) >= 3 else parts
    key = "_".join(tail)
    return key.lower().replace(" ", "_").replace("-", "_")


def _collect_choice_options(
    node: Any, path_parts: List[str]
) -> Tuple[List[OptionLeaf], List[Tuple[str, Any]]]:
    """Flatten a categorical choice subtree into terminal option leaves.

    Descends through *pure-categorical* branch children (which are sub-option
    groups, e.g. Woman_Identities under Gender_Identity).  Any branch that is
    not pure-categorical, and any non-option leaf, is returned separately so the
    caller can sample it independently.
    """
    options: List[OptionLeaf] = []
    independent: List[Tuple[str, Any]] = []
    for key, child in _upper_children(node).items():
        child_path = path_parts + [key]
        joined = " > ".join(child_path)
        if _is_leaf(child):
            m = _meta(child)
            if m.get("sampling_role") in OPTION_ROLES:
                is_unknown = bool(m.get("is_unknown_marker"))
                weight = float(m.get("prevalence_weight", 1.0) or 1.0)
                options.append(OptionLeaf(joined, key, weight, is_unknown))
            else:
                independent.append((joined, child))
        elif _all_leaves_are_options(child):
            sub_opts, _ = _collect_choice_options(child, child_path)
            options.extend(sub_opts)
        else:
            independent.append((joined, child))
    return options, independent


def build_variable_plan(person_tree: OntologyNode, root_name: str = "PERSON") -> VariablePlan:
    """Discover every sampleable variable from the ontology's structural metadata."""
    plan = VariablePlan()
    seen_scalar: set[str] = set()

    def emit_leaf(joined: str, node: Any) -> None:
        m = _meta(node)
        role = m.get("sampling_role")
        modality = m.get("modality_type")
        ctype = str(m.get("construct_type", "other"))
        if role == "identifier":
            plan.identifiers.append(IdentifierVar(_column_key(joined), joined, joined.split(">")[-1].strip()))
        elif role == "trait_state":
            key = _column_key(joined)
            if key not in seen_scalar:
                seen_scalar.add(key)
                plan.trait_scalars.append(
                    TraitScalarVar(key, joined, joined.split(">")[-1].strip(), str(m.get("value_format", "float")))
                )
        elif modality == "ordinal":
            plan.ordinals.append(OrdinalVar(_column_key(joined), joined, ctype))
        # lone continuous leaves are folded into their parent scale by recurse()

    def recurse(node: Any, path_parts: List[str]) -> None:
        name = path_parts[-1] if path_parts else root_name
        children = _upper_children(node)
        if not children:
            return

        leaf_children = {k: v for k, v in children.items() if _is_leaf(v)}
        branch_children = {k: v for k, v in children.items() if not _is_leaf(v)}

        # Direct categorical options whose exclusivity group is THIS node →
        # this node is a CHOICE point (one option is selected for it).
        direct_choice = any(
            _meta(v).get("sampling_role") in OPTION_ROLES
            and _meta(v).get("exclusivity_group") == name
            for v in leaf_children.values()
        )

        if direct_choice:
            options, independent = _collect_choice_options(node, path_parts)
            if options:
                ctype = str(_meta(next(iter(leaf_children.values()))).get("construct_type", "other"))
                plan.categoricals.append(
                    CategoricalVar(name, " > ".join(path_parts), ctype, options)
                )
            # Non-option siblings / impure branches sampled independently.
            for joined, child in independent:
                if _is_leaf(child):
                    emit_leaf(joined, child)
                else:
                    recurse(child, [p.strip() for p in joined.split(">")])
            return

        # ── CONTAINER / SCALE node ───────────────────────────────────────────
        # Continuous leaves directly under this node form one correlated scale.
        cont_leaves = [
            (_column_key(" > ".join(path_parts + [k])), " > ".join(path_parts + [k]))
            for k, v in leaf_children.items()
            if _meta(v).get("modality_type") == "continuous"
            and _meta(v).get("sampling_role") in ("scale_facet", "scale_dimension")
        ]
        if cont_leaves:
            ctype = str(_meta(next(iter(leaf_children.values()))).get("construct_type", "other"))
            plan.continuous_scales.append(
                ContinuousScaleVar(name, " > ".join(path_parts), ctype, cont_leaves)
            )

        # Other leaf children (ordinal / trait / identifier / stray options).
        stray_options: List[Tuple[OptionLeaf, str]] = []
        for k, v in leaf_children.items():
            m = _meta(v)
            role = m.get("sampling_role")
            joined = " > ".join(path_parts + [k])
            if m.get("modality_type") == "continuous" and role in ("scale_facet", "scale_dimension"):
                continue  # already folded into the scale above
            if role in OPTION_ROLES:
                grp = str(m.get("exclusivity_group") or name)
                stray_options.append((
                    OptionLeaf(joined, k, float(m.get("prevalence_weight", 1.0) or 1.0), bool(m.get("is_unknown_marker"))),
                    grp,
                ))
            else:
                emit_leaf(joined, v)

        # Stray leaf options whose exclusivity group ≠ parent name (e.g.
        # Vital_Status > Status, Preferred_Language).  Each distinct exclusivity
        # group under this container is one mutually-exclusive categorical
        # variable; branch children are recursed independently below.
        if stray_options:
            by_group: Dict[str, List[OptionLeaf]] = defaultdict(list)
            for opt, grp in stray_options:
                by_group[grp].append(opt)
            for grp, opts in by_group.items():
                plan.categoricals.append(
                    CategoricalVar(grp, " > ".join(path_parts), "other", opts)
                )

        for k, v in branch_children.items():
            recurse(v, path_parts + [k])

    recurse(person_tree, [root_name])

    # Assign unique, path-derived output keys (variable names can repeat across
    # subtrees, e.g. "Tradition"; keying output by name would lose data).
    used: set[str] = set()
    for cat in plan.categoricals:
        base = _column_key(cat.path) or cat.name.lower()
        key, n = base, 2
        while key in used:
            key, n = f"{base}_{n}", n + 1
        used.add(key)
        cat.key = key
    return plan


# ─────────────────────────────────────────────────────────────────────────────
# Sampler configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SamplingConfig:
    # Population
    n_profiles: int = 10_000
    seed: int = 42
    entropy_threshold: float = 0.99      # normalised Shannon entropy to clear

    # Chronological age — sampled as a CONTINUOUS variable by linear-spaced
    # max-entropy spread over [age_min, age_max]; never categorised into bands.
    age_min: int = 16
    age_max: int = 80
    reference_year: int = 2026

    # Continuous scales whose marginal is forced to a uniform, full-range spread
    # via stratified linear-spaced anchors — the fundamentals (Big Five).
    balance_scale_markers: Tuple[str, ...] = ("Big_Five",)
    facet_jitter: float = 9.0            # within-scale facet spread (rejection-bounded)

    # Coherence: forward-checking constraint satisfaction from _coherence_rules.
    coherence: bool = True

    # Prevalence↔power trade-off for the curated (prevalence-weighted) variables.
    # Kish (1988) multipurpose "power allocation": category i is drawn ∝ p_i^q,
    # a single uniform compromise exponent between population-proportional (q=1)
    # and equal / max-entropy (q=0). q=0.5 (square-root) is the canonical
    # compromise — it caps dominant categories and lifts low-prevalence subgroups
    # so both remain analysable. Design weights (population/sample prevalence) are
    # reported so population-level interpretation stays recoverable.
    prevalence_compromise_exponent: float = 0.6
    min_cell_count: int = 100            # reporting threshold (flag thinner cells)


# ─────────────────────────────────────────────────────────────────────────────
# A sampled profile
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HighResProfile:
    profile_id: str
    demographics: Dict[str, Any]
    categorical: Dict[str, str]              # variable key → chosen option label
    continuous: Dict[str, float]             # column key → percentile (real-valued)
    ordinal: Dict[str, float]                # column key → continuous position [0,100]
    scalar: Dict[str, Any]                   # column key → numeric / date
    identifiers: Dict[str, str]
    selected_leaf_nodes: List[str]           # chosen categorical leaf paths

    def n_resolved_leaves(self) -> int:
        return (
            len(self.categorical) + len(self.continuous)
            + len(self.ordinal) + len(self.scalar) + len(self.identifiers)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Max-entropy population sampler
# ─────────────────────────────────────────────────────────────────────────────

def normalised_entropy(counts: Sequence[int]) -> float:
    total = sum(counts)
    k = len(counts)
    if total == 0 or k <= 1:
        return 1.0 if k <= 1 else 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return h / math.log(k)


_GENDER_DIVERSE_TOKENS = (
    "nonbinary", "genderqueer", "agender", "genderfluid", "two_spirit",
    "demigender", "third_gender", "nonconforming", "gender_diverse",
)


def classify_gender_type(label: str) -> str:
    """Map a gender-identity leaf label to a coarse type (used to derive modality)."""
    l = label.lower()
    if "questioning" in l:
        return "questioning"
    if "trans_woman" in l or "transfeminine" in l:
        return "trans_woman"
    if "trans_man" in l or "transmasculine" in l:
        return "trans_man"
    if "cis_woman" in l:
        return "cis_woman"
    if "cis_man" in l:
        return "cis_man"
    if any(t in l for t in _GENDER_DIVERSE_TOKENS):
        return "nonbinary"
    if "woman" in l:
        return "cis_woman"
    if "man" in l:
        return "cis_man"
    return "other"


class MaxEntropyPopulationSampler:
    """Samples a population at maximal entropy subject to hard coherence constraints.

    Selection is **forward-checking constraint satisfaction**: variables are drawn
    in a dependency-respecting order and each categorical is sampled uniformly over
    only the options that remain *feasible* given the partial assignment (e.g. a
    male-assigned person can never draw a cisgender-woman identity), while
    female-biology variables are simply *omitted* (not-applicable) for male-assigned
    profiles.  Continuous fundamentals (age, Big Five) are spread by stratified
    linear-spaced anchors, so their marginals are uniform and full-range — no
    Gaussian central tendency, no range restriction, no pile-up at 0/100.
    """

    def __init__(
        self,
        plan: VariablePlan,
        config: SamplingConfig,
        coherence_rules: Optional[Dict[str, Any]] = None,
        population_priors: Optional[Dict[str, Any]] = None,
    ):
        self.plan = plan
        self.cfg = config
        self.rng = random.Random(config.seed)

        cr = coherence_rules or {}
        applicability = cr.get("applicability", []) if config.coherence else []
        constraints = cr.get("constraints", []) if config.coherence else []
        derivations = cr.get("derivations", []) if config.coherence else []

        self._cat_by_name: Dict[str, CategoricalVar] = {c.name: c for c in plan.categoricals}
        self._applicable_if: Dict[str, Dict[str, Any]] = {}
        for rule in applicability:
            for v in rule.get("variables", []):
                self._applicable_if[v] = rule.get("applicable_if", {})
        self._constraints_by_target: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for rule in constraints:
            self._constraints_by_target[rule.get("target")].append(rule)
        # Variables deterministically derived from the sampled gender type
        # (Gender_Modality, Pronouns, …) for coherence.
        self._gender_type_derivations: Dict[str, Dict[str, str]] = {}
        for rule in derivations:
            if "from_gender_type" in rule and rule.get("target"):
                self._gender_type_derivations[rule["target"]] = rule["from_gender_type"]

        # ── Population priors + prevalence↔power compromise ──────────────────
        pp = population_priors or {}
        self._variable_weights: Dict[str, Dict[str, float]] = pp.get("variable_weights", {})
        self._q: float = min(1.0, max(0.0, config.prevalence_compromise_exponent))
        # realistic (q=1) prevalence per curated variable, for design weights.
        self._realistic_prev: Dict[str, Dict[str, float]] = {}
        for vname, w in self._variable_weights.items():
            tot = sum(max(v, 0.0) for v in w.values()) or 1.0
            self._realistic_prev[vname] = {k: max(v, 0.0) / tot for k, v in w.items()}
        # rare-option label penalties: {multiplier: [substrings]}
        self._rare_penalties: List[Tuple[float, Tuple[str, ...]]] = sorted(
            (
                (float(mult), tuple(s.lower() for s in subs))
                for mult, subs in pp.get("rare_option_penalties", {}).items()
                if mult.replace(".", "", 1).isdigit()
            ),
            key=lambda x: x[0],
        )
        # Condition variables that must be sampled before their dependents.
        self._priority_vars: List[str] = [
            v for v in cr.get("condition_variables_sampled_first", [])
            if v in self._cat_by_name
        ]

        # Stratified linear-spaced anchor pools → uniform full-range marginals.
        n = config.n_profiles
        self._balance_scale_paths = {
            s.path for s in plan.continuous_scales
            if any(m in s.path for m in config.balance_scale_markers)
        }
        self._anchor_pools: Dict[str, List[float]] = {
            p: self._linspace_pool(0.0, 100.0, n) for p in self._balance_scale_paths
        }
        self._anchor_cursor: Dict[str, int] = {p: 0 for p in self._anchor_pools}
        self._age_pool = self._linspace_pool(float(config.age_min), float(config.age_max), n)
        self._age_cursor = 0

        # Reporting accumulators.
        self._age_values: List[int] = []
        self._scale_means: Dict[str, List[float]] = defaultdict(list)
        self._sex_counts: Counter = Counter()
        self._gender_counts: Counter = Counter()
        self._modality_counts: Counter = Counter()
        self._omitted_counts: Counter = Counter()
        self._curated_counts: Dict[str, Counter] = {v: Counter() for v in self._variable_weights}

    # ── linear-spaced (max-entropy, full-range) pools ─────────────────────────
    def _linspace_pool(self, lo: float, hi: float, n: int) -> List[float]:
        if n <= 1:
            return [self.rng.uniform(lo, hi)]
        pool = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
        self.rng.shuffle(pool)
        return pool

    # ── constraint evaluation (forward checking) ──────────────────────────────
    def _cond_holds(self, cond: Dict[str, Any], assignment: Dict[str, str], age: int) -> bool:
        if not cond:
            return False
        var = cond.get("variable")
        if var == "__age__":
            if "gt" in cond and not age > cond["gt"]:
                return False
            if "ge" in cond and not age >= cond["ge"]:
                return False
            if "lt" in cond and not age < cond["lt"]:
                return False
            if "le" in cond and not age <= cond["le"]:
                return False
            return True
        val = assignment.get(var)
        if "in" in cond:
            return val in set(cond["in"])
        if "not_in" in cond:
            return val is not None and val not in set(cond["not_in"])
        return False

    def _is_applicable(self, name: str, assignment: Dict[str, str], age: int) -> bool:
        cond = self._applicable_if.get(name)
        return True if cond is None else self._cond_holds(cond, assignment, age)

    def _feasible_options(self, cat: CategoricalVar, assignment: Dict[str, str], age: int) -> List[OptionLeaf]:
        forbidden: set = set()
        for rule in self._constraints_by_target.get(cat.name, []):
            if self._cond_holds(rule.get("condition", {}), assignment, age):
                forbidden.update(rule.get("forbid", []))
        feasible = [o for o in cat.options if o.label not in forbidden]
        return feasible or list(cat.options)   # never strand a variable

    # ── realistic option weighting ────────────────────────────────────────────
    def _rare_penalty(self, label: str) -> float:
        l = label.lower()
        for mult, subs in self._rare_penalties:      # smallest multiplier first
            if any(s in l for s in subs):
                return mult
        return 1.0

    def _option_weights(self, cat: CategoricalVar, feasible: List[OptionLeaf]) -> List[float]:
        """Weights for a categorical draw.

        Prevalence-weighted ("curated") variables use Kish power-compromise
        allocation: w_i ∝ p_i^q, where p_i is the realistic prevalence and q is
        the global compromise exponent (q=1 proportional, q=0 equal). This caps
        dominant categories and lifts low-prevalence subgroups so both stay
        analysable. Explicitly zero-weighted options (e.g. Antarctica) stay 0.
        Non-curated variables keep a rare/extreme-label penalty.
        """
        curated = self._variable_weights.get(cat.name)
        if not curated:
            return [self._rare_penalty(o.label) for o in feasible]
        return [max(curated.get(o.label, 0.0), 0.0) ** self._q for o in feasible]

    def _weighted_choice(self, options: List[OptionLeaf], weights: List[float]) -> OptionLeaf:
        if not any(w > 0 for w in weights):
            return self.rng.choice(options)
        return self.rng.choices(options, weights=weights, k=1)[0]

    # ── continuous samplers ───────────────────────────────────────────────────
    def _bounded_jitter(self, anchor: float) -> float:
        """Truncated-Gaussian facet via rejection → no probability pile-up at 0/100."""
        for _ in range(8):
            v = anchor + self.rng.gauss(0.0, self.cfg.facet_jitter)
            if 0.0 <= v <= 100.0:
                return v
        return min(100.0, max(0.0, anchor))

    def _sample_scale(self, scale: ContinuousScaleVar) -> Dict[str, float]:
        if scale.path in self._anchor_pools:           # Big Five: stratified anchor
            c = self._anchor_cursor[scale.path]
            pool = self._anchor_pools[scale.path]
            anchor = pool[c] if c < len(pool) else self.rng.uniform(0.0, 100.0)
            self._anchor_cursor[scale.path] = c + 1
        else:                                           # other scales: full-range uniform
            anchor = self.rng.uniform(0.0, 100.0)
        out: Dict[str, float] = {}
        for col, _path in scale.leaves:
            out[col] = round(self._bounded_jitter(anchor), 2)
        if scale.leaves:
            # The trait LEVEL is the anchor (uniform, full-range); facets are
            # coherent noisy indicators around it. Reporting the level as the
            # anchor keeps the trait marginal exactly max-entropy rather than
            # letting facet-averaging regress it toward the centre.
            out[f"{_column_key(scale.path)}_mean_pct"] = round(anchor, 2)
            self._scale_means[scale.path].append(anchor)
        return out

    def _sample_ordinal(self) -> float:
        # Continuous position on a 0..100 support scale (no discrete Likert levels,
        # no round/binary values) — uniform, full-range, max entropy.
        return round(self.rng.uniform(0.0, 100.0), 2)

    def _next_age(self) -> int:
        c = self._age_cursor
        if c < len(self._age_pool):
            self._age_cursor = c + 1
            return int(round(self._age_pool[c]))
        return self.rng.randint(self.cfg.age_min, self.cfg.age_max)

    # ── single profile ────────────────────────────────────────────────────────
    def _sample_profile(self, index: int) -> HighResProfile:
        age_years = self._next_age()
        self._age_values.append(age_years)
        year_of_birth = self.cfg.reference_year - age_years

        assignment: Dict[str, str] = {}     # var name → label (for constraints)
        chosen: Dict[str, OptionLeaf] = {}  # cat.key → option

        sex_var = self._cat_by_name.get("Sex_Assigned_At_Birth")
        gender_var = self._cat_by_name.get("Gender_Identity")
        modality_var = self._cat_by_name.get("Gender_Modality")

        # 1) Sex — power-compromise prevalence (Male/Female dominant, Intersex lifted).
        if sex_var:
            o = self._weighted_choice(sex_var.options, self._option_weights(sex_var, sex_var.options))
            chosen[sex_var.key] = o
            assignment[sex_var.name] = o.label
            self._sex_counts[o.label] += 1
            self._curated_counts.get(sex_var.name, Counter())[o.label] += 1
        # 2) Gender — power-compromise prevalence over the sex-feasible options
        #    (cross-sex cisgender identities removed by the coherence constraints).
        gender_type = "other"
        if gender_var:
            feas = self._feasible_options(gender_var, assignment, age_years)
            o = self._weighted_choice(feas, self._option_weights(gender_var, feas))
            chosen[gender_var.key] = o
            assignment[gender_var.name] = o.label
            gender_type = classify_gender_type(o.label)
            self._gender_counts[o.label] += 1
            self._curated_counts.get(gender_var.name, Counter())[o.label] += 1
        # 3) Variables derived from the gender type (modality, pronouns, …).
        derived_keys: set = set()
        for tvar_name, gmap in self._gender_type_derivations.items():
            tvar = self._cat_by_name.get(tvar_name)
            if not tvar:
                continue
            lab = gmap.get(gender_type)
            opt = next((o for o in tvar.options if o.label == lab), None) or self.rng.choice(tvar.options)
            chosen[tvar.key] = opt
            assignment[tvar.name] = opt.label
            derived_keys.add(tvar.key)
            if tvar_name == "Gender_Modality":
                self._modality_counts[opt.label] += 1

        special = {v.key for v in (sex_var, gender_var) if v} | derived_keys

        def draw(cat: CategoricalVar) -> None:
            if cat.key in chosen or cat.key in special:
                return
            if not self._is_applicable(cat.name, assignment, age_years):
                self._omitted_counts[cat.name] += 1
                return
            feasible = self._feasible_options(cat, assignment, age_years)
            o = self._weighted_choice(feasible, self._option_weights(cat, feasible))
            chosen[cat.key] = o
            assignment[cat.name] = o.label
            if cat.name in self._curated_counts:
                self._curated_counts[cat.name][o.label] += 1

        # 4) Constraint-condition variables first, then everything else, each
        #    sampled at realistic prevalence over its still-feasible options.
        for vname in self._priority_vars:
            draw(self._cat_by_name[vname])
        for cat in self.plan.categoricals:
            draw(cat)

        categorical: Dict[str, str] = {}
        name_to_label: Dict[str, str] = {}
        selected: List[str] = []
        for cat in self.plan.categoricals:
            if cat.key in chosen:
                o = chosen[cat.key]
                categorical[cat.key] = o.label
                name_to_label[cat.name] = o.label
                selected.append(o.path)

        continuous: Dict[str, float] = {}
        for scale in self.plan.continuous_scales:
            continuous.update(self._sample_scale(scale))
        ordinal = {ov.column_key: self._sample_ordinal() for ov in self.plan.ordinals}
        scalar = {tv.column_key: self._sample_trait_scalar(tv, age_years, year_of_birth)
                  for tv in self.plan.trait_scalars}
        identifiers = {
            idv.column_key: f"{idv.label[:12]}-{uuid.UUID(int=self.rng.getrandbits(128)).hex[:12]}"
            for idv in self.plan.identifiers
        }
        demographics = self._extract_demographics(name_to_label, age_years, gender_type, continuous)
        return HighResProfile(
            profile_id=f"profile_{index:05d}",
            demographics=demographics,
            categorical=categorical,
            continuous=continuous,
            ordinal=ordinal,
            scalar=scalar,
            identifiers=identifiers,
            selected_leaf_nodes=selected,
        )

    def _sample_trait_scalar(self, tv: TraitScalarVar, age_years: int, year_of_birth: int) -> Any:
        key = tv.column_key
        if "age_years" in key:
            return age_years
        if "age_months" in key:
            return age_years * 12 + self.rng.randint(0, 11)
        if "age_days" in key:
            return int(age_years * 365.25) + self.rng.randint(0, 364)
        if "date_of_birth" in key:
            return (date(year_of_birth, 1, 1) + timedelta(days=self.rng.randint(0, 364))).isoformat()
        if "year_of_birth" in key:
            return year_of_birth
        if "data_collection_timestamp" in key:
            return date(self.cfg.reference_year, self.rng.randint(1, 12), self.rng.randint(1, 28)).isoformat()
        if tv.value_format == "iso_date":
            return date(year_of_birth, 1, 1).isoformat()
        if tv.value_format == "int":
            return self.rng.randint(0, 100)
        return round(self.rng.uniform(0, 100), 2)

    @staticmethod
    def _level(pct: float) -> str:
        return "very low" if pct < 20 else "low" if pct < 40 else "average" if pct < 60 \
            else "high" if pct < 80 else "very high"

    def _extract_demographics(
        self, cat_labels: Dict[str, str], age_years: int, gender_type: str, continuous: Dict[str, float]
    ) -> Dict[str, Any]:
        dem: Dict[str, Any] = {"age_years": age_years}
        for name in ("Sex_Assigned_At_Birth", "Gender_Identity", "Gender_Modality",
                     "Relationship_Status", "Citizenship_Status"):
            if name in cat_labels:
                dem[name.lower()] = cat_labels[name]
        dem["gender_type"] = gender_type
        big_five: Dict[str, Dict[str, Any]] = {}
        for k, v in continuous.items():
            if "big_five" in k and k.endswith("_mean_pct"):
                trait = k.split("big_five_")[-1].replace("_mean_pct", "")
                big_five[trait] = {"pct": v, "level": self._level(v)}
        if big_five:
            dem["big_five"] = big_five
        return dem

    # ── population draw (the while loop) ──────────────────────────────────────
    def sample_population(self) -> Tuple[List[HighResProfile], Dict[str, Any]]:
        target = self.cfg.n_profiles
        profiles: List[HighResProfile] = []
        while len(profiles) < target:
            profiles.append(self._sample_profile(len(profiles) + 1))
        return profiles, self._report(len(profiles))

    def _report(self, n: int) -> Dict[str, Any]:
        def hist_entropy(values: List[float], lo: float, hi: float, bins: int = 32) -> float:
            if not values or hi <= lo:
                return 1.0
            counts = [0] * bins
            for v in values:
                idx = int((v - lo) / (hi - lo) * bins)
                counts[min(bins - 1, max(0, idx))] += 1
            return round(normalised_entropy(counts), 6)

        ent = {"age": hist_entropy(self._age_values, self.cfg.age_min, self.cfg.age_max)}
        for path, means in self._scale_means.items():
            if path in self._balance_scale_paths:
                ent["bigfive:" + path.split(" > ")[-1].lower()] = hist_entropy(means, 0.0, 100.0)
        ok = all(v >= self.cfg.entropy_threshold for v in ent.values())

        def prop(counter: Counter) -> Dict[str, float]:
            tot = sum(counter.values()) or 1
            return {k: round(v / tot, 4) for k, v in counter.most_common()}

        # Per curated variable: realistic prevalence, achieved (compromise) share,
        # design weight (= realistic/achieved → reweight to population), min cell.
        subgroup: Dict[str, Any] = {}
        min_cells = []
        for vname, counts in self._curated_counts.items():
            tot = sum(counts.values()) or 1
            realistic = self._realistic_prev.get(vname, {})
            cats: Dict[str, Any] = {}
            for label, c in counts.most_common():
                ach = c / tot
                rp = realistic.get(label, 0.0)
                cats[label] = {
                    "n": c,
                    "achieved": round(ach, 4),
                    "realistic": round(rp, 4),
                    "design_weight": round(rp / ach, 4) if ach > 0 else None,
                }
            if counts:
                mc = min(counts.values())
                min_cells.append((vname, mc))
            subgroup[vname] = {"n_categories": len(counts), "min_cell_count": min(counts.values()) if counts else 0,
                               "categories": cats}

        thin = sorted([(v, m) for v, m in min_cells if m < self.cfg.min_cell_count], key=lambda x: x[1])
        return {
            "n_profiles": n,
            "entropy_threshold": self.cfg.entropy_threshold,
            "max_entropy_targets": "chronological_age + Big Five personality (continuous, linear-spaced, full-range)",
            "normalised_entropy_per_max_entropy_target": ent,
            "max_entropy_reached": ok,
            "subgroup_allocation": {
                "method": "Kish power-compromise allocation (p^q)",
                "compromise_exponent_q": self._q,
                "min_cell_count_threshold": self.cfg.min_cell_count,
                "variables_below_threshold": {v: m for v, m in thin},
                "per_variable": subgroup,
            },
            "sex_marginal": prop(self._sex_counts),
            "gender_identity_marginal": prop(self._gender_counts),
            "gender_modality_marginal": prop(self._modality_counts),
            "applicability_omissions": dict(self._omitted_counts),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience entry point
# ─────────────────────────────────────────────────────────────────────────────

def sample_high_res_population(
    ontology: Dict[str, Any], config: SamplingConfig
) -> Tuple[VariablePlan, List[HighResProfile], Dict[str, Any]]:
    """Sample from a full ontology dict (PERSON + _coherence_rules)."""
    plan = build_variable_plan(ontology["PERSON"])
    sampler = MaxEntropyPopulationSampler(
        plan, config,
        coherence_rules=ontology.get("_coherence_rules"),
        population_priors=ontology.get("_population_priors"),
    )
    profiles, report = sampler.sample_population()
    return plan, profiles, report


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation
# ─────────────────────────────────────────────────────────────────────────────

def profile_to_record(p: HighResProfile, include_leaves: bool = True) -> Dict[str, Any]:
    """Full, human-readable expanded record for a single profile."""
    rec: Dict[str, Any] = {
        "profile_id": p.profile_id,
        "demographics": p.demographics,
        "categorical_attributes": p.categorical,
        "continuous_attributes": p.continuous,
        "ordinal_attributes": p.ordinal,
        "scalar_attributes": p.scalar,
        "identifiers": p.identifiers,
    }
    if include_leaves:
        rec["selected_leaf_nodes"] = p.selected_leaf_nodes
    return rec


def encode_population_columnar(
    plan: VariablePlan, profiles: List[HighResProfile], encoding: str = "index"
) -> Dict[str, Any]:
    """Compact, self-describing columnar encoding for a large population.

    Variable keys are stored ONCE in ``schema``; each profile stores only its
    aligned value vectors, so the file does not repeat ~750 keys per profile.

    encoding="index": categorical values stored as integer indices into each
        variable's option catalog (smallest on disk).
    encoding="labels": categorical values stored as option labels (larger, but
        directly readable without the catalog).
    """
    cat_keys = [c.key for c in plan.categoricals]
    cat_catalog = {c.key: [o.label for o in c.options] for c in plan.categoricals}
    cat_index = {c.key: {o.label: i for i, o in enumerate(c.options)} for c in plan.categoricals}

    # Numeric column order: continuous (+ per-scale means) then ordinals.
    sample_cont = profiles[0].continuous if profiles else {}
    cont_keys = list(sample_cont.keys())
    ord_keys = [o.column_key for o in plan.ordinals]
    numeric_keys = cont_keys + ord_keys

    encoded_profiles: List[Dict[str, Any]] = []
    for p in profiles:
        if encoding == "index":
            cat_vec: List[Any] = [cat_index[k].get(p.categorical.get(k), -1) for k in cat_keys]
        else:
            cat_vec = [p.categorical.get(k) for k in cat_keys]
        num_vec = [p.continuous.get(k) for k in cont_keys] + [p.ordinal.get(k) for k in ord_keys]
        encoded_profiles.append({
            "id": p.profile_id,
            "demographics": p.demographics,
            "categorical": cat_vec,
            "numeric": num_vec,
            "scalar": p.scalar,
            "identifiers": p.identifiers,
        })

    return {
        "_format": (
            "columnar-v1 (encoding=%s). For profile i: decode categorical[j] with "
            "schema.categorical_keys[j] (index→schema.categorical_catalog[key][index] "
            "when encoding=index); numeric[j] aligns to schema.numeric_keys[j]. Use "
            "production_profile_sampling.load_population() to reconstruct dicts." % encoding
        ),
        "schema": {
            "encoding": encoding,
            "categorical_keys": cat_keys,
            "categorical_catalog": cat_catalog,
            "categorical_variables": {
                c.key: {"name": c.name, "path": c.path, "construct_type": c.construct_type}
                for c in plan.categoricals
            },
            "numeric_keys": numeric_keys,
            "continuous_key_count": len(cont_keys),
            "ordinal_key_count": len(ord_keys),
        },
        "profiles": encoded_profiles,
    }


def load_population(blob: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Reconstruct expanded profile dicts from a columnar-encoded population."""
    schema = blob["schema"]
    cat_keys = schema["categorical_keys"]
    catalog = schema["categorical_catalog"]
    numeric_keys = schema["numeric_keys"]
    encoding = schema.get("encoding", "index")
    out: List[Dict[str, Any]] = []
    for ep in blob["profiles"]:
        categorical: Dict[str, Any] = {}
        for k, v in zip(cat_keys, ep["categorical"]):
            if encoding == "index":
                opts = catalog[k]
                categorical[k] = opts[v] if isinstance(v, int) and 0 <= v < len(opts) else None
            else:
                categorical[k] = v
        numeric = {k: v for k, v in zip(numeric_keys, ep["numeric"])}
        out.append({
            "profile_id": ep["id"],
            "demographics": ep["demographics"],
            "categorical_attributes": categorical,
            "numeric_attributes": numeric,
            "scalar_attributes": ep["scalar"],
            "identifiers": ep["identifiers"],
        })
    return out
