"""
semantic_scale.py
-----------------
Converts raw profile feature values into human/LLM-interpretable semantic labels.

Design principles
~~~~~~~~~~~~~~~~~
- Continuous percentile features (0–100): quintile bracket + semantic poles
    neuroticism_mean_pct = 78  →  "High Neuroticism (78th pct) — neurotic / anxious"
- Absolute-numeric features (age, income, ...): formatted with units
    age_years = 35  →  "Age: 35 years"
- Categorical one-hot features: level label as-is
    sex = "Female"  →  "Sex: Female"

Built-in PercentileScale definitions exist for all Big Five traits and their
standard NEO-PI-R facets. Any future inventory (Dark Triad, HEXACO, …) auto-falls
back to a generic "Low {label} ↔ High {label}" scheme — zero code changes needed.

Override / extension priority (highest to lowest):
  1. Exact dimension key match     ("big_five_neuroticism_anxiety")
  2. Parent-prefix strip match     ("big_five_neuroticism_*" → neuroticism scale)
  3. Generic inventory-agnostic    ("*_high_label ↔ *_low_label" = dim label)

Public API
~~~~~~~~~~
  registry = SemanticScaleRegistry()            # uses built-ins
  registry.fmt_feature("big_five_neuroticism_mean_pct", 78.3)
    → "High Neuroticism (78th pct) — neurotic, anxious, emotionally reactive"
  registry.fmt_profile(continuous_attributes, categorical_attributes)
    → multi-line human/LLM-readable profile string
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Core data structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PercentileScale:
    """Semantic metadata for one continuous percentile dimension."""
    dimension_key: str        # normalised column fragment, e.g. "big_five_neuroticism"
    dimension_label: str      # human label, e.g. "Neuroticism"
    low_pole: str             # low end description, e.g. "Emotionally stable, calm"
    high_pole: str            # high end description, e.g. "Neurotic, anxious"
    # Five quintile labels from very-low to very-high (index by quintile 0–4)
    quintile_labels: Tuple[str, str, str, str, str] = (
        "Very Low", "Low", "Moderate", "High", "Very High"
    )

    def quintile_idx(self, value: float) -> int:
        """Map 0–100 percentile value to quintile index 0–4."""
        if value < 20:  return 0
        if value < 40:  return 1
        if value < 60:  return 2
        if value < 80:  return 3
        return 4

    def quintile_label(self, value: float) -> str:
        return self.quintile_labels[self.quintile_idx(value)]

    def brief(self, value: float) -> str:
        """'High Neuroticism (78th pct)'"""
        pct_str = _ordinal(round(value))
        ql = self.quintile_label(value)
        return f"{ql} {self.dimension_label} ({pct_str} pct)"

    def full(self, value: float) -> str:
        """'High Neuroticism (78th pct) — neurotic, anxious, emotionally reactive'"""
        ql = self.quintile_idx(value)
        pole = self.high_pole if ql >= 3 else (self.low_pole if ql <= 1 else None)
        base = self.brief(value)
        return f"{base} — {pole}" if pole else base


# ─────────────────────────────────────────────────────────────────────────────
# Ordinal helper
# ─────────────────────────────────────────────────────────────────────────────

def _ordinal(n: int) -> str:
    s = ["th", "st", "nd", "rd", "th", "th", "th", "th", "th", "th"]
    v = n % 100
    if 11 <= v <= 13:
        return f"{n}th"
    return f"{n}{s[n % 10]}"


# ─────────────────────────────────────────────────────────────────────────────
# Absolute-numeric feature units  (key fragment → unit suffix)
# ─────────────────────────────────────────────────────────────────────────────

_ABSOLUTE_UNITS: Dict[str, str] = {
    "age_years": "years",
    "income": "USD/year",
    "bmi": "kg/m²",
    "weight": "kg",
    "height": "cm",
    "education_years": "years of education",
}


def _strip_pct_suffix(s: str) -> str:
    for suf in ("_pct", "_years", "_score", "_proxy", "_index", "_z", "_norm"):
        if s.endswith(suf):
            return s[: -len(suf)]
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Built-in Big Five trait + facet scales (NEO-PI-R pole descriptions)
# ─────────────────────────────────────────────────────────────────────────────

def _ps(key: str, label: str, low: str, high: str,
        ql: Tuple[str, str, str, str, str] = ("Very Low", "Low", "Moderate", "High", "Very High")
        ) -> PercentileScale:
    return PercentileScale(
        dimension_key=key, dimension_label=label,
        low_pole=low, high_pole=high, quintile_labels=ql
    )


_BIG_FIVE_SCALES: List[PercentileScale] = [
    # ── Neuroticism ──────────────────────────────────────────────────────────
    _ps("big_five_neuroticism", "Neuroticism",
        "Emotionally stable, calm, resilient",
        "Neurotic, anxious, emotionally reactive",
        ("Very stable", "Stable", "Moderate", "Anxious", "Highly neurotic")),
    _ps("big_five_neuroticism_anxiety", "Anxiety",
        "Calm, relaxed, unworried",
        "Anxious, tense, apprehensive"),
    _ps("big_five_neuroticism_anger_hostility", "Anger / Hostility",
        "Slow to anger, easy-going",
        "Easily angered, irritable, hostile"),
    _ps("big_five_neuroticism_depression", "Depression (N)",
        "Rarely despondent, optimistic",
        "Prone to sadness, guilt, hopelessness"),
    _ps("big_five_neuroticism_self_consciousness", "Self-Consciousness",
        "Comfortable socially, not easily embarrassed",
        "Self-conscious, shy, easily embarrassed"),
    _ps("big_five_neuroticism_impulsiveness", "Impulsiveness",
        "Able to resist urges, self-controlled",
        "Impulsive, difficulty resisting cravings"),
    _ps("big_five_neuroticism_vulnerability", "Vulnerability",
        "Handles stress well, composed under pressure",
        "Vulnerable to stress, prone to panic"),

    # ── Extraversion ─────────────────────────────────────────────────────────
    _ps("big_five_extraversion", "Extraversion",
        "Introverted, reserved, solitary",
        "Extraverted, outgoing, energetic",
        ("Highly introverted", "Introverted", "Ambivert", "Extraverted", "Highly extraverted")),
    _ps("big_five_extraversion_warmth", "Warmth",
        "Reserved, cool with others",
        "Warm, affectionate, friendly"),
    _ps("big_five_extraversion_gregariousness", "Gregariousness",
        "Prefers solitude, avoids crowds",
        "Enjoys crowds, highly sociable"),
    _ps("big_five_extraversion_assertiveness", "Assertiveness",
        "Unassuming, reluctant to lead",
        "Assertive, dominant, confident"),
    _ps("big_five_extraversion_activity", "Activity Level",
        "Leisurely pace, relaxed",
        "Fast-paced, energetic, busy"),
    _ps("big_five_extraversion_excitement_seeking", "Excitement Seeking",
        "Cautious, avoids stimulation",
        "Thrill-seeking, excitement-hungry"),
    _ps("big_five_extraversion_positive_emotions", "Positive Emotions",
        "Less exuberant, serious",
        "Joyful, enthusiastic, optimistic"),

    # ── Openness to Experience ────────────────────────────────────────────────
    _ps("big_five_openness_to_experience", "Openness to Experience",
        "Conventional, practical, down-to-earth",
        "Open, curious, imaginative, creative",
        ("Highly conventional", "Conventional", "Moderate", "Open", "Highly open")),
    _ps("big_five_openness_fantasy", "Fantasy",
        "Pragmatic, grounded, literal",
        "Vivid imagination, fantasy-prone"),
    _ps("big_five_openness_aesthetics", "Aesthetics",
        "Little interest in arts",
        "Deep aesthetic appreciation, moved by beauty"),
    _ps("big_five_openness_feelings", "Feelings",
        "Emotionally reserved, limited introspection",
        "Receptive to feelings, high emotional awareness"),
    _ps("big_five_openness_actions", "Actions (Openness)",
        "Prefers routine, set habits",
        "Welcomes novelty, variety-seeking"),
    _ps("big_five_openness_ideas", "Ideas",
        "Practical, limited intellectual curiosity",
        "Intellectually curious, enjoys debate"),
    _ps("big_five_openness_values", "Values",
        "Conservative, respects tradition",
        "Challenges authority, non-conformist"),

    # ── Agreeableness ─────────────────────────────────────────────────────────
    _ps("big_five_agreeableness", "Agreeableness",
        "Competitive, skeptical, challenging",
        "Cooperative, trusting, empathetic",
        ("Highly competitive", "Competitive", "Moderate", "Agreeable", "Highly agreeable")),
    _ps("big_five_agreeableness_trust", "Trust",
        "Suspicious, cynical",
        "Trusting, assumes good intentions"),
    _ps("big_five_agreeableness_straightforwardness", "Straightforwardness",
        "Manipulative, deceptive when convenient",
        "Frank, sincere, unwilling to manipulate"),
    _ps("big_five_agreeableness_altruism", "Altruism",
        "Self-interested, little concern for others",
        "Generous, helpful, eager to assist"),
    _ps("big_five_agreeableness_compliance", "Compliance",
        "Aggressive, competitive, uncooperative",
        "Deferent, avoids conflict, easy-going"),
    _ps("big_five_agreeableness_modesty", "Modesty",
        "Arrogant, self-satisfied",
        "Humble, self-effacing, dislikes boasting"),
    _ps("big_five_agreeableness_tender_mindedness", "Tender-Mindedness",
        "Tough-minded, unsympathetic, objective",
        "Sympathetic, moved by others' needs"),

    # ── Conscientiousness ─────────────────────────────────────────────────────
    _ps("big_five_conscientiousness", "Conscientiousness",
        "Flexible, spontaneous, disorganised",
        "Disciplined, organised, dependable, goal-directed",
        ("Highly disorganised", "Low C", "Moderate", "Conscientious", "Highly conscientious")),
    _ps("big_five_conscientiousness_competence", "Competence",
        "Feels unprepared, doubts own ability",
        "Capable, efficient, capable of handling things"),
    _ps("big_five_conscientiousness_order", "Order",
        "Disorganised, careless about environment",
        "Orderly, tidy, well-organised"),
    _ps("big_five_conscientiousness_dutifulness", "Dutifulness",
        "Flexible about obligations, casual about ethics",
        "Dutiful, guided by conscience, principled"),
    _ps("big_five_conscientiousness_achievement_striving", "Achievement Striving",
        "Easygoing, content with modest achievement",
        "Ambitious, driven, works hard toward goals"),
    _ps("big_five_conscientiousness_self_discipline", "Self-Discipline",
        "Procrastinates, easily distracted",
        "Self-disciplined, persistent, finishes tasks"),
    _ps("big_five_conscientiousness_deliberation", "Deliberation",
        "Impulsive, acts without thinking",
        "Cautious, deliberate, careful before acting"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

class SemanticScaleRegistry:
    """
    Resolves a feature column name to a PercentileScale (or generic fallback)
    and formats feature values as human/LLM-readable text.

    Extend with custom scales by passing extra_scales or a path to a JSON file:
        registry = SemanticScaleRegistry(
            extra_scales=[PercentileScale("dark_triad_narcissism", "Narcissism",
                                          "Humble, modest", "Narcissistic, grandiose")]
        )
    """

    def __init__(
        self,
        extra_scales: Optional[Sequence[PercentileScale]] = None,
        json_path: Optional[str | Path] = None,
    ) -> None:
        self._scales: Dict[str, PercentileScale] = {}
        for sc in _BIG_FIVE_SCALES:
            self._scales[sc.dimension_key] = sc
        if extra_scales:
            for sc in extra_scales:
                self._scales[sc.dimension_key] = sc
        if json_path:
            self._load_json(Path(json_path))

    def _load_json(self, p: Path) -> None:
        """Load extra scales from JSON [{dimension_key, dimension_label, low_pole, high_pole}, ...]."""
        for d in json.loads(p.read_text(encoding="utf-8")):
            sc = PercentileScale(**d)
            self._scales[sc.dimension_key] = sc

    # ------------------------------------------------------------------

    def get_scale(self, col: str) -> Optional[PercentileScale]:
        """
        Resolve feature column → PercentileScale.
        col may be a full DataFrame column ("profile_cont_big_five_neuroticism_mean_pct")
        or a short key fragment ("big_five_neuroticism_mean_pct").

        Lookup order:
          1. Exact match on stripped key
          2. Parent-prefix strip: remove last '_*' token one step at a time
          3. None (caller should use generic fallback)
        """
        # Normalise: strip prefix + suffix
        key = col
        for pfx in ("profile_cont_", "profile_cat__"):
            if key.startswith(pfx):
                key = key[len(pfx):]
                break
        key = _strip_pct_suffix(key)  # remove _mean, _pct, _years, etc. at the end
        # Also strip trailing _mean token
        if key.endswith("_mean"):
            key = key[: -len("_mean")]

        # Walk up the hierarchy: strip one trailing token at a time
        while key:
            if key in self._scales:
                return self._scales[key]
            # Remove the last underscore-separated token
            idx = key.rfind("_")
            if idx < 0:
                break
            key = key[:idx]

        return None

    def _generic_label(self, col: str) -> str:
        """Human label from column name when no scale is registered."""
        key = col
        for pfx in ("profile_cont_", "profile_cat__profile_cat_", "profile_cat__"):
            if key.startswith(pfx):
                key = key[len(pfx):]
                break
        key = _strip_pct_suffix(key)
        if key.endswith("_mean"):
            key = key[: -len("_mean")]
        return key.replace("_", " ").title()

    # ------------------------------------------------------------------

    def fmt_feature(self, col: str, value: float, verbose: bool = True) -> str:
        """
        Format one (column, value) pair as a human-readable string.

        Continuous percentile → quintile label + poles if verbose.
        Absolute numeric (age, etc.) → value + unit.
        """
        # Absolute numeric (not percentile)
        inner = col.removeprefix("profile_cont_").removeprefix("profile_cat__")
        for frag, unit in _ABSOLUTE_UNITS.items():
            if frag in inner:
                val_int = int(round(value))
                return f"{self._generic_label(col)}: {val_int} {unit}"

        # Percentile continuous
        scale = self.get_scale(col)
        if scale is not None:
            return scale.full(value) if verbose else scale.brief(value)

        # Generic fallback: Low / High {label}
        label = self._generic_label(col)
        q_idx = (
            0 if value < 20 else
            1 if value < 40 else
            2 if value < 60 else
            3 if value < 80 else
            4
        )
        qlabels = ("Very Low", "Low", "Moderate", "High", "Very High")
        pct_str = _ordinal(round(value))
        return f"{qlabels[q_idx]} {label} ({pct_str} pct)"

    # ------------------------------------------------------------------

    def fmt_profile(
        self,
        continuous: Dict[str, float],
        categorical: Dict[str, str],
        *,
        include_facets: bool = False,
        verbose: bool = True,
        skip_fragments: Optional[Sequence[str]] = ("heuristic", "resilience_index",
                                                    "shift_sensitivity", "resilience"),
    ) -> str:
        """
        Generate a multi-line LLM-readable profile description.

        Shows:
          - Categorical attributes (sex, etc.)
          - Absolute-numeric attributes (age)
          - Per-dimension MEAN values (or single facet if no mean exists)
          - Individual facets only if include_facets=True

        Example output:
            Sex: Female
            Age: 34 years
            Neuroticism: High (78th pct) — neurotic, anxious, emotionally reactive
            Extraversion: Moderate (49th pct)
            Openness to Experience: Very High (91st pct) — open, curious, creative
            Agreeableness: High (68th pct) — cooperative, trusting
            Conscientiousness: Low (32nd pct) — flexible, spontaneous
        """
        _skip = set(skip_fragments) if skip_fragments else set()

        def _should_skip(key: str) -> bool:
            k = key.lower()
            return any(frag in k for frag in _skip)

        lines: List[str] = []

        # Categorical
        for attr_key, attr_val in sorted(categorical.items()):
            if _should_skip(attr_key):
                continue
            lines.append(f"{attr_key.replace('_', ' ').title()}: {attr_val}")

        # Absolute numeric (age etc.)
        for col_key, val in sorted(continuous.items()):
            if _should_skip(col_key):
                continue
            inner = col_key.removeprefix("profile_cont_")
            for frag, unit in _ABSOLUTE_UNITS.items():
                if frag in inner:
                    val_int = int(round(val))
                    label = self._generic_label(col_key)
                    lines.append(f"{label}: {val_int} {unit}")
                    break

        # Collect mean cols (summary-level) and facet cols separately
        mean_keys: Dict[str, float] = {}   # dimension_fragment → value
        facet_keys: Dict[str, float] = {}  # full col_key → value

        for col_key, val in continuous.items():
            if _should_skip(col_key):
                continue
            inner = col_key.removeprefix("profile_cont_")
            # Skip absolute-numeric already handled
            if any(frag in inner for frag in _ABSOLUTE_UNITS):
                continue
            stripped = _strip_pct_suffix(inner)
            if stripped.endswith("_mean"):
                dim_frag = stripped[: -len("_mean")]
                mean_keys[dim_frag] = val
            else:
                facet_keys[col_key] = val

        # Render mean-level (one line per dimension)
        rendered_dims: set = set()
        for dim_frag, val in sorted(mean_keys.items()):
            full_col = f"profile_cont_{dim_frag}_mean_pct"
            line = self.fmt_feature(full_col, val, verbose=verbose)
            lines.append(line)
            rendered_dims.add(dim_frag)

        # Render facets where no mean was emitted (or always if include_facets)
        for col_key, val in sorted(facet_keys.items()):
            inner = col_key.removeprefix("profile_cont_")
            stripped = _strip_pct_suffix(inner)
            # Check if parent dim was already rendered
            parent = stripped.rsplit("_", 1)[0] if "_" in stripped else stripped
            if not include_facets and parent in rendered_dims:
                continue
            lines.append(f"  {self.fmt_feature(col_key, val, verbose=verbose)}")
            rendered_dims.add(parent)

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton for convenience
# ─────────────────────────────────────────────────────────────────────────────

_default_registry: Optional[SemanticScaleRegistry] = None


def get_default_registry() -> SemanticScaleRegistry:
    """Return the module-level default SemanticScaleRegistry (lazy-initialised)."""
    global _default_registry
    if _default_registry is None:
        _default_registry = SemanticScaleRegistry()
    return _default_registry


def fmt_feature(col: str, value: float, verbose: bool = True) -> str:
    """Module-level convenience wrapper for SemanticScaleRegistry.fmt_feature."""
    return get_default_registry().fmt_feature(col, value, verbose=verbose)


def fmt_profile(
    continuous: Dict[str, float],
    categorical: Dict[str, str],
    *,
    include_facets: bool = False,
    verbose: bool = True,
    skip_fragments: Optional[Sequence[str]] = ("heuristic", "resilience_index",
                                                "shift_sensitivity", "resilience"),
) -> str:
    """Module-level convenience wrapper for SemanticScaleRegistry.fmt_profile."""
    return get_default_registry().fmt_profile(
        continuous, categorical,
        include_facets=include_facets, verbose=verbose,
        skip_fragments=skip_fragments,
    )
