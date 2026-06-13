"""
FeatureRegistry — auto-discovers profile feature hierarchy from DataFrame columns.

Naming convention (generalisable to any future inventory):
  profile_cont_{inventory_tokens}_{dimension}_{facet}_pct   → hierarchical continuous
  profile_cont_{single_field}                               → scalar continuous
  profile_cat__{prefix}_{group}_{level}                     → one-hot categorical

Examples that work today AND with future extensions:
  profile_cont_big_five_agreeableness_altruism_pct
  profile_cont_dark_triad_machiavellianism_pct          (future)
  profile_cont_hexaco_honesty_humility_sincerity_pct    (future)
  profile_cont_age_years
  profile_cat__profile_cat_sex_Female
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FeatureLeaf:
    col: str              # original DataFrame column name
    label: str            # short human label ("Altruism")
    is_mean: bool = False # True when this is an aggregated mean col for the group


@dataclass
class FeatureDimension:
    """Mid-level node — e.g. Agreeableness within Big Five."""
    key: str                       # "big_five.agreeableness"
    label: str                     # "Agreeableness"
    inventory_key: str             # "big_five"
    leaves: List[FeatureLeaf] = field(default_factory=list)
    color: str = "#888888"
    is_categorical: bool = False   # True iff all leaves are one-hot

    @property
    def mean_col(self) -> Optional[str]:
        """Representative column: explicit mean col if present, else first facet."""
        for lf in self.leaves:
            if lf.is_mean:
                return lf.col
        return self.leaves[0].col if self.leaves else None

    @property
    def facet_cols(self) -> List[str]:
        return [lf.col for lf in self.leaves if not lf.is_mean]

    @property
    def all_cols(self) -> List[str]:
        return [lf.col for lf in self.leaves]


@dataclass
class FeatureInventory:
    """Top-level node — e.g. Big Five, Demographics."""
    key: str                                  # "big_five" or "demographics"
    label: str                                # "Big Five" or "Demographics"
    dimensions: Dict[str, FeatureDimension] = field(default_factory=dict)
    color: str = "#888888"

    @property
    def all_cols(self) -> List[str]:
        return [c for dim in self.dimensions.values() for c in dim.all_cols]

    @property
    def mean_cols(self) -> List[str]:
        return [dim.mean_col for dim in self.dimensions.values() if dim.mean_col]

    @property
    def facet_cols(self) -> List[str]:
        return [c for dim in self.dimensions.values() for c in dim.facet_cols]

    @property
    def continuous_dimensions(self) -> List[FeatureDimension]:
        return [d for d in self.dimensions.values() if not d.is_categorical]

    @property
    def categorical_dimensions(self) -> List[FeatureDimension]:
        return [d for d in self.dimensions.values() if d.is_categorical]


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_STRIP_SUFFIXES = ("_pct", "_years", "_score", "_proxy", "_index", "_z", "_norm")
_MEAN_TOKENS    = {"mean"}
# These first tokens always map to "demographics" even if only one column
_DEMO_SINGLETONS = {"age", "income", "education", "bmi", "weight", "height"}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip_suffix(s: str) -> str:
    for suf in _STRIP_SUFFIXES:
        if s.endswith(suf):
            return s[: -len(suf)]
    return s


def _title(s: str) -> str:
    return s.replace("_", " ").title()


def _safe_hex(color) -> str:
    """Convert any color spec to hex, never raising."""
    try:
        return mcolors.to_hex(color)
    except Exception:
        return "#888888"


def _assign_palette(keys: List[str]) -> Dict[str, str]:
    palette = sns.color_palette("tab10", max(len(keys), 1))
    return {k: _safe_hex(c) for k, c in zip(keys, palette)}


def _sub_palette(base_hex: str, n: int) -> List[str]:
    """Generate n distinguishable sub-colours around a base hex colour."""
    if n <= 0:
        return []
    try:
        raw = sns.light_palette(base_hex, n_colors=n + 3, reverse=False)[2:-1]
        # Ensure we have exactly n entries
        raw = (raw * ((n // len(raw)) + 1))[:n]
        return [_safe_hex(c) for c in raw]
    except Exception:
        return [base_hex] * n


# ─────────────────────────────────────────────────────────────────────────────
# FeatureRegistry
# ─────────────────────────────────────────────────────────────────────────────

class FeatureRegistry:
    """
    Auto-discovers profile feature hierarchy from DataFrame column names.
    Produces:  FeatureInventory → FeatureDimension → FeatureLeaf

    No hardcoded variable names. Works for Big Five, Dark Triad, HEXACO,
    or any future personality subtrees added under the profile_cont_* prefix.
    """

    def __init__(self, df: pd.DataFrame):
        self.inventories: Dict[str, FeatureInventory] = {}
        self._col_labels: Dict[str, str] = {}   # col → "Inventory — Dim — Leaf" string
        self._build_continuous(df)
        self._build_categorical(df)
        self._assign_colors()

    # ── Continuous columns ────────────────────────────────────────────────────

    def _build_continuous(self, df: pd.DataFrame) -> None:
        cont_cols = [c for c in df.columns if c.startswith("profile_cont_")]
        if not cont_cols:
            return

        # Token lists (suffix-stripped)
        parsed: Dict[str, List[str]] = {
            col: _strip_suffix(col[len("profile_cont_"):]).split("_")
            for col in cont_cols
        }

        # Build prefix trie: depth → prefix → [cols]
        prefix_depth: Dict[int, Dict[tuple, List[str]]] = defaultdict(lambda: defaultdict(list))
        for col, tokens in parsed.items():
            for d in range(1, len(tokens) + 1):
                prefix_depth[d][tuple(tokens[:d])].append(col)

        # Inventory prefix = shallowest prefix that:
        #   (a) covers ≥2 columns AND
        #   (b) has ≥2 distinct sub-prefixes at depth+1
        inventory_prefixes: Dict[tuple, List[str]] = {}
        for d in sorted(prefix_depth.keys()):
            for p, cs in prefix_depth[d].items():
                if len(cs) < 2:
                    continue
                children = {
                    cp for cp in prefix_depth[d + 1]
                    if cp[:d] == p
                }
                if len(children) >= 2:
                    inventory_prefixes[p] = cs

        # For each column: shallowest matching inventory prefix wins
        col_to_inv: Dict[str, Optional[tuple]] = {}
        for col, tokens in parsed.items():
            best: Optional[tuple] = None
            for d in range(1, len(tokens) + 1):
                if tuple(tokens[:d]) in inventory_prefixes:
                    best = tuple(tokens[:d])
                    break
            col_to_inv[col] = best

        # Group: inv_prefix → dim_token → [cols]
        inv_dim_cols: Dict[tuple, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        for col, inv_prefix in col_to_inv.items():
            tokens = parsed[col]
            if inv_prefix is None:
                group = "demographics" if tokens[0] in _DEMO_SINGLETONS else "other"
                inv_dim_cols[("__scalar__", group)]["__scalar__"].append(col)
            else:
                d = len(inv_prefix)
                dim_tok = tokens[d] if d < len(tokens) else "__scalar__"
                inv_dim_cols[inv_prefix][dim_tok].append(col)

        # Populate inventories
        for inv_prefix, dim_dict in inv_dim_cols.items():
            if inv_prefix[0] == "__scalar__":
                inv_key = inv_prefix[1]
            else:
                inv_key = "_".join(inv_prefix)
            inv_label = _title(inv_key)

            if inv_key not in self.inventories:
                self.inventories[inv_key] = FeatureInventory(key=inv_key, label=inv_label)
            inv = self.inventories[inv_key]

            for dim_tok, cols in dim_dict.items():
                dim_key = f"{inv_key}.{dim_tok}"
                dim_label = _title(dim_tok) if dim_tok != "__scalar__" else inv_label
                if dim_key not in inv.dimensions:
                    inv.dimensions[dim_key] = FeatureDimension(
                        key=dim_key, label=dim_label, inventory_key=inv_key
                    )
                dim = inv.dimensions[dim_key]

                for col in cols:
                    tokens = parsed[col]
                    if inv_prefix[0] == "__scalar__":
                        leaf_label = _title("_".join(tokens))
                        remaining: List[str] = []
                    else:
                        d = len(inv_prefix) + 1
                        remaining = tokens[d:]
                        leaf_label = _title("_".join(remaining)) if remaining else dim_label
                    is_mean = any(t in _MEAN_TOKENS for t in remaining)
                    dim.leaves.append(FeatureLeaf(col=col, label=leaf_label, is_mean=is_mean))
                    full_label = f"{dim_label} — {leaf_label}" if leaf_label != dim_label else dim_label
                    self._col_labels[col] = full_label

    # ── Categorical (one-hot) columns ─────────────────────────────────────────

    def _build_categorical(self, df: pd.DataFrame) -> None:
        cat_cols = [c for c in df.columns if c.startswith("profile_cat__")]
        if not cat_cols:
            return

        # group → {level: col}
        group_map: Dict[str, Dict[str, str]] = defaultdict(dict)
        for col in cat_cols:
            rest = col[len("profile_cat__"):]
            # Strip any redundant "profile_cat_" repetition
            while rest.startswith("profile_cat_"):
                rest = rest[len("profile_cat_"):]
            parts = rest.split("_")
            if len(parts) < 2:
                continue
            level = parts[-1]
            group = "_".join(parts[:-1])
            group_map[group][level] = col

        for group, level_cols in group_map.items():
            # Categorical groups go into "demographics" inventory
            inv_key = "demographics"
            if inv_key not in self.inventories:
                self.inventories[inv_key] = FeatureInventory(key=inv_key, label="Demographics")
            inv = self.inventories[inv_key]

            dim_key = f"demographics.{group}"
            if dim_key not in inv.dimensions:
                inv.dimensions[dim_key] = FeatureDimension(
                    key=dim_key,
                    label=_title(group),
                    inventory_key=inv_key,
                    is_categorical=True,
                )
            dim = inv.dimensions[dim_key]

            for level, col in sorted(level_cols.items()):
                dim.leaves.append(FeatureLeaf(col=col, label=level, is_mean=False))
                self._col_labels[col] = f"{_title(group)} = {level}"

    # ── Colours ───────────────────────────────────────────────────────────────

    def _assign_colors(self) -> None:
        inv_keys = sorted(self.inventories.keys())
        inv_pal = _assign_palette(inv_keys)
        for inv_key, inv in self.inventories.items():
            inv.color = inv_pal[inv_key]
            dims = list(inv.dimensions.values())
            if len(dims) > 1:
                sub = _sub_palette(inv.color, len(dims))
                for dim, c in zip(dims, sub):
                    dim.color = c
            elif dims:
                dims[0].color = inv.color

    # ── Public API ────────────────────────────────────────────────────────────

    def label(self, col: str) -> str:
        return self._col_labels.get(col, col.split("_")[-1].title())

    def all_continuous_cols(self) -> List[str]:
        return [c for inv in self.inventories.values()
                for dim in inv.dimensions.values()
                if not dim.is_categorical
                for c in dim.all_cols]

    def dimension_mean_cols(self) -> List[Tuple[str, str, str]]:
        """[(col, dim_label, inv_label), ...] — one representative col per dimension."""
        result = []
        for inv in self.inventories.values():
            for dim in inv.dimensions.values():
                if not dim.is_categorical and dim.mean_col:
                    result.append((dim.mean_col, dim.label, inv.label))
        return result

    def col_to_dimension(self, col: str) -> Optional[FeatureDimension]:
        for inv in self.inventories.values():
            for dim in inv.dimensions.values():
                if col in dim.all_cols:
                    return dim
        return None

    def col_to_inventory(self, col: str) -> Optional[FeatureInventory]:
        for inv in self.inventories.values():
            if col in inv.all_cols:
                return inv
        return None

    def top_features_by_weight(
        self,
        moderator_weights: Optional[pd.DataFrame],
        n: int = 12,
        continuous_only: bool = True,
    ) -> List[Tuple[str, str]]:
        """
        [(col, label), ...] top-n features by |normalised_weight_pct|.
        Falls back to dimension mean-cols ranked by label if weights unavailable.
        `continuous_only` skips one-hot columns (unsuitable for scatter plots).
        """
        known = set(self._col_labels.keys())

        if moderator_weights is not None and "normalized_weight_pct" in moderator_weights.columns:
            mw = moderator_weights.copy()
            mw["abs_w"] = mw["normalized_weight_pct"].abs()
            mw = mw.sort_values("abs_w", ascending=False)
            result = []
            for _, row in mw.iterrows():
                col = row.get("term", "")
                if col not in known:
                    continue
                if continuous_only:
                    dim = self.col_to_dimension(col)
                    if dim and dim.is_categorical:
                        continue
                result.append((col, self._col_labels[col]))
                if len(result) >= n:
                    break
            if result:
                return result

        # Fallback: dimension mean-cols (preferred over raw facets for scatter)
        return [(c, lbl) for c, lbl, _ in self.dimension_mean_cols()[:n]]

    def categorical_group_info(self) -> List[Tuple[str, str, List[Tuple[str, str]]]]:
        """
        [(dim_label, inv_label, [(level_label, col), ...]), ...]
        — one entry per categorical dimension.
        """
        result = []
        for inv in self.inventories.values():
            for dim in inv.categorical_dimensions:
                levels = [(lf.label, lf.col) for lf in dim.leaves]
                result.append((dim.label, inv.label, levels))
        return result

    def summary(self) -> str:
        lines = ["FeatureRegistry:"]
        for inv in self.inventories.values():
            lines.append(f"  [{inv.label}] ({len(inv.all_cols)} cols)")
            for dim in inv.dimensions.values():
                tag = "categorical" if dim.is_categorical else f"{len(dim.facet_cols)} facets"
                lines.append(f"    {dim.label}: {tag} + {'mean col' if dim.mean_col else 'no mean'}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Clustermap row-colour builder — one annotation bar per DIMENSION
# ─────────────────────────────────────────────────────────────────────────────

def build_row_color_annotations(
    registry: FeatureRegistry,
    profiles: List[str],
    meta: pd.DataFrame,
    max_bars: int = 10,
) -> pd.DataFrame:
    """
    Build DataFrame of row-colour annotations for seaborn clustermap.
    One column per *dimension* (not per inventory), so Big Five → 5 bars,
    Demographics → Age bar + Sex bar, etc.

    Continuous dimensions: quintile colour gradient of mean_col.
    Categorical dimensions: distinct colour per level.

    `max_bars` caps total bars to avoid visual clutter.
    """
    annotations: Dict[str, pd.Series] = {}
    count = 0

    for inv in registry.inventories.values():
        for dim in inv.dimensions.values():
            if count >= max_bars:
                break

            if dim.is_categorical:
                level_cols = [(lf.col, lf.label) for lf in dim.leaves]
                if not level_cols:
                    continue
                level_pal = sns.color_palette("Set2", len(level_cols))
                # Default to first level colour
                series = pd.Series(
                    _safe_hex(level_pal[0]), index=profiles, name=dim.label
                )
                for (col, _level_lbl), color in zip(level_cols, level_pal):
                    if col in meta.columns:
                        mask = meta.reindex(profiles)[col].fillna(0).astype(bool)
                        series[mask] = _safe_hex(color)
                annotations[dim.label] = series
                count += 1

            else:
                mc = dim.mean_col
                if not mc or mc not in meta.columns:
                    continue
                vals = meta.reindex(profiles)[mc].fillna(meta[mc].median())
                try:
                    quintiles = pd.qcut(vals, 5, labels=False, duplicates="drop")
                except Exception:
                    quintiles = pd.Series(2, index=pd.Index(profiles))
                sub_pal = _sub_palette(dim.color, 5)
                colors = pd.Series(
                    [
                        sub_pal[min(int(q), 4)] if not pd.isna(q) else "#cccccc"
                        for q in quintiles
                    ],
                    index=profiles,
                    name=dim.label,
                )
                annotations[dim.label] = colors
                count += 1

        if count >= max_bars:
            break

    if not annotations:
        return pd.DataFrame()
    return pd.concat(list(annotations.values()), axis=1)
