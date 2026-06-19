from __future__ import annotations

"""
Shared visual theme + hierarchy helpers for the stage-01 sample overviews
=========================================================================
One palette, one title system and one set of ontology-walking helpers so the
PROFILE / OPINION / ATTACK / integrated overview figures look like one product.

Design rules baked in here:
  * a single navy→teal→ember categorical palette (colour-blind-safe ordering);
  * a two-line title block (title + muted subtitle) that is pinned to the top
    of the paper so multi-panel section titles never collide with it;
  * Helvetica/Arial everywhere, white paper, hairline grids only.
"""

from pathlib import Path

# ── palette ──────────────────────────────────────────────────────────────────
PAL = dict(
    navy="#0f2240", blue="#1d4e89", sky="#2980b9", teal="#2a9d8f",
    green="#27ae60", orange="#e76f51", red="#c0392b", amber="#c89b3c",
    purple="#8e44ad", ink="#14213d", muted="#5a6b86", slate="#34495e",
    panel="#ffffff", line="#dbe3ef", grid="#eef2f8", gold="#f0c040",
)
# long categorical sequence (domains / families / tactics)
SEQ = ["#1d4e89", "#2a9d8f", "#e76f51", "#c89b3c", "#2980b9", "#8e44ad",
       "#16a085", "#d35400", "#7f8c8d", "#c0392b", "#27ae60", "#2c3e50",
       "#e67e22", "#1abc9c", "#9b59b6", "#34495e", "#f39c12", "#0f2240"]
# diverging ramp for adversarial direction:  −1 erode ⟵ 0 neutral ⟶ +1 amplify
DIR_COLORS = {1: "#c0392b", -1: "#1d4e89", 0: "#b7c1d3"}
DIR_LABELS = {1: "+1 amplify", -1: "−1 erode", 0: "0 neutral"}

FONT = dict(family="Helvetica Neue, Arial, sans-serif", color=PAL["ink"])


# ── title / layout ───────────────────────────────────────────────────────────
def base_layout(title: str, subtitle: str = "", h: int = 620, w: int = 1100,
                t: int | None = None, multipanel: bool = False) -> dict:
    """Standard layout. ``multipanel`` widens the top margin so subplot section
    titles sit clearly below the pinned two-line title block."""
    if t is None:
        t = 138 if multipanel else 98
    text = f"<b>{title}</b>"
    if subtitle:
        text += (f"<br><span style='font-size:13px;color:{PAL['muted']}'>"
                 f"{subtitle}</span>")
    return dict(
        title=dict(text=text, x=0.5, xanchor="center", y=0.975, yanchor="top",
                   font=dict(size=22, **FONT)),
        paper_bgcolor="white", plot_bgcolor="white", font=FONT,
        height=h, width=w, margin=dict(t=t, l=70, r=40, b=64),
    )


def style_subplot_titles(fig, size: int = 14, color: str | None = None) -> None:
    """Give every auto-generated subplot title a consistent weight/size."""
    color = color or PAL["slate"]
    for ann in fig.layout.annotations:
        ann.font = dict(size=size, color=color, family=FONT["family"])


def save_fig(fig, out_dir: Path, name: str, scale: int = 2,
             html: bool = True) -> None:
    """Write the static PNG to ``out_dir/pngs`` and (always, by default) the
    interactive HTML to ``out_dir/htmls`` so every figure is explorable."""
    png_dir = out_dir / "pngs"
    png_dir.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(png_dir / f"{name}.png"), scale=scale)
    if html:
        html_dir = out_dir / "htmls"
        html_dir.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(html_dir / f"{name}.html"), include_plotlyjs="cdn")
    print(f"  wrote {name}  (png{'+html' if html else ''})")


# ── ontology hierarchy walking ───────────────────────────────────────────────
def is_branch(key: str) -> bool:
    """Branch keys start with an uppercase letter; metadata keys are lowercase."""
    return bool(key) and key[0].isupper()


def leaf_count(node) -> int:
    if not isinstance(node, dict):
        return 1
    kids = [k for k in node if is_branch(k)]
    if not kids:
        return 1
    return sum(leaf_count(node[k]) for k in kids)


def walk_hierarchy(root_node: dict, root_label: str, max_depth: int = 3,
                   leaf_attr: str | None = None):
    """Flatten a nested ontology dict into parallel sunburst/treemap arrays.

    Returns a dict with ids/parents/labels/values/domains and, when
    ``leaf_attr`` is given, ``leaf_attrs`` carrying that metadata value for
    terminal nodes (``None`` for internal nodes).
    """
    ids, parents, labels, values, domains, attrs = [], [], [], [], [], []
    ids.append(root_label); parents.append(""); labels.append(root_label)
    values.append(leaf_count(root_node)); domains.append("root"); attrs.append(None)

    def walk(node, path, depth, domain):
        for k in [k for k in node if is_branch(k)]:
            child = node[k]
            cid = path + " > " + k
            child_branches = [x for x in child if is_branch(x)] if isinstance(child, dict) else []
            ids.append(cid); parents.append(path)
            labels.append(k.replace("_", " "))
            values.append(leaf_count(child))
            domains.append(domain or k)
            if leaf_attr and not child_branches and isinstance(child, dict):
                attrs.append(child.get(leaf_attr))
            else:
                attrs.append(None)
            if depth < max_depth and child_branches:
                walk(child, cid, depth + 1, domain or k)

    walk(root_node, root_label, 1, "")
    return dict(ids=ids, parents=parents, labels=labels, values=values,
                domains=domains, leaf_attrs=attrs)


def domain_color_map(domains):
    uniq = [d for d in dict.fromkeys(domains) if d != "root"]
    return {d: SEQ[i % len(SEQ)] for i, d in enumerate(uniq)}


# ── small formatting helpers ─────────────────────────────────────────────────
def human(n: float) -> str:
    """Compact human-readable integer (1_234_567 → '1.23M')."""
    n = float(n)
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(n) >= div:
            return f"{n / div:.2f}{suf}"
    return f"{int(n):,}"
