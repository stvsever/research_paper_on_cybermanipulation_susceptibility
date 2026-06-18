from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import pandas as pd

from analyze_exposure_network import analyze_network
from build_interactive_exposure_map import build_interactive_exposure_map
from build_multiresolution_exposure_map import build_multiresolution_exposure_map
from common import (
    INTERACTION_WEIGHTS,
    FIGURES_DIR,
    REPORTS_DIR,
    ROLE_DEFINITIONS,
    ROLE_LABELS,
    ROOT,
    TABLES_DIR,
    artifact_path,
    ensure_dirs,
    write_json,
)
from prepare_exposure_network_data import prepare_inputs
from visualize_exposure_network import make_figures


def _metric(summary: dict[str, object], key: str) -> object:
    return summary.get(key, "--")


def _fmt(value: object, digits: int = 2) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if abs(value) < 1 and value != 0:
            return f"{value:.{digits}%}"
        return f"{value:,.{digits}f}"
    return str(value)


def _img(name: str, alt: str) -> str:
    path = FIGURES_DIR / f"{name}.png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(alt)}">'
        f"<figcaption>{html.escape(alt)}</figcaption></figure>"
    )


def _table(path: Path, rows: int = 12) -> str:
    df = pd.read_csv(path).head(rows)
    return df.to_html(index=False, classes="data-table", border=0, justify="left", float_format=lambda x: f"{x:,.3f}")


def _role_definitions_markdown() -> str:
    lines = ["| Role | Definition |", "|---|---|"]
    for role, definition in ROLE_DEFINITIONS.items():
        lines.append(f"| {ROLE_LABELS[role]} | {definition} |")
    return "\n".join(lines)


def _role_definitions_html() -> str:
    items = []
    for role, definition in ROLE_DEFINITIONS.items():
        items.append(
            f"<dt>{html.escape(ROLE_LABELS[role])}</dt><dd>{html.escape(definition)}</dd>"
        )
    return f'<dl class="role-definitions">{"".join(items)}</dl>'


def _markdown_report(summary: dict[str, object]) -> str:
    return f"""# Directed Exposure Network Substrate

## Technical Summary

This report isolates the empirical directed exposure-network substrate that will later be used to study network-conditioned cyber-manipulation effects. The first inspection window is the validated 60-position pilot slice, but the folder is structured to extend to the full PolitiSky24 exposure network.

- The pilot slice contains **{_fmt(_metric(summary, "pilot_nodes"))} positions** and **{_fmt(_metric(summary, "pilot_directed_edges"))} directed exposure edges**.
- The largest weak component contains **{_fmt(_metric(summary, "largest_weak_component"))} positions**, so the pilot slice is mostly connected while retaining peripheral positions.
- A node is an observed user position. A directed edge is a weighted plausible exposure relation from a visible target to the source user who engaged with that target.
- Edge direction is **TargetUserId -> SourceUserId**: the visible actor/content source points toward the user who engaged and was plausibly exposed.
- This is structural only. No profiles, opinions, attack vectors, baseline scores, post scores, or synthetic susceptibility values are used here.

## Exposure Definition And Edge Construction

The exposure graph is derived from observed engagement events. If a source user liked, reposted, or quoted a target user, the target user was plausibly visible to the source user.

```text
exposure_raw_weight = {INTERACTION_WEIGHTS["Like"]:.2f} * Like
                    + {INTERACTION_WEIGHTS["Repost"]:.2f} * Repost
                    + {INTERACTION_WEIGHTS["Quote"]:.2f} * Quote
```

Repeated interactions are summed, log-compressed, and normalized to `[0, 1]`. The local edge columns keep the same direction: `source_position_id` is the visible target; `target_position_id` is the exposed source.

![Exposure direction and formula](../figures/exposure_direction_and_formula.png)

## The 60-Position Pilot Slice As A Directed Exposure Map

The pilot slice is a readable window into the larger exposure network. In the figure, each node is a user position, each arrow is a directed exposure edge, node size follows full-network outgoing visibility, and edge width/opacity follows normalized exposure weight. Node identifiers are intentionally omitted from the figure to keep the map publication-ready.

![Pilot directed exposure network](../figures/pilot_directed_exposure_network.png)

The color legend uses interpretable exposure-mechanism roles:

{_role_definitions_markdown()}

## Sender Reach, Receiver Exposure, And Asymmetry

The same position can be a strong sender, strong receiver, both, or neither. This matters for later `PN` analysis because visible high-susceptibility senders can affect many downstream profiles, while exposed receivers define the local peer context each profile sees.

![Pilot sender reach ranking](../figures/pilot_sender_reach_ranking.png)

![Pilot receiver exposure ranking](../figures/pilot_receiver_exposure_ranking.png)

## Exposure Roles In The Pilot Slice

The role-level summary is included only to show how the interpretable mechanism classes differ in exposure sent and received. Positions outside the sender, receiver, bridge, and peripheral definitions are grouped as context positions; the two-dimensional layout should not be read as a literal community-core map.

![Pilot role exposure summary](../figures/pilot_role_exposure_summary.png)

## Interactive Sample-Size Explorer

The static 60-position map is the publication-ready snapshot. For structural inspection, the standalone interactive explorer expands one deterministic nested sample from 30 to 500 nodes. Increasing the sample size only adds nodes; it does not resample the already selected positions.

[Open the interactive exposure map](exposure_network_interactive.html)

## Full-Scale Multi-Resolution Explorer

The 30-500 node explorer is the readable microscope. The full-scale explorer extends the same empirical substrate from 500 positions to all observed PolitiSky24 positions. It uses adaptive node layouts up to 2,000 positions, then switches to a hierarchical macro-community view with community and ego drill-down.

[Open the multi-resolution exposure map](exposure_network_multiresolution.html)

## How This Expands To The Full Exposure Network

This report package reads the canonical graph substrate from `data/exposure_networks/politisky24_bluesky_v1/`. The current first report focuses on a derived 60-position pilot slice, while the same conventions can be extended to full-network community flow, propagation reach, profile-position assignment, and final outcome models over `B`, `BN`, `P`, and `PN`.

## Limitations

- Observed engagement is not a full Bluesky feed-ranking model.
- An edge implies plausible exposure, not confirmed reading or persuasion.
- The 60-position pilot slice is deliberately role-balanced, not a random population sample.
- This report prepares structural network covariates only; it does not estimate cyber-manipulation effects.
"""


def _html_report(summary: dict[str, object]) -> str:
    cards = [
        ("Pilot positions", _fmt(_metric(summary, "pilot_nodes"))),
        ("Directed edges", _fmt(_metric(summary, "pilot_directed_edges"))),
        ("Largest component", _fmt(_metric(summary, "largest_weak_component"))),
        ("Prompt-ready", _fmt(_metric(summary, "all_nodes_prompt_ready"))),
    ]
    card_html = "\n".join(
        f'<div class="metric-card"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'
        for label, value in cards
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Directed Exposure Network Substrate</title>
  <style>
    :root {{
      --ink: #1f2430;
      --muted: #6f768a;
      --grid: #e6e8f0;
      --panel: #ffffff;
      --surface: #fcfcfd;
      --blue: #a3befa;
      --orange: #f0986e;
    }}
    body {{
      margin: 0;
      background: var(--surface);
      color: var(--ink);
      font-family: Inter, Aptos, "Segoe UI", Arial, sans-serif;
      line-height: 1.55;
    }}
    main {{
      width: min(1180px, calc(100vw - 48px));
      margin: 0 auto;
      padding: 42px 0 64px;
    }}
    h1 {{
      font-size: 40px;
      line-height: 1.05;
      margin: 0 0 18px;
      letter-spacing: 0;
    }}
    h2 {{
      font-size: 24px;
      margin: 42px 0 10px;
    }}
    p, li {{
      font-size: 16px;
      color: #303746;
    }}
    .summary {{
      border-left: 5px solid var(--orange);
      background: #fff7f0;
      padding: 18px 22px;
      border-radius: 8px;
      margin: 22px 0;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 22px 0 28px;
    }}
    .metric-card {{
      background: var(--panel);
      border: 1px solid var(--grid);
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .metric-card span {{
      display: block;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
      font-weight: 700;
    }}
    .metric-card strong {{
      display: block;
      margin-top: 6px;
      font-size: 28px;
      line-height: 1;
    }}
    code, pre {{
      font-family: "SF Mono", Menlo, Consolas, monospace;
    }}
    pre {{
      background: #f4f5f7;
      border: 1px solid var(--grid);
      border-radius: 8px;
      padding: 16px;
      overflow-x: auto;
    }}
    figure {{
      margin: 24px 0 36px;
      background: var(--panel);
      border: 1px solid var(--grid);
      border-radius: 10px;
      padding: 12px;
    }}
    figure img {{
      width: 100%;
      height: auto;
      display: block;
    }}
    figcaption {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 8px;
    }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      margin: 14px 0 24px;
      background: var(--panel);
    }}
    .data-table th, .data-table td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--grid);
      text-align: left;
    }}
    .data-table th {{
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 11px;
    }}
    .role-definitions {{
      display: grid;
      grid-template-columns: minmax(180px, 0.35fr) 1fr;
      gap: 10px 18px;
      background: var(--panel);
      border: 1px solid var(--grid);
      border-radius: 10px;
      padding: 18px;
      margin: 18px 0 30px;
    }}
    .role-definitions dt {{
      font-weight: 800;
      color: var(--ink);
      text-transform: lowercase;
    }}
    .role-definitions dd {{
      margin: 0;
      color: #303746;
    }}
    @media (max-width: 840px) {{
      .metric-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      h1 {{ font-size: 32px; }}
    }}
  </style>
</head>
<body>
<main>
  <h1>Directed Exposure Network Substrate</h1>
  <section class="summary">
    <p><strong>This report isolates the empirical directed exposure-network substrate.</strong> The first inspection window is the validated 60-position pilot slice, but the folder is structured to extend to the full PolitiSky24 exposure network.</p>
  </section>
  <div class="metric-grid">{card_html}</div>

  <h2>Exposure Definition And Edge Construction</h2>
  <p>The graph is derived from observed engagement events. If a source user liked, reposted, or quoted a target user, the target user was plausibly visible to the source user.</p>
  <pre>exposure_raw_weight = {INTERACTION_WEIGHTS["Like"]:.2f} * Like
                    + {INTERACTION_WEIGHTS["Repost"]:.2f} * Repost
                    + {INTERACTION_WEIGHTS["Quote"]:.2f} * Quote</pre>
  <p>Repeated interactions are summed, log-compressed, and normalized to <code>[0, 1]</code>. The local edge columns preserve direction: <code>source_position_id</code> is the visible target; <code>target_position_id</code> is the exposed source.</p>
  {_img("exposure_direction_and_formula", "Direction and weighting formula for the exposure graph.")}

  <h2>The 60-Position Pilot Slice As A Directed Exposure Map</h2>
  <p><strong>A node is an observed user position.</strong> A directed edge is a weighted plausible exposure relation from a visible target to the source user who engaged with that target. In the figure, node size follows full-network outgoing visibility, edge width/opacity follows normalized exposure weight, and node identifiers are intentionally omitted to keep the figure publication-ready.</p>
  {_img("pilot_directed_exposure_network", "Directed exposure network for the 60-position pilot slice.")}
  <p>The color legend uses interpretable exposure-mechanism roles. Positions outside the sender, receiver, bridge, and peripheral definitions are grouped as context positions; the two-dimensional layout should not be read as a literal community-core map.</p>
  {_role_definitions_html()}

  <h2>Sender Reach, Receiver Exposure, And Asymmetry</h2>
  <p>The same position can be a strong sender, strong receiver, both, or neither. This distinction is central for later tests of whether private susceptibility interacts with empirical exposure-network position.</p>
  {_img("pilot_sender_reach_ranking", "Top sender-reach positions in the pilot slice.")}
  {_img("pilot_receiver_exposure_ranking", "Top receiver-exposure positions in the pilot slice.")}
  {_table(TABLES_DIR / "top_pilot_senders.csv", rows=8)}
  {_table(TABLES_DIR / "top_pilot_receivers.csv", rows=8)}

  <h2>Exposure Roles In The Pilot Slice</h2>
  <p>The role-level summary is included only to show how the interpretable mechanism classes differ in exposure sent and received. It is a compact structural check, not an additional model or outcome analysis.</p>
  {_img("pilot_role_exposure_summary", "Role-level exposure sent and received inside the pilot slice.")}
  {_table(TABLES_DIR / "pilot_role_summary.csv", rows=12)}

  <h2>Interactive Sample-Size Explorer</h2>
  <p>The static 60-position map is the publication-ready snapshot. For structural inspection, the standalone interactive explorer expands one deterministic nested sample from 30 to 500 nodes. Increasing the sample size only adds nodes; it does not resample the already selected positions.</p>
  <p><a href="exposure_network_interactive.html">Open the interactive exposure map</a></p>

  <h2>Full-Scale Multi-Resolution Explorer</h2>
  <p>The 30-500 node explorer is the readable microscope. The full-scale explorer extends the same empirical substrate from 500 positions to all observed PolitiSky24 positions. It uses adaptive node layouts up to 2,000 positions, then switches to a hierarchical macro-community view with community and ego drill-down.</p>
  <p><a href="exposure_network_multiresolution.html">Open the multi-resolution exposure map</a></p>

  <h2>How This Expands To The Full Exposure Network</h2>
  <p>This report package reads the canonical graph substrate from <code>data/exposure_networks/politisky24_bluesky_v1/</code>. The current first report focuses on a derived 60-position pilot slice, while the same conventions can be extended to full-network community flow, propagation reach, profile-position assignment, and final outcome models over <code>B</code>, <code>BN</code>, <code>P</code>, and <code>PN</code>.</p>

  <h2>Limitations</h2>
  <ul>
    <li>Observed engagement is not a full Bluesky feed-ranking model.</li>
    <li>An edge implies plausible exposure, not confirmed reading or persuasion.</li>
    <li>The 60-position pilot slice is deliberately role-balanced, not a random population sample.</li>
    <li>This report prepares structural network covariates only; it does not estimate cyber-manipulation effects.</li>
  </ul>
</main>
</body>
</html>
"""


def build_report() -> dict[str, object]:
    ensure_dirs()
    input_manifest = prepare_inputs()
    summary = analyze_network()
    figures = make_figures()
    interactive = build_interactive_exposure_map()
    multiresolution = build_multiresolution_exposure_map()

    markdown = _markdown_report(summary)
    html_report = _html_report(summary)
    markdown_path = REPORTS_DIR / "exposure_network_report.md"
    html_path = REPORTS_DIR / "exposure_network_report.html"
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(html_report, encoding="utf-8")

    manifest = {
        "reports": {
            "markdown": artifact_path(markdown_path),
            "html": artifact_path(html_path),
        },
        "inputs": input_manifest,
        "summary": summary,
        "figures": figures,
        "interactive": interactive,
        "multiresolution": multiresolution,
        "root": artifact_path(ROOT),
    }
    write_json(REPORTS_DIR / "analysis_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2))
