# Directed Exposure Network Substrate Report And Visualizations

This folder is the independent report and visualization workspace for the empirical PolitiSky24 exposure-network substrate. It is separate from the runtime `network_exposure` utility package used by the pipeline.

It starts with a derived 60-position pilot slice as a readable inspection window, but the folder is not limited to 60 nodes. It reads the canonical graph package from `data/exposure_networks/politisky24_bluesky_v1/`; it does not keep a second copy of graph-scale input data.

## Core Semantics

Directed edge:

```text
TargetUserId -> SourceUserId
```

If a source user liked, reposted, or quoted a target user, the target user was plausibly visible to the source user.

Interaction weighting:

```text
exposure_raw_weight = 0.35 * Like + 0.80 * Repost + 0.90 * Quote
```

Repeated interactions are summed, log-compressed, and normalized to `[0, 1]`.

Local edge columns preserve the same interpretation:

- `source_position_id`: original `TargetUserId`, the visible target/content source.
- `target_position_id`: original `SourceUserId`, the exposed user who engaged.

## Rebuild

Run from this folder:

```bash
env UV_CACHE_DIR=/tmp/uv-cache-politisky24 MPLCONFIGDIR=/tmp/matplotlib-politisky24 \
  uv run --with pandas --with networkx --with scipy --with matplotlib --with seaborn \
  python scripts/build_exposure_network_report.py
```

Main outputs:

- `reports/exposure_network_report.html`
- `reports/exposure_network_interactive.html`
- `reports/exposure_network_multiresolution.html`
- `reports/exposure_network_report.md`
- `figures/pilot_directed_exposure_network.png`
- `figures/pilot_directed_exposure_network.svg`
- `data/derived/pilot_nodes.csv`
- `data/derived/pilot_edges.csv`
- `data/derived/interactive_sample_nodes.csv`
- `data/derived/interactive_sample_edges.csv`
- `data/derived/interactive_sample_layouts.csv`
- `data/derived/interactive_sample_quality.csv`
- `data/derived/interactive_sample_summary.csv`
- `data/derived/multiresolution_nodes.csv`
- `data/derived/multiresolution_edges_backbone.csv`
- `data/derived/multiresolution_macro_communities.csv`
- `data/derived/multiresolution_macro_flows.csv`
- `data/derived/multiresolution_level_layouts.csv`
- `data/derived/multiresolution_community_lenses.csv`
- `data/derived/multiresolution_community_lens_edges.csv`
- `data/derived/multiresolution_ego_edges.csv`
- `data/derived/multiresolution_quality.csv`
- `tables/role_definitions.csv`

The `reports/`, `figures/`, and compact `tables/` directories are retained as reader-facing artifacts. Rebuilt `data/derived/` files are local intermediates and are ignored by Git.

## Script Stack

- `scripts/prepare_exposure_network_data.py`: validates the canonical graph package and derives the local 60-position pilot slice from `assignment_positions.csv` plus induced `edges_prompt_top30.csv` edges.
- `scripts/analyze_exposure_network.py`: computes local pilot-slice directed exposure metrics.
- `scripts/visualize_exposure_network.py`: renders publication-style figures for the directed network, formula legend, sender/receiver rankings, and role exposure summary.
- `scripts/build_interactive_exposure_map.py`: builds a deterministic nested 30-500 node interactive exposure-map explorer with adaptive per-step layouts, readable edge backbones, and sample-quality diagnostics.
- `scripts/build_multiresolution_exposure_map.py`: builds the 500-to-all-node hierarchical explorer. Individual node-link rendering is used up to 2,000 positions; high-resolution views switch to macro-community flow with community and ego drill-down.
- `scripts/build_exposure_network_report.py`: runs the full local analysis and builds the report.

## Current Scope

This report is structural only. It does not use profiles, opinions, attacks, baseline scores, post scores, or synthetic susceptibility values.

The multi-resolution explorer distinguishes:

- `observed_all`: all observed PolitiSky24 positions in the local graph.
- `prompt_ready`: positions with enough incoming exposure peers for prompt-context construction.
- `analysis_sample`: deterministic nested prefixes used for readable inspection.

The all-node view is not a literal all-edge hairball. At high resolution, the valid visual object is macro-community exposure flow, with local inspection through selected-community and ego exposure lenses.
