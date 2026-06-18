# PolitiSky24 Bluesky Exposure Network v1

This folder is the stable pipeline-facing exposure-network substrate for the local lab pipeline.
It is derived from the PolitiSky24 Bluesky engagement graph and replaces draft-analysis paths under `LOCAL_DRAFTS`.

## Edge Semantics

A directed edge is stored as:

```text
source_position_id -> target_position_id
```

Meaning: the source position is a visible peer whose output is plausibly exposed to the target position.
The upstream raw interpretation is: if a source user liked, reposted, or quoted a target user, the target user's output was plausibly visible to the source user. For the pipeline, the stored orientation is normalized to visible peer -> exposed receiver.

## Interaction Weights

```text
exposure_raw_weight = 0.35 * Like + 0.80 * Repost + 0.90 * Quote
```

Repeated interactions are summed, log-compressed, and normalized to `[0, 1]` upstream.

## Pipeline Use

- `assignment_positions.csv` assigns generated profiles to empirical positions in Stage 01b.
- `edges_full.csv` defines the full empirical exposure neighborhood.
- `edges_prompt_top30.csv` supports bounded prompt exemplar selection.
- `node_metrics.csv`, `neighborhood_metrics.csv`, and `propagation_metrics.csv` provide network covariates for analysis.

## Limitations

Observed engagement is a plausible exposure signal, not proof of reading, trust, friendship, or communication frequency.
The graph is empirical exposure structure, not profile similarity.

## Row Counts

{
  "nodes.csv": 8483,
  "edges_full.csv": 1179750,
  "edges_prompt_top30.csv": 208926,
  "node_metrics.csv": 8483,
  "neighborhood_metrics.csv": 8483,
  "propagation_metrics.csv": 8483,
  "assignment_positions.csv": 8483
}
