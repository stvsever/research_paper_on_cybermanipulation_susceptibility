# Test Run 3 — Individual + empirical exposure-network layer (current integrated reference)

Run 3 re-runs the run-2 production design on the current cluster pipeline and additionally runs the empirical exposure-network layer, re-implemented onto this pipeline's logic: cluster-batched opinion evaluation, the DISARM Plan/Prepare/Execute attack triplet, and the full high-resolution production profile/opinion ontologies.

| Component | Value |
|-----------|-------|
| Scenarios | 100, sampled (seed 120, stratified over 7 issue domains) from the 10,000-row integrated production set |
| Each scenario | 1 full profile + 1 near-unique DISARM triplet + 1 opinion parent cluster |
| Ontology source | `src/backend/ontology/separate/production` |
| Simulation model | `deepseek/deepseek-v4-flash` through OpenRouter |
| Layers | individual + empirical exposure-network (PolitiSky24) |
| LLM calls | ~410 (100 B + 100 BN + 100 P + 100 PN + a few JSON repairs) |

Reproduce: `bash scripts/tests/run_3.sh --network` (omit `--network` for the individual layer only; add `--verbose` for a live progress monitor).

## Measurement backbone (four states per opinion-cluster leaf)

| Symbol | Meaning |
|--------|---------|
| B  | private baseline (stage 02) |
| BN | network-exposure baseline, after incoming empirical peer baseline context (stage 02b) |
| P  | private post-attack (stage 04) |
| PN | network-exposure post-attack, after same-condition incoming peer post context (stage 04b) |

`d` = per-leaf adversarial direction. AE_private = (P-B)*d; AE_total_network = (PN-B)*d; PN_increment_effectivity = (PN-P)*d.

## Headline results (100 profiles, 1,517 leaf measurements)

- **Individual layer.** Mean adversarial effectivity AE_private = +18.9; 90.4% of leaf measurements moved toward the attacker's goal; between-profile SD of AE_private = 17.0 (strong inter-individual heterogeneity).
- **Network layer.** Same-condition peer context amplifies the attack: mean PN_increment_effectivity = +6.8; 80.6% of measurements amplified the private post-attack effect; 87.3% of measurements had real incoming peer context.
- **Attack factors.** The DISARM triplet is decomposed into separable factors (the shared 2nd-level Plan/Prepare/Execute tactic node); each factor's marginal contribution is in `visuals/network_exposure_analysis/tables/attack_factor_decomposition.csv`.
- The central-sender amplification correlation (reach vs final effect) is near zero at n=100 and not robustly signed; this pilot gives interpretable means but is not powered for the network-position correlations.

## Output layout

| Directory | Contents |
|-----------|----------|
| `config/` | run configuration |
| `logs/` | per-stage logs + the console log |
| `provenance/` | raw LLM calls + run manifest |
| `stage_outputs/` | canonical per-stage data for post-hoc analysis (all B/BN/P/PN phases) |
| `analysis/` | `datasets/`, `sem/`, `report/` (effect tables, SEM, moderation) |
| `visuals/` | `dashboard_results.html`, `figures/`, `embeddings/`, `embeddings_production/`, `network_exposure_analysis/` |
| `publication/` | `publication_assets/`, `paper/` |

Key entry points: the interactive dashboard `visuals/dashboard_results.html`; the comprehensive exposure-network report `visuals/network_exposure_analysis/reports/run_3_network_exposure_report.html`; the interactive network map `visuals/network_exposure_analysis/network_exposure_interactive.html`; the four-state long table `stage_outputs/05_compute_effectivity_deltas/network_exposure_long.csv`.
