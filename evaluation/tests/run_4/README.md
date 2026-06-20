# Test Run 4: Concentrated 2-domain exposure-network reference (network layer working end to end)

Run 4 is the exposure-network reference run. It keeps the run-3 integrated production design (full profiles, DISARM Plan/Prepare/Execute triplets, cluster-batched opinion evaluation) but fixes the methodology so the empirical exposure-network layer actually carries signal, and it concentrates the scenario budget into two issue domains so the network is dense enough for the position analyses.

| Component | Value |
|-----------|-------|
| Scenarios | 200, sampled (seed 404) from the 10,000-row integrated production set, restricted to two opinion parent clusters |
| Focus domains | `Information_Integrity_And_Platforms`, `Democratic_Resilience_And_Institutions` (100 scenarios each) |
| Each scenario | 1 full profile + 1 near-unique DISARM triplet + 1 opinion parent cluster (all its directional leaves) |
| Leaf measurements | 4,300 (per profile x opinion leaf) |
| Ontology source | `src/backend/ontology/separate/production` |
| Simulation model | `deepseek/deepseek-v4-flash` through OpenRouter |
| Layers | individual + empirical exposure-network (PolitiSky24) |
| LLM calls | 818 (200 B + 200 BN + 200 P + 200 PN + a few JSON repairs), 1 BN fallback, 0 PN fallbacks |

Reproduce: `bash scripts/tests/run_4.sh` (network layer on by default; add `--verbose` for a live monitor, `--no-network` for the individual layer only).

## Why this run differs from run 3 (methodological fixes)

Run 3 showed almost no network effect and a wrong-signed position correlation. The causes and fixes:

1. **Sparsity.** Spreading 100 scenarios over 7 domains left roughly 3.5 scored peers per measurement and 1 to 5 profiles per `domain x Execute tactic` condition, so the position statistics were pure noise. Run 4 concentrates 200 scenarios into 2 domains, lifting this to about 19.7 scored peers per measurement and roughly 16 profiles per condition. A hard `--network-scenario-cap` of 500 keeps the production run bounded; above it a whole-word `media` keyword filter over the DISARM triplet selects the subset congruent with the social-media exposure substrate.
2. **The network mechanism was suppressed.** The BN/PN prompts over-anchored, so the agent ignored peer context (median move of 0). The prompts were rebalanced for calibrated social influence (partial movement toward a clear exposure-weighted peer consensus, scaled by consensus strength and the profile's conformity-relevant traits). The agent now responds: median post-attack move of 12 score points, 99 percent of baseline measurements show some peer adjustment.
3. **Baseline-exposure polarity noise.** On ambiguously-signed leaves the BN agent re-derived the item and flipped the scale (`bn_increment` SD of 301). The BN prompt now preserves the private-baseline polarity, and a deterministic anchoring guard in stage 05 holds confident reversals at the baseline (raw values are kept in the `*_raw` columns). `bn_increment` SD drops to 57 with 0 guard corrections needed on this run.
4. **Operationalization.** Network position is now direct exposure sender reach `outgoing_visibility_weight = sum_i w_(j->i)` (the sender-side influence the hypotheses are about), not eigenvector centrality.
5. **The position test was the wrong test.** The condition-level alignment is confounded by an amplification ceiling. The clean, primary test is the receiver-level peer-position pull (below), measured where the opinion is elicited.

## Measurement backbone (four states per opinion-cluster leaf)

| Symbol | Meaning |
|--------|---------|
| B  | private baseline (stage 02) |
| BN | network-exposure baseline, after incoming empirical peer baseline context (stage 02b) |
| P  | private post-attack (stage 04) |
| PN | network-exposure post-attack, after same-leaf incoming peer post context (stage 04b) |

`d` = per-leaf adversarial direction. AE_private = (P-B)*d; AE_total_network = (PN-B)*d; PN_increment_effectivity = (PN-P)*d; net_social_amplification = ((PN-P)-(BN-B))*d.

## Headline results (200 profiles, 4,300 leaf measurements)

- **Individual layer.** Mean AE_private = +18.2; 95 percent of leaf measurements moved toward the attacker's goal. The delta identities `AE_private == (P-B)*d`, `AE_total_network == (PN-B)*d` and the SEM `adversarial_effectivity == (post-baseline)*d` all hold exactly.
- **Network layer works.** Mean PN_increment_effectivity = +7.8; 80 percent of measurements amplify the private post-attack effect (run 3 was near zero and slightly negative). Mean final AE_total_network = +26.0 (91 percent toward goal).
- **Primary mechanism (confirms the hypothesis).** Exposure to peers whose consensus sits further toward the attacker's goal than the profile drives further adversarial movement: the receiver-level correlation between the direction-aware peer-position pull and the post-network increment is r = +0.34, monotonic across bins (mean increment of +15.8, 96 percent amplifying, when peers pull toward the goal; mean of -1.3 when peers resist). This is the social-conformity mechanism by which the exposure network propagates the attack.
- **Position placement (ecological, ceiling-confounded).** The condition-level centrality-susceptibility alignment correlates negatively with the mean increment because conditions already saturated by the individual attack have little network headroom. This is reported with that caveat; the receiver-level mechanism above is the network test.

## Output layout

| Directory | Contents |
|-----------|----------|
| `config/` | run configuration |
| `logs/` | per-stage logs + the console and launch logs |
| `provenance/` | raw LLM calls (818 request/response records) + run manifest |
| `stage_outputs/` | canonical per-stage data for post-hoc analysis (all B/BN/P/PN assessments, contexts and the four-state long table with raw + guarded deltas) |
| `analysis/` | `datasets/`, `sem/`, `report/` (effect tables, SEM, moderation) |
| `visuals/` | `dashboard_results.html`, `figures/`, `embeddings/`, `embeddings_production/`, `network_exposure_analysis/` |
| `publication/` | `publication_assets/`, `paper/` |

Key entry points: the interactive dashboard `visuals/dashboard_results.html`; the comprehensive exposure-network report `visuals/network_exposure_analysis/reports/run_4_network_exposure_report.html`; the four-state long table `stage_outputs/05_compute_effectivity_deltas/network_exposure_long.csv`.
