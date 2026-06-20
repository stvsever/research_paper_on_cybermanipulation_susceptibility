# Test Run 5: Single-domain, reduced-profile reference (interpretable individual layer)

Run 5 keeps the run-4 exposure-network methodology (conformity-calibrated BN/PN prompts, baseline-polarity anchoring guard, direct sender-reach position metric, receiver-level mechanism test) and adds two changes that make the INDIVIDUAL layer interpretable. It is a small, cheap reference run.

| Component | Value |
|-----------|-------|
| Scenarios | 60, sampled (seed 505) from the 10,000-row integrated production set, restricted to one opinion parent cluster |
| Focus domain | `Information_Integrity_And_Platforms` (18 directional leaves) |
| Leaf measurements | 1,080 (per profile x opinion leaf) |
| Profile | reduced from ~526 to ~336 traits (about 36 percent dropped) |
| Simulation model | `deepseek/deepseek-v4-flash` through OpenRouter |
| Layers | individual + empirical exposure-network (PolitiSky24) |
| LLM calls | ~245 (60 B + 60 BN + 60 P + 60 PN + a few JSON repairs), 0 PN fallbacks |

Reproduce: `bash scripts/tests/run_5.sh` (network on by default; `--no-network` for the individual layer only; `--verbose` for a live monitor).

## Why this run differs from run 4 (methodological fixes for the individual layer)

In run 4 the moderator models were uninterpretable: out-of-sample CV-R2 near zero, no significant moderators, and the largest nominal feature contributions were gamification user-types with no political meaning. Two causes and fixes:

1. **The profile was over-comprehensive.** The pre-built profiles carry ~526 traits across many overlapping taxonomies, which both dilutes what the agent can condition on and leaves the moderator models under-determined (features greatly exceed profiles). Stage 01 now drops a curated set of redundant or low-relevance subtrees (the HEXACO, Eysenck and Hexad "user types" personality taxonomies, which duplicate or are unrelated to the Big Five, plus the goals, values, perceived-safety/legal, criminal-record, administrative and reproductive subtrees), keeping the research core: comprehensive demographics and socioeconomics, the Big Five, the full political-psychology battery (ideology, GAL/TAN, libertarian/authoritarian, nationalism, populism, system justification, moral foundations), religion and digital/media literacy. This is a filter on the already-sampled profiles (`--profile-skip-subtrees`, with a curated default); it does not re-sample, and it is applied in one place so the dropped traits leave both the agent prompt and every downstream analysis.
2. **The attack axis was only pooled.** The conditional susceptibility estimator now carries both the pooled task and per-DISARM-Execute-tactic tasks, so a specific attack vector can be selected in the dashboard rather than only the pooled view.

## Headline results (60 profiles, 1,080 leaf measurements)

- **Individual layer works and is now interpretable.** Mean AE_private = +16.8; 95.5 percent of measurements moved toward the attacker's goal. Between-profile SD of mean AE = 10.3 (range 0 to +65.8), confirming strong inter-individual heterogeneity. The delta identities all hold exactly.
- **Moderators now make sense.** The hierarchical variance decomposition attributes about 60.7 percent of the explained moderation to the political-psychology block (was buried under redundant personality taxonomies before). In the curated multivariate model, openness to experience is a significant moderator (b = +2.78, p = 0.030, bootstrap 95 percent CI excludes 0) and neuroticism is positive with a bootstrap CI excluding 0 (b = +3.15); both directions are theory-consistent (openness and emotional reactivity raise persuadability). The full ~160-feature ridge still overfits at n = 60 (negative CV-R2), so the curated model and the variance decomposition are the operative evidence at this sample size.
- **Most vs least movable opinions.** Most movable: Government Distrust of Elections (+19.4), Freedom from Content Moderation Preference (+18.1), Counter-Disinformation Agency Support (+18.1). Least movable: the technical platform-governance items (recommender-system choice, cross-platform researcher data access, whistleblower protections).
- **Network layer still works.** Mean post-network increment = +8.6 (82 percent amplifying); the receiver-level peer-position pull confirms the propagation mechanism (r = +0.24, n = 1,080, monotonic); BN is clean (increment SD 69); about 15.6 scored peers per measurement on the single dense domain. The condition-level placement view (1 domain gives only a handful of conditions) is underpowered and is not the network test; the receiver-level mechanism is.

## Output layout

| Directory | Contents |
|-----------|----------|
| `config/` | run configuration |
| `logs/` | per-stage logs + console and launch logs |
| `provenance/` | raw LLM calls (~245 request/response records) + run manifest |
| `stage_outputs/` | canonical per-stage data (all B/BN/P/PN assessments, contexts, four-state deltas with raw + guarded values) |
| `analysis/` | `datasets/`, `sem/`, `report/` (effect tables, conditional susceptibility index incl. per-tactic tasks, moderation) |
| `visuals/` | `dashboard_results.html`, `figures/`, `embeddings/`, `network_exposure_analysis/` |
| `publication/` | `publication_assets/`, `paper/` |

Key entry points: the interactive dashboard `visuals/dashboard_results.html` (conditional estimator now scoped per Execute tactic); the exposure-network report `visuals/network_exposure_analysis/reports/run_5_network_exposure_report.html`; the four-state long table `stage_outputs/05_compute_effectivity_deltas/network_exposure_long.csv`.
