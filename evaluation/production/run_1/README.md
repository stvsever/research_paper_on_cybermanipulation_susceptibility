# Production Run 1: Individual-layer susceptibility at full scale (10,000 scenarios)

This is the production run of the individual (private) susceptibility layer over the
ENTIRE 10,000-row integrated scenario set. It is launched with
`bash scripts/production/run_1.sh`. The empirical exposure-network layer is OFF, so
the whole LLM budget goes to the individual layer at full scale. The run is
deliberately scoped to the raw measurement: it produces the per-scenario, per-leaf
baseline, post-attack and effectivity scores. The post-hoc analyses and the figures
are layered on afterwards from these saved scores.

## Design

| Component | Value |
|-----------|-------|
| Scenarios | all 10,000 rows of the integrated production set (no sub-sampling), stratified across the 7 issue domains (about 1,429 per domain) |
| Selection | `--n-scenarios 10000 --no-max-entropy-subsample`, seed 1001 |
| Domains | all 7 opinion parent clusters |
| Profile | deeply reduced research-core, about 159 features (a 53 percent reduction on the previous 336-feature reduced set, and a 71 percent reduction on the full 540-feature integrated profile) |
| Each scenario | 1 reduced profile + 1 DISARM Plan/Prepare/Execute triplet + 1 opinion parent cluster (all directional leaves) |
| Stages | 01 to 05 only (scenario build, baseline B, attack spec, post-attack P, effectivity deltas) |
| Layers | individual only (no 01b/02b/04b/05b) |
| Simulation model | `deepseek/deepseek-v4-flash` through OpenRouter |
| LLM calls | about 20,000 (10,000 baseline B + 10,000 post-attack P, plus a few JSON repairs); stage 03 is deterministic |

### The reduced profile

The mapped profile is the research-relevant core that the simulating agent conditions
on. It KEEPS:

- the full hierarchical Big Five (all five traits and their facets, about 45 features);
- the core demographic markers: sex, age, gender identity, citizenship, country of
  birth, relationship status, and self-identified ethnicity;
- the complete political-psychology battery: the political-profile inventories
  (right-wing authoritarianism, social dominance, system justification, populism,
  nationalism, collective narcissism, political trust and efficacy, and so on), the
  two-axis ideological dimensions, the GAL-TAN and libertarian-authoritarian scales,
  and moral-foundations theory.

It DROPS the high-cardinality taxonomies that are not the susceptibility core: the
political-participation taxonomy, the socioeconomic / employment / housing / migration
/ education / household life-circumstance taxonomies, the religion-spirituality
sub-tree, and the over-detailed identity spectra (sexual-orientation spectrum,
sex-characteristics biology, gender modality / expression / pronouns, relationship
structure, developmental-psychology stages). The redundant personality taxonomies
(HEXACO, Eysenck, Hexad) and the goals / values / safety / criminal / administrative
subtrees were already dropped in the first reduction pass.

## What is measured and saved

Two private opinion states per opinion-cluster leaf:

| Symbol | Meaning |
|--------|---------|
| B | private baseline (stage 02) |
| P | private post-attack (stage 04) |

with `d` the per-leaf adversarial direction. The primary outcome is
`adversarial_effectivity = (P - B) * d` (positive means the opinion moved toward the
attacker's goal). The signed change `delta = P - B` and its magnitude `abs_delta` are
also kept.

### Lean storage

The run is configured for a small on-disk footprint while keeping every score:

- Stage 05 writes only the compact CSV delta tables; the large redundant JSONL mirrors
  (`sem_long_encoded.jsonl`, `sem_long_rows.jsonl`, the profile-embedded
  `effectivity_deltas.jsonl`) are skipped (`--lean-storage`).
- Raw LLM provenance is written during the run only to drive the live progress monitor
  and is deleted at the end, so the final run does not retain it.
- No `_report` / `_report_assets` and no figures are generated (stages 06 to 09 are
  not run).

Nothing is lost for later analysis: the full source content of any scenario (its
complete profile, attack triplet and opinion cluster) is recoverable by joining
`src/backend/pipeline/separate/01_create_scenarios/samples/02_integrated/integrated_scenarios_10000.jsonl`
on `scenario_id`.

### Saved data

| Path | Contents |
|------|----------|
| `stage_outputs/05_compute_effectivity_deltas/sem_long_raw.csv` | the canonical per-(scenario, leaf) table: `scenario_id`, `opinion_leaf`, `opinion_domain`, `baseline_score` (B), `post_score` (P), `delta_score`, `abs_delta_score`, `adversarial_effectivity`, `adversarial_direction`, and the DISARM Plan/Prepare/Execute tactics + complexity tier, plus the reduced profile columns |
| `stage_outputs/05_compute_effectivity_deltas/sem_long_encoded.csv` | the same rows with model-ready encodings (z-scores, fixed-effect dummies) for downstream modelling |
| `stage_outputs/05_compute_effectivity_deltas/profile_level_effectivity.csv` | per-profile rollup of the effect |
| `stage_outputs/05_compute_effectivity_deltas/delta_summary.json` | run-level summary statistics |
| `stage_outputs/02_assess_baseline_opinions/` | the baseline B assessments per scenario |
| `stage_outputs/04_assess_post_attack_opinions/` | the post-attack P assessments per scenario |
| `stage_outputs/01_create_scenarios/` | the scenario records and the sampling / compatibility audit |
| `config/`, `logs/` | the run configuration and the console log |

The identity `adversarial_effectivity = (post_score - baseline_score) * adversarial_direction`
holds exactly on every row, and the true analysis scenario is recovered from the
`scenario_id` prefix (`sem_long` encodes it as `<scenario>__<leaf>`).

## Reproduce

```bash
bash scripts/production/run_1.sh            # quiet
bash scripts/production/run_1.sh --verbose  # live progress monitor (per-stage call counts, rate, ETA)
```

The launcher reads `OPENROUTER_API_KEY` from `.env` and checks the projected
OpenRouter budget before spending. Failed calls auto repair (JSON) and retry, then
fall back deterministically, so the run always completes with full per-scenario
coverage.

## Notes

- This run supersedes an earlier 1,000-scenario smoke run that used the same layout.
- The analyses and visuals (conditional susceptibility, block-wise family model,
  multilevel moderation, inferential tests, paper figures) are intentionally not run
  here and are produced separately from the saved scores.
