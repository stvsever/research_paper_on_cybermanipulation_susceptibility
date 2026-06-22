# Production Run 2: Network Exposure Layer

Production run 2 is the production run for the empirical exposure-network layer
of the cybermanipulation susceptibility study. It complements production run 1,
which focused on individual/private susceptibility over a broader scenario
space. Run 2 deliberately uses a smaller, fixed full-factorial design so that
the same generated profiles occupy the same empirical network positions across
all selected opinion and attack conditions.

The final scenario panel is:

```text
100 profiles x 7 opinions x 5 attacks = 3,500 scenarios
```

Run 2 contains two complementary network analyses. The main production run keeps
the profile-position assignment fixed across all conditions and remains the
source of truth for the full fixed-factorial pipeline outputs. The branch-local
`counterfactual_alignment_gradient/` analysis reuses the same profiles,
opinions, attacks, empirical graph, and private post-attack measurements, but
evaluates Stage `04b` with condition-specific profile-position assignments to
provide the direct experimental mechanism test for H6/H7.

This design supports the central network question: whether private
susceptibility becomes more consequential when susceptible or resilient profiles
occupy high-reach sender positions in the empirical exposure graph.

## Design Summary

| Component | Value |
| --- | --- |
| Primary layer | Empirical exposure-network layer |
| Scenario unit | `profile x opinion x attack` |
| Profiles | `100` deterministic profiles |
| Opinions | `7` directional production opinion leaves |
| Attacks | `5` social-media-relevant test attack leaves |
| Scenario count | `3,500` |
| Opinion x attack cells | `35`, each containing the same `100` profile-network positions |
| Exposure graph | `politisky24_bluesky_v1` |
| Profile-position assignment | `ranked_seeded_profiles`, seed `120` |
| Main run assignment | Fixed profile-position assignment across all `35` conditions |
| H6/H7 mechanism branch | `counterfactual_alignment_gradient/` |
| H6/H7 branch assignment | Condition-specific counterfactual profile-position assignment |
| Pipeline stages | `01` through `08b`; report build stage `09` is not run |
| Model | `deepseek/deepseek-v4-flash` via OpenRouter |
| Temperature | `0.15` |
| Max repair iterations | `2` |
| Raw LLM provenance | Generated during execution but omitted from the curated official package |
| Run launcher | `bash scripts/production/run_2.sh full` |

## Official Repository Package

This folder is the curated official repository package for production run 2. It
keeps the methodological configuration, manifests, logs, compact final tables,
network-analysis reports, figures, supplementary tables, and publication-facing
artifacts needed to inspect and reuse the run.

The full local execution workspace is larger and remains outside the official
repository. The curated package intentionally omits raw LLM request/response
provenance, smoke and structural preflight outputs, failed sandbox attempts,
Python caches, duplicated top-level output copies, and large raw JSONL mirrors
such as `scenarios_with_*.jsonl` and `live_results.jsonl`. The committed Stage
`05` CSV files are the compact analysis tables for downstream statistical work.

## Scientific Aim

The run estimates how inter-individual susceptibility profiles interact with
empirically grounded exposure-network position to amplify or attenuate
cybermanipulation effects across a population.

Operationally, the run asks:

> For a given opinion target and attack vector, is the final network-exposed
> attack effect larger when profiles with high direct sender reach are also
> highly susceptible to the attack?

The empirical exposure network is fixed before opinion outcomes are measured.
Network position is therefore a design feature, not an outcome of simulated
susceptibility.

## Ontology Configuration

Run 2 uses a run-local mixed ontology bundle:

```text
evaluation/production/run_2/config/ontology_mixed/
  PROFILE/profile.json
  ATTACK/attack.json
  OPINION/opinion.json
```

The pipeline accepts a single `--ontology-root`, so the mixed bundle makes the
component choices explicit and reproducible without changing global ontology
roots.

| Axis | Source | Schema version | SHA256 |
| --- | --- | --- | --- |
| `PROFILE` | Current codebase test profile ontology | `v4-test-run-1-production` | `cf14270c41d8a823e379ab482783c58f53e3a8ad279a26a56e714ccaa02883d6` |
| `ATTACK` | Current codebase test attack ontology | `v4-test-run-1-test` | `5faf9cab3d214dc21b62d848ac65a6f57fdba0f9c707bb638d6137ccf47f384c` |
| `OPINION` | Updated official production opinion ontology | `v4-test-run-1-production-r3` | `d9f43a9723b35b10b13cb9ea7e2a53bf1eb1da4ad558773783c499de6a971333` |

The mixed ontology choice is deliberate:

- the test profile ontology preserves a tractable and validated profile space;
- the test attack ontology preserves compatibility with the validated attack
  execution layer while controlling cost;
- the production opinion ontology aligns run 2 with the paper-level issue
  position space used by the broader project.

## Scenario Construction

Scenarios are generated as a fixed full factorial over the selected profile,
opinion, and attack panels:

```text
PROFILE x OPINION x ATTACK
```

This is necessary for the network layer. Stages `02b` and `04b` compare each
profile with same-condition peers. Random row-level subsampling would break the
condition-specific peer set and would make network comparisons depend on which
rows happened to be sampled. In this design, every opinion x attack condition
contains all `100` profile-network positions.

| Axis | Selection rule | Count | Role |
| --- | --- | ---: | --- |
| `PROFILE` | Deterministic diverse profile panel, seed `120` | `100` | Each profile is assigned one empirical exposure-network position and reused across all conditions. |
| `OPINION` | Explicit full paths from the production `Issue_Position_Taxonomy` | `7` | Each opinion defines the policy target and adversarial direction. |
| `ATTACK` | Explicit social-media-relevant attack leaves from the test attack ontology | `5` | Each attack defines the manipulation mechanism applied to every profile x opinion cell. |
| Scenario panel | Full factorial crossing | `3,500` | Balanced repeated-measures panel for private and network-layer comparisons. |

Stage `01` records `0` compatibility exclusions for this design.

## Statistical Design And Scope

Run 2 is a fixed full-factorial experiment over a purposive ontology subset. It
is not a random sample from all possible profiles, opinions, attacks, platforms,
or networks. Its strength is internal validity within the selected state space:
profile composition, network position, opinion target, and attack condition are
explicitly controlled by design.

The same `100` profiles are repeatedly measured across all `35` opinion x attack
conditions. This supports within-profile and within-position comparison across
conditions and prevents the H6/H7 mechanism tests from being driven by changing
profile samples. Each condition-level estimate aggregates over the same `100`
assigned network positions.

The condition-level network tests use `35` planned condition cells. This is
adequate for a parsimonious directional mechanism test and for strong
visualization of the alignment mechanism, but it is not an unrestricted
high-dimensional inference problem. Claims are therefore framed around the
planned alignment metric, the selected ontology panel, and the fixed empirical
exposure graph.

The fixed-position main run is the appropriate design for validating the full
network pipeline and for estimating network effects under one deterministic
empirical profile-position assignment. However, H6/H7 concern the mechanism by
which network-wide effects change when susceptible or resilient profiles occupy
high-reach sender positions. The branch-local counterfactual alignment-gradient
analysis therefore introduces a targeted experimental manipulation of
profile-position assignment while holding the profiles, opinions, attacks,
private post-attack scores, and empirical graph fixed.

## Opinion Panel

Opinions are selected from the production `Issue_Position_Taxonomy`. The seven
selected leaves cover the directionally encoded issue-position domains and
include both adversarial directions. Full paths are used because some leaf labels
are duplicated in the production opinion ontology.

| Domain | Opinion path | Direction | Rationale |
| --- | --- | ---: | --- |
| `Defense_And_National_Security` | `Issue_Position_Taxonomy > Defense_And_National_Security > Alliance_Commitment_Support` | `-1` | Preserves a validated anchor and targets alliance cohesion. |
| `Foreign_Policy_And_Geopolitics` | `Issue_Position_Taxonomy > Foreign_Policy_And_Geopolitics > Unilateralism_Preference` | `+1` | Captures coalition-fragmentation pressure through increased unilateralism. |
| `Information_Integrity_And_Platforms` | `Issue_Position_Taxonomy > Information_Integrity_And_Platforms > Trust_In_Mainstream_Journalism` | `-1` | Preserves a validated anchor and targets epistemic trust. |
| `Democratic_Resilience_And_Institutions` | `Issue_Position_Taxonomy > Democratic_Resilience_And_Institutions > Trust_In_Electoral_Process` | `-1` | Captures institutional legitimacy and democratic resilience. |
| `Critical_Infrastructure_And_Energy_Sovereignty` | `Issue_Position_Taxonomy > Critical_Infrastructure_And_Energy_Sovereignty > Energy_Sovereignty_Investment_Support` | `-1` | Captures resilience of energy and critical-infrastructure capacity. |
| `Supranational_And_Regional_Integration` | `Issue_Position_Taxonomy > Supranational_And_Regional_Integration > National_Sovereignty_Priority` | `+1` | Captures pressure toward fragmentation of supranational coordination. |
| `Macroeconomic_And_Fiscal_Policy` | `Issue_Position_Taxonomy > Macroeconomic_And_Fiscal_Policy > Trade_With_Authoritarian_States_Support` | `+1` | Captures economic dependency exposure; full path disambiguates a duplicated leaf label. |

## Attack Panel

Attacks are selected from the current codebase test attack ontology. The panel
is intentionally social-media-relevant because run 2 centers on an empirical
social-media exposure graph.

| Attack leaf | Ontology family | Platform relevance | Rationale |
| --- | --- | --- | --- |
| `Headline_And_Lede_Misframing` | `Claim_Frame_And_Narrative_Manipulation` | Feed cards, link previews, news headlines, short summaries | Captures platform-native headline and lede framing. |
| `Quote_Context_Stripping` | `Claim_Frame_And_Narrative_Manipulation` | Quote posts, snippets, screenshots, clipped context | Represents decontextualized framing without adding compatibility exclusions. |
| `Credentialed_Domain_Persona_Fabrication` | `Source_Identity_And_Legitimacy_Manipulation` | Expert-looking accounts, institutional impersonation, authority cues | Captures source-legitimacy manipulation through credible-looking social accounts. |
| `Repost_Bot_Amplification` | `Amplification_Visibility_And_Attention_Manipulation` | Reposts, artificial engagement, visibility manipulation | Directly matches the graph's repost and visibility logic. |
| `Petition_Astroturf` | `Social_Proof_Network_And_Community_Manipulation` | Manufactured consensus, mobilization signals, coordinated social proof | Captures peer and community pressure that can interact with network exposure. |

## Exposure Network

The empirical exposure graph is `politisky24_bluesky_v1`. A directed edge means:

```text
visible peer profile -> exposed receiver profile
```

In the underlying graph metadata this is recorded as:

```text
source_position_id -> target_position_id
```

The source position is a visible peer whose output was plausibly exposed to the
target position, derived from source-user liking, reposting, or quoting target
user content in the raw platform data.

The edge-weight formula is:

| Interaction | Weight |
| --- | ---: |
| Like | `0.35` |
| Repost | `0.80` |
| Quote | `0.90` |

Each generated profile is assigned to one empirical network position. The same
assignment is reused across every opinion and attack condition, so profile
composition and network position are held constant.

Stage `01b` records the assigned network-role distribution:

| Network role | Count |
| --- | ---: |
| `bridge` | `20` |
| `context_position` | `20` |
| `high_exposure_receiver` | `22` |
| `high_visibility_sender` | `22` |
| `peripheral` | `16` |

## Measurement Backbone

Run 2 preserves four opinion measurements:

| Symbol | Meaning | Pipeline field |
| --- | --- | --- |
| `B` | Private baseline opinion | `baseline_assessment.score` |
| `BN` | Baseline opinion after empirical incoming peer baseline context | `network_exposure_assessment.score` |
| `P` | Private post-attack opinion | `post_attack_assessment.score` |
| `PN` | Post-attack opinion after same-condition empirical incoming peer post-attack context | `post_attack_network_exposure_assessment.score` |

Let `d` be the adversarial direction of the opinion leaf. `+1` means upward
movement is attack-aligned; `-1` means downward movement is attack-aligned.

The core direction-aware quantities are:

| Quantity | Formula | Interpretation |
| --- | --- | --- |
| Private susceptibility | `AE_private = (P - B) * d` | Private attack success for one profile. |
| Baseline network increment | `BN_increment = BN - B` | How pre-attack peer context shifts baseline opinion. |
| Post-network increment | `PN_increment = PN - P` | How post-attack peer context shifts private post-attack opinion. |
| Post-network effectivity | `PN_increment_effectivity = (PN - P) * d` | Whether peer context amplifies or dampens the attack direction. |
| Total network attack effect | `AE_total_network = (PN - B) * d` | Final direction-aware attack effect after private and network exposure. |

The primary individual susceptibility variable is `AE_private`, not raw
`P - B`, because the hypotheses concern movement toward the attacker's goal.

## Hypotheses And Estimands

| Question or hypothesis | Primary estimand | Design requirement | Interpretation |
| --- | --- | --- | --- |
| Core RQ: interaction between susceptibility and empirical exposure-network position | `B`, `BN`, `P`, `PN`, and direction-aware deltas within condition | Same `100` profiles assigned to fixed network positions and crossed with all `7 x 5` opinion-attack cells | Differences across conditions are not confounded by changing sampled profiles or changing network positions. |
| H1: private susceptibility heterogeneity | `AE_private = (P - B) * d` | Full `profile x opinion x attack` panel | Estimates whether private attack susceptibility varies across profiles, opinions, and attack vectors. |
| H2: incoming peer-context amplification | `PN_increment_effectivity = (PN - P) * d` | Complete same-condition peer sets | Tests whether post-attack peer context further amplifies or attenuates attack-aligned movement. |
| H6: central susceptible sender amplification | Positive association between `achieved_alignment_z` and mean `PN_increment_effectivity` / `AE_total_network` | Counterfactual alignment-gradient branch with `35` condition cells, each containing the same `100` profiles under condition-specific assignment | Tests whether network-wide attack effects increase when more susceptible profiles occupy higher-reach sender positions. |
| H7: central resilient sender attenuation | Lower mean network attack effect at negative `achieved_alignment_z` values | Same branch design, with resilient profiles assigned toward higher-reach sender positions in negative-alignment cells | Tests the attenuation side of the same mechanism: high-reach resilient senders reduce network-wide attack propagation. |

For each opinion x attack condition, define:

```text
s_i = AE_private_i
r_i = outgoing_visibility_weight_i
w_i = r_i / sum_j r_j
```

Then:

```text
unweighted_susceptibility = mean_i(s_i)
sender_reach_weighted_susceptibility = sum_i(w_i * s_i)
sender_reach_susceptibility_alignment =
  sender_reach_weighted_susceptibility - unweighted_susceptibility
```

The scale-free version is:

```text
sender_reach_susceptibility_alignment_z =
  sender_reach_susceptibility_alignment / sd_i(s_i)
```

Positive alignment means high-reach senders are more susceptible than the
condition average. Negative alignment means high-reach senders are more
resilient than the condition average. Near-zero alignment means susceptibility
is not meaningfully concentrated among high-reach senders.

## Counterfactual Alignment-Gradient Branch

The branch `counterfactual_alignment_gradient/` is the targeted H6/H7 mechanism
analysis for run 2. It starts from the completed main production run, reuses the
already measured private post-attack susceptibility values from Stage `04`, and
evaluates Stage `04b` under condition-specific profile-position assignments.

The branch preserves the same `100 x 7 x 5 = 3,500` scenario structure. The
experimental manipulation is the alignment between private susceptibility and
empirical sender reach.

| Design element | Value |
| --- | --- |
| Branch root | `counterfactual_alignment_gradient/` |
| Source private susceptibility | Main run Stage `04` |
| Branch measurement stage | Stage `04b` |
| Branch Stage `05` mode | `condition_specific` |
| Alignment targets | `[-0.90, -0.60, -0.30, 0.00, +0.30, +0.60, +0.90]` |
| Condition cells | `35` |
| Rows | `3,500` |
| Primary predictor | `achieved_alignment_z` |

This branch is counterfactual by design. It does not replace the fixed-position
main run; it isolates the H6/H7 mechanism that the fixed-position run can only
observe under one empirical assignment.

## Pipeline Stages

| Stage | Name | Main output | Role in run 2 |
| --- | --- | --- | --- |
| `01` | `create_scenarios` | `stage_outputs/01_create_scenarios/` | Builds the `3,500` scenario fixed factorial panel and compatibility audit. |
| `01b` | `assign_exposure_network_positions` | `stage_outputs/01b_assign_exposure_network_positions/` | Assigns the `100` profiles to empirical exposure-network positions. |
| `02` | `assess_baseline_opinions` | `stage_outputs/02_assess_baseline_opinions/` | Measures private baseline opinion `B`. |
| `02b` | `assess_network_exposure_opinions` | `stage_outputs/02b_assess_network_exposure_opinions/` | Measures baseline network-informed opinion `BN`. |
| `03` | `run_opinion_attacks` | `stage_outputs/03_run_opinion_attacks/` | Generates deterministic attack vector specifications. |
| `04` | `assess_post_attack_opinions` | `stage_outputs/04_assess_post_attack_opinions/` | Measures private post-attack opinion `P`. |
| `04b` | `assess_post_attack_network_exposure_opinions` | `stage_outputs/04b_assess_post_attack_network_exposure_opinions/` | Measures post-attack network-informed opinion `PN`. |
| `05` | `compute_effectivity_deltas` | `stage_outputs/05_compute_effectivity_deltas/` | Computes private and network-aware effectivity deltas. |
| `06` | `construct_structural_equation_model` | `stage_outputs/06_construct_structural_equation_model/` | Generates SEM-oriented analytical outputs where applicable. |
| `07` | `generate_research_visuals` | `stage_outputs/07_generate_research_visuals/` | Generates research visualizations. |
| `08` | `generate_publication_assets` | `stage_outputs/08_generate_publication_assets/` | Exports static publication assets. |
| `08b` | `analyze_network_exposure_run` | `stage_outputs/08b_analyze_network_exposure_run/` and `network_exposure_analysis/` | Generates the network-exposure analysis report, tables, figures, and manifest. |

Stage `09_build_research_report` is intentionally not run for this production
folder (`--no-build-report`). The network-layer analysis is kept as a separate
artifact under `network_exposure_analysis/`.

## Analysis Products

The network-exposure analysis is generated by Stage `08b`. The primary products
are:

| Product | Location | Purpose |
| --- | --- | --- |
| Analysis manifest | `network_exposure_analysis/reports/analysis_manifest.json` | Machine-readable inventory of generated network-analysis artifacts. |
| Validation report | `network_exposure_analysis/reports/run_2_network_exposure_validation.html` | Integrated report for network-layer validation and interpretation. |
| Figures | `network_exposure_analysis/figures/` | Paper-facing and diagnostic network-exposure figures. |
| Tables | `network_exposure_analysis/tables/` | Condition-level and profile-level analysis tables. |
| Stage manifest | `stage_outputs/08b_analyze_network_exposure_run/manifest.json` | Pipeline-stage provenance for the analysis run. |
| Alignment-gradient H6/H7 report | `counterfactual_alignment_gradient/network_exposure_analysis/reports/run_2_alignment_gradient_alignment_gradient_h3h4.html` | Branch-specific report for the manipulated alignment-gradient H6/H7 analysis. |

The paper-facing analysis sequence is:

1. Validate the empirical exposure network and profile-position assignment.
2. Estimate private susceptibility per profile and condition: `AE_private`.
3. Plot condition-specific susceptibility by direct sender reach.
4. Compute `sender_reach_susceptibility_alignment_z` for each condition.
5. Relate alignment to `mean_total_network_effect`.
6. Relate alignment to `mean_post_network_increment` to isolate the network
   mechanism.

## Figure Logic

The central figures for the network layer have distinct roles:

| Figure | Purpose | Scientific reading |
| --- | --- | --- |
| Empirical exposure-network overlay | Shows the fixed graph with condition-specific susceptibility overlaid on nodes | A visual check of whether susceptible or resilient profiles occupy high-reach sender positions in each opinion x attack condition. |
| Susceptibility x sender-reach plane | Places profiles by `AE_private` and `outgoing_visibility_weight` | Identifies candidate vulnerability hubs, resilience anchors, susceptible peripheral profiles, and lower-risk peripheral profiles. |
| Alignment versus network attack effect | Plots condition-level alignment against network effect metrics | Primary quantitative test of whether high-reach susceptible placement predicts post-attack network amplification. |

Quadrants are useful for interpretation, but inferential claims use the
continuous alignment metric rather than quadrant labels alone.

## Validity Criteria

A completed run 2 folder is valid only if the following checks pass:

| Check | Required value |
| --- | --- |
| Stage `01` scenario count | `3,500` |
| Selected profile count | `100` |
| Selected opinion leaf count | `7` |
| Selected attack leaf count | `5` |
| Compatibility exclusions | `0` |
| Stage `01b` assigned profile count | `100` |
| Stage `02` private baseline assessments | Complete for all scenarios |
| Stage `02b` baseline network assessments | Complete for all profile x opinion tasks and expanded scenario rows |
| Stage `04` private post-attack assessments | Complete for all scenarios |
| Stage `04b` post-attack network assessments | Complete for all scenarios |
| Stage `05` deltas | Present and aligned with `B`, `BN`, `P`, and `PN` |
| Stage `08b` network analysis | Manifest, report, figures, and tables present |

For the counterfactual alignment-gradient branch, the completed validation
record is:

| Check | Observed value |
| --- | ---: |
| Condition cells | `35` |
| Stage `04b` records | `3,500` |
| Completed tasks | `3,500` |
| Fallbacks | `0` |
| Skipped tasks | `0` |
| Failed rows | `0` |
| Merged unique tasks | `3,500` |
| Branch Stage `05` records | `3,500` |

Fallback counts, failed rows, heuristic failures, and repair attempts are
interpreted from the stage manifests and logs before paper-level analysis.

## Folder Contents

| Path | Contents |
| --- | --- |
| `README.md` | Human-facing methodological and provenance guide for production run 2. |
| `config/` | Run-local ontology bundle, source manifest, selected panels, and pipeline configuration. |
| `logs/` | Launcher and per-stage execution logs. |
| `provenance/` | Run manifest. Raw LLM request/response provenance is omitted from the curated official package. |
| `stage_outputs/` | Canonical machine-readable outputs for each pipeline stage. |
| `network_exposure_analysis/` | Network-layer report, figures, tables, and analysis manifest generated by Stage `08b`. |
| `counterfactual_alignment_gradient/` | Branch-local H6/H7 alignment-gradient design, merged compact outputs, branch Stage `05`, reports, figures, tables, and validation manifests. |

The stage manifests are the authoritative record of row counts, stage
parameters, generated artifacts, and completion metadata.

## Reproduce

The production launcher is:

```bash
bash scripts/production/run_2.sh full
```

The launcher validates dependencies, ontology hashes, selected opinion paths,
selected attack leaves, static design counts, and required API-key state before
starting paid execution.

Available launcher modes:

| Mode | Purpose |
| --- | --- |
| `structural` | No paid LLM calls; validates Stage `01` and `01b` for the full `3,500`-scenario design. |
| `smoke` | Paid `30 profiles x 2 opinions x 2 attacks = 120` subset through Stage `08b`. |
| `full` | Full paid `100 profiles x 7 opinions x 5 attacks = 3,500` production run through Stage `08b`. |

The full run is configured with:

```text
--output-root evaluation/production/run_2
--run-id run_2
--n-profiles 100
--seed 120
--attack-ratio 1.0
--no-use-test-ontology
--ontology-root evaluation/production/run_2/config/ontology_mixed
--profile-candidate-multiplier 8
--assess-network-exposure
--assess-post-attack-network-exposure
--network-exposure-top-k 8
--post-attack-network-exposure-top-k 8
--post-attack-network-min-peers 1
--analyze-network-exposure-run
--openrouter-model deepseek/deepseek-v4-flash
--temperature 0.15
--max-repair-iter 2
--coherence-threshold 0.74
--realism-threshold 0.72
--save-raw-llm
--generate-visuals
--export-static-figures
--no-build-report
--stop-after-stage 08b
```

## Interpretation Guardrails

- Describe the exposure graph as an engagement-derived plausible exposure graph,
  not as verified social ties, subscriptions, or confirmed reading behavior.
- The edge direction is `visible peer -> exposed receiver`.
- The graph is fixed before `B`, `BN`, `P`, and `PN` are measured.
- Peer context is information presented to the model; the model response is not
  assumed to be a mechanical average of peer scores.
- `top_k=8` limits how many peer exemplars enter the prompt. It is not the
  scientific peer-neighborhood size used in network metrics.
- Interpret H6 and H7 through planned continuous alignment metrics, not through
  visual quadrant membership alone.
- Interpret the fixed-position main run as the full production network run
  under one deterministic profile-position assignment; interpret the
  counterfactual alignment-gradient branch as the targeted experimental
  mechanism test for H6/H7.
- Run 2 supports internally valid comparisons across the selected fixed
  factorial state space. Claims are scoped to the selected profiles, opinions,
  attacks, and empirical exposure graph.

## Relation To Production Run 1

Production run 1 estimates private susceptibility over a larger individual-layer
state space with the empirical network layer disabled. Production run 2 narrows
the scenario space to a balanced `100 x 7 x 5` panel so the same profiles and
network positions can be compared across every selected opinion and attack
condition. The two runs therefore answer complementary questions:

| Run | Primary layer | Main contribution |
| --- | --- | --- |
| `run_1` | Individual/private layer | Full-scale private susceptibility and opinion-domain analysis. |
| `run_2` | Network exposure layer | How private susceptibility interacts with empirical sender reach and peer exposure. |

Run 2 is not an exhaustive ontology sweep. It is a targeted, balanced
network-layer experiment designed to preserve same-condition peer sets and
strengthen H6/H7 analysis while keeping the paid scenario count bounded.
