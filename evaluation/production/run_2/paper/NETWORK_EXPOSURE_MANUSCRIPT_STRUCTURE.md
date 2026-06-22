# Network Exposure Layer Manuscript Structure

Purpose: define the base structure for the manuscript Methods and Results
sections covering the network exposure layer of production `run_2`, with the
fixed-position main run as the general network evidence base and the
`counterfactual_alignment_gradient/` branch as the primary H3/H4 mechanism test.

This document is a structure scaffold. It should guide later prose writing, not
serve as the final manuscript text.

## Core Framing

| Manuscript element | Source run | Scientific role | Primary scope |
| --- | --- | --- | --- |
| Main production network run | `evaluation/production/run_2/` | Full fixed-factorial network pipeline and general network-layer evidence | H1, H2, measurement completeness, private/network deltas, fixed exposure graph |
| Alignment-gradient branch | `evaluation/production/run_2/counterfactual_alignment_gradient/` | Targeted experimental mechanism test for profile susceptibility x sender reach | H3 and H4 |
| Existing methods/provenance base | `evaluation/production/run_2/README.md` | Run design, ontologies, graph, pipeline stages, guardrails | Methods backbone |

## Methods And Materials Structure

### 1. Study Aim And Network-Layer Rationale

Required content:

- State the network-layer research question.
- Explain why a fixed full factorial panel was needed for same-condition peer
  comparison.
- Clarify that the exposure graph is empirical and fixed before outcome
  measurement.
- Separate the fixed-position main run from the counterfactual H3/H4 branch.

Do not yet overclaim:

- Do not describe the graph as verified reading behavior.
- Do not frame the ontology subset as a random sample from all possible issues
  or attacks.

### 2. Scenario Panel And Ontology Sources

Required content:

| Axis | Design choice | Manuscript role |
| --- | --- | --- |
| Profiles | `100` deterministic profiles from the test profile ontology | Repeated profile panel across all opinion x attack cells |
| Opinions | `7` production opinion leaves | Paper-aligned issue-position domains |
| Attacks | `5` social-media-relevant test attack leaves | Platform-relevant manipulation mechanisms |
| Scenario count | `100 x 7 x 5 = 3,500` | Balanced full factorial network panel |

Source artifacts:

- `README.md`
- `config/ontology_source_manifest.json`
- `config/opinion_panel.json`
- `config/attack_panel.json`
- `config/scenario_design.json`

### 3. Empirical Exposure Network

Required content:

- Graph: `politisky24_bluesky_v1`.
- Edge meaning: `visible peer -> exposed receiver`.
- Edge weights: like `0.35`, repost `0.80`, quote `0.90`.
- Profile-position assignment: one empirical position per generated profile.
- Main run: profile-position assignment fixed across all `35` conditions.
- Branch: same empirical positions, but condition-specific profile-position
  reassignment for the H3/H4 intervention.

Candidate figure:

| Candidate | Path | Recommended use |
| --- | --- | --- |
| Empirical exposure network substrate | `network_exposure_analysis/figures/empirical_exposure_network_run2.png` | Supplement or methods figure |

### 4. Measurement Backbone

Required content:

| Symbol | Measurement | Pipeline source |
| --- | --- | --- |
| `B` | Private baseline opinion | Stage `02` |
| `BN` | Baseline opinion after empirical peer context | Stage `02b` |
| `P` | Private post-attack opinion | Stage `04` |
| `PN` | Post-attack opinion after same-condition peer context | Stage `04b` |

Direction-aware deltas:

| Quantity | Formula | Role |
| --- | --- | --- |
| `AE_private` | `(P - B) * d` | Private attack susceptibility |
| `PN_increment_effectivity` | `(PN - P) * d` | Primary post-network amplification outcome |
| `AE_total_network` | `(PN - B) * d` | Secondary final network attack effect |

Candidate figure:

| Candidate | Path | Recommended use |
| --- | --- | --- |
| Measurement backbone deltas | `network_exposure_analysis/figures/measurement_backbone_deltas.png` | Supplement or compact methods schematic |

### 5. Fixed-Position Main Run Analysis

Required content:

- Use the fixed-position main run for general network-layer results.
- Treat scenario rows as repeated measurements from the same profile-position
  panel.
- Report H1/H2-relevant descriptive and model results from the main run.
- Explain that natural fixed-position alignment had limited alignment range and
  was not sufficient as the primary H3/H4 mechanism test.

Core source tables:

| Table | Path | Use |
| --- | --- | --- |
| Stage status | `network_exposure_analysis/tables/stage_status.csv` | Completeness and quality |
| Effect summary | `network_exposure_analysis/tables/effect_summary.csv` | Overall private/network deltas |
| Attack summary | `network_exposure_analysis/tables/attack_summary.csv` | Attack-vector differences |
| Opinion summary | `network_exposure_analysis/tables/opinion_summary.csv` | Opinion-domain differences |
| Correlations | `network_exposure_analysis/tables/correlations.csv` | H2 and network-position diagnostics |
| Fixed-position alignment table | `network_exposure_analysis/tables/centrality_alignment_outcome_link.csv` | Natural alignment range and limitation |

### 6. Counterfactual Alignment-Gradient Design For H3/H4

Required content:

- The branch reuses Stage `04` private post-attack susceptibility measurements.
- The branch reruns Stage `04b` under condition-specific exposure assignments.
- The empirical graph and 100 network positions remain fixed; profile-position
  mapping changes by condition.
- Manipulation targets: `[-0.90, -0.60, -0.30, 0.00, +0.30, +0.60, +0.90]`.
- Latin schedule: each attack receives all seven alignment targets once; each
  alignment target appears in five condition cells.
- Primary predictor: continuous `achieved_alignment_z`, not the categorical
  target label.

Core source artifacts:

| Artifact | Path | Use |
| --- | --- | --- |
| Design manifest | `counterfactual_alignment_gradient/design/alignment_design_manifest.json` | Branch design and hashes |
| Condition schedule | `counterfactual_alignment_gradient/design/condition_schedule.csv` | Target assignment schedule |
| Alignment balance | `counterfactual_alignment_gradient/network_exposure_analysis/tables/alignment_design_balance.csv` | Manipulation balance |
| Quality gates | `counterfactual_alignment_gradient/network_exposure_analysis/tables/quality_gates.csv` | Branch validity |

### 7. Statistical Analysis Plan

Required content:

| Hypothesis | Unit of inference | Primary model | Primary outcome |
| --- | --- | --- | --- |
| H1 | Profile/scenario-level, scoped to design | Main run individual/private susceptibility analyses | `AE_private` |
| H2 | Scenario/profile summaries, scoped to design | Main run peer activation and post-network increment analyses | `PN_increment_effectivity` |
| H3 | Condition cell, `n = 35` | `outcome ~ achieved_alignment_z + C(opinion) + C(attack)` | `mean(PN_increment_effectivity)` |
| H4 | Condition cell, `n = 35` | Same model, attenuation side of the same alignment mechanism | `mean(PN_increment_effectivity)` and `mean(AE_total_network)` |

Required inferential guardrails:

- Use condition-level cells for H3/H4; do not treat 3,500 profile rows as
  independent H3/H4 tests.
- Use HC3 robust standard errors for the displayed H3/H4 inference.
- Use within-attack permutation p-values as a robustness check.
- Use BH-FDR q-values across the two displayed H3/H4 endpoints as transparent
  sensitivity, not as the primary planned-test decision rule.
- Report `AE_total_network` as secondary; keep `PN_increment_effectivity` as the
  primary network mechanism endpoint.

## Results Structure

### 1. General Private And Network Effects In The Fixed-Position Main Run

Purpose:

- Present the base empirical facts of production `run_2` before the H3/H4
  mechanism result.
- The main figure or table for this section still needs to be selected.
- Do not yet commit this section to a specific figure. The correct object should
  show the main empirical pattern of the fixed-position network run in a compact
  and paper-facing way.

Decision material:

| Candidate artifact | Path | Current status |
| --- | --- | --- |
| Overall effect summary | `network_exposure_analysis/tables/effect_summary.csv` | Candidate source for base factual claims |
| Attack summary | `network_exposure_analysis/tables/attack_summary.csv` | Candidate source if attack differences are foregrounded |
| Opinion summary | `network_exposure_analysis/tables/opinion_summary.csv` | Candidate source if opinion-domain differences are foregrounded |
| Main run figures | `network_exposure_analysis/figures/` | Candidate figure pool; final selection deferred |

### 2. H3/H4 Network-Mechanism Result

Purpose:

- Present H3 and H4 as one connected mechanism: network-wide attack effects
  increase when more susceptible profiles occupy higher-reach sender positions,
  and attenuate when more resilient profiles occupy those positions.
- Shape the results narrative around the three-panel composite figure only.
- Use the component figures as supporting artifacts, not as separate main
  results competing with the composite.

Main figure:

| Figure | Path | Use |
| --- | --- | --- |
| Three-panel H3/H4 network mechanism figure | `counterfactual_alignment_gradient/network_exposure_analysis/publication_figures/network_mechanism_abc_with_outcome_test/main_figure_network_mechanism_abc_with_outcome_test.png` | Primary H3/H4 result figure; use PDF or TIFF export for final journal submission if needed |

Narrative sequence:

- Panel A first establishes the empirical exposure substrate. It shows the full
  `35`-condition network overlay: the same directed exposure graph is held
  fixed, while condition-specific profile-position assignments change which
  susceptible or resilient profiles occupy sender positions.
- Panel B then abstracts the same mechanism into condition-specific
  vulnerability planes. It shows how each condition places private
  susceptibility across sender-reach percentiles, making the experimental
  alignment gradient visually inspectable.
- Panel C provides the inferential test. It links achieved sender-reach
  susceptibility alignment to post-network amplification and final network
  attack effect using condition-level fixed-effect models.

Supporting result anchors:

- Analysis unit: `35` opinion x attack condition cells, each averaging `100`
  profiles.
- Manipulation check: target-achieved alignment tolerance passed; the design
  created the intended sender-reach susceptibility gradient.
- Primary endpoint: `mean_pn_increment_effectivity`; beta `2.399` score points
  per 1 SD alignment increase; HC3 95% CI `[1.478, 3.320]`; HC3 p `1.79e-05`;
  within-attack permutation p `0.00020`.
- Secondary endpoint: `mean_ae_total_network`; beta `2.697` score points per 1
  SD alignment increase; HC3 95% CI `[0.953, 4.440]`; HC3 p `0.00398`;
  within-attack permutation p `0.00200`.

Supporting tables:

| Table | Path | Manuscript role |
| --- | --- | --- |
| Figure 4 outcome-test summary | `counterfactual_alignment_gradient/network_exposure_analysis/tables/figure4_outcome_test_summary.csv` | Source for main H3/H4 coefficient values |
| Alignment design balance | `counterfactual_alignment_gradient/network_exposure_analysis/tables/alignment_design_balance.csv` | Source for manipulation-balance claims |

### 3. Profile-Trait Bridge Between Direct And Network Susceptibility

Purpose:

- Test whether profile characteristics that predict direct private
  susceptibility also predict post-network amplification.
- Use this as the individual-to-network bridge before the H5/H6
  position-mechanism result.
- Keep inference profile-level; do not treat scenario rows as independent.

Main figure:

| Figure | Path | Use |
| --- | --- | --- |
| Profile predictors of direct versus network-conditioned susceptibility | `paper/profile_trait_direct_vs_network/main_figure_profile_trait_direct_vs_network.png` | Main network-layer bridge result |

Supporting result anchors:

- Primary source: alignment-gradient branch, `100` profiles, `35` condition
  measurements per profile, `0` post-attack network fallbacks.
- Profile-level direct/network outcome correlation: `r = 0.28`, 95% bootstrap
  CI `[0.08, 0.47]`.
- Focal trait coefficient-vector similarity: `r = 0.73`, 95% bootstrap CI
  `[0.36, 0.89]`.
- Fixed-position sensitivity should be reported as supplementary because it
  excludes post-network fallback rows and retains fewer profiles.

Core source tables:

| Table | Path | Manuscript role |
| --- | --- | --- |
| Profile summary | `paper/profile_trait_direct_vs_network/profile_trait_direct_vs_network_profile_summary.csv` | Profile-level analysis dataset |
| Coefficients | `paper/profile_trait_direct_vs_network/profile_trait_direct_vs_network_coefficients.csv` | Parallel model estimates |
| Bootstrap summary | `paper/profile_trait_direct_vs_network/profile_trait_direct_vs_network_bootstrap_summary.csv` | CI source for figure and text |
| Fixed-position sensitivity | `paper/profile_trait_direct_vs_network/profile_trait_direct_vs_network_fixed_position_sensitivity.csv` | Supplementary robustness |

### 4. Robustness And Sensitivity

Purpose:

- Demonstrate that H3/H4 conclusions are not an artifact of one warning class,
  fallback row set, or classical OLS assumption.

Candidate figure/table:

| Artifact | Path | Recommendation |
| --- | --- | --- |
| Robustness coefficient figure | `counterfactual_alignment_gradient/network_exposure_analysis/figures/alignment_gradient_robustness_coefficients.png` | Supplement or compact main figure panel |
| Robustness table | `counterfactual_alignment_gradient/network_exposure_analysis/tables/h3h4_robustness_results.csv` | Supplement |
| Original vs branch comparison | `counterfactual_alignment_gradient/network_exposure_analysis/tables/original_vs_branch_alignment_comparison.csv` | Important supplement; short main-text mention |

### 5. Interpretation Boundaries

Required content:

- Scope claims to the selected profile, opinion, attack, and empirical graph
  panel.
- State that the branch is a counterfactual mechanism experiment, not a claim
  that the platform naturally assigned profiles to those positions.
- State that the main run remains the source for fixed-position production
  outputs and H1/H2/general network conclusions.
- State that H3/H4 are supported as network-mechanism claims under manipulated
  sender-reach susceptibility alignment.

## Recommended Main Manuscript Figure Set

| Order | Figure | Source | Purpose |
| ---: | --- | --- | --- |
| 1 | Base Run 2 empirical facts figure | To be selected from `network_exposure_analysis/` | H1/H2/general fixed-position network facts; final figure still undecided |
| 2 | Profile-trait direct versus network susceptibility figure | `paper/profile_trait_direct_vs_network/main_figure_profile_trait_direct_vs_network.png` | Individual-to-network bridge result |
| 3 | Three-panel H3/H4 network mechanism figure | `counterfactual_alignment_gradient/network_exposure_analysis/publication_figures/network_mechanism_abc_with_outcome_test/main_figure_network_mechanism_abc_with_outcome_test.png` | Main H3/H4 narrative and conclusion |
| Optional | Design/measurement schematic | `publication_assets/figures/figure_1_study_design.png` or `network_exposure_analysis/figures/measurement_backbone_deltas.png` | Include only if the broader paper needs additional methods orientation |

## Recommended Supplementary Material

| Supplement item | Source |
| --- | --- |
| Full run configuration and ontology panel | `README.md`, `config/*.json` |
| Stage quality diagnostics | `network_exposure_analysis/tables/stage_status.csv`, `counterfactual_alignment_gradient/network_exposure_analysis/tables/quality_gates.csv` |
| Full attack/opinion descriptive tables | `network_exposure_analysis/tables/attack_summary.csv`, `network_exposure_analysis/tables/opinion_summary.csv` |
| Empirical network substrate | `network_exposure_analysis/figures/empirical_exposure_network_run2.png` |
| Fixed-position original network overlay | `network_exposure_analysis/figures/scenario_vulnerability_network_overlays.png` |
| Component H3/H4 network overlay | `counterfactual_alignment_gradient/network_exposure_analysis/figures/branch_full_35_condition_network_overlay.png` |
| Component H3/H4 vulnerability planes | `counterfactual_alignment_gradient/network_exposure_analysis/figures/branch_condition_vulnerability_planes.png` |
| Component H3/H4 outcome test | `counterfactual_alignment_gradient/network_exposure_analysis/figures/alignment_gradient_fixed_effect_outcomes.png` |
| Component H3/H4 manipulation check | `counterfactual_alignment_gradient/network_exposure_analysis/figures/target_vs_achieved_alignment.png` |
| Original versus branch alignment comparison | `counterfactual_alignment_gradient/network_exposure_analysis/tables/original_vs_branch_alignment_comparison.csv` |
| Robustness and sensitivity table | `counterfactual_alignment_gradient/network_exposure_analysis/tables/h3h4_robustness_results.csv` |
| Robustness coefficient figure | `counterfactual_alignment_gradient/network_exposure_analysis/figures/alignment_gradient_robustness_coefficients.png` |
| Full branch report | `counterfactual_alignment_gradient/network_exposure_analysis/reports/run_2_alignment_gradient_alignment_gradient_h3h4.html` |
| Profile-trait fixed-position sensitivity | `paper/profile_trait_direct_vs_network/profile_trait_direct_vs_network_fixed_position_sensitivity.csv` |
| Profile-trait univariate supplement | `paper/profile_trait_direct_vs_network/profile_trait_direct_vs_network_univariate_supplement.csv` |

## Open Decisions Before Prose Drafting

| Decision | Default recommendation |
| --- | --- |
| Whether H1/H2 are fully covered in this paper section or only briefly linked to broader results | Briefly cover them here, but keep H3/H4 as the network-exposure focus |
| Which base Run 2 empirical figure/table should introduce the fixed-position results | Still undecided; choose after comparing which artifact best communicates the central base facts |
| Whether to include exact model table in main text | Include compact H3/H4 coefficient table or report values in text; full robustness table in supplement |
| Whether to name the branch "counterfactual" throughout | Yes, consistently, because the profile-position mapping is experimentally manipulated |

## Writing Order

1. Convert Methods sections 1-7 into concise prose.
2. Select the base Run 2 empirical facts figure or table.
3. Draft the Results section around the base Run 2 object and the three-panel
   H3/H4 mechanism figure.
4. Add interpretation boundaries after the H3/H4 result.
5. Cross-check every numerical claim against the source CSV or manifest.
