# Limitations And Pipeline Notes

This file records methodological notes surfaced while validating Run 2.

## Current Interpretation

- Run 2 is a validation and demonstration run, not final inferential evidence by itself.
- Scenario-level rows repeat profiles across opinions and attack vectors. Descriptive correlations are useful for sanity checks, but final models should account for repeated profile outcomes.
- H2 has a promising scenario-level descriptive relationship: `peer_private_attack_activation` versus `pn_increment_effectivity` has Pearson `r = 0.159`.

## BN Delta Expansion Note

Stage `02b` canonical profile-opinion assessments report:

- mean `BN - B`: `-12.4`
- mean absolute `BN - B`: `45.45142857142857`

Stage `05` expanded scenario rows report:

- mean `network_exposure_delta_score`: `-12.369714285714286`
- mean absolute `network_exposure_delta_score`: `179.336`

The means are aligned, but the absolute means differ substantially. This likely reflects canonical profile-opinion BN tasks being expanded across repeated scenario rows with separately elicited private baseline assessments.

For the Run 2 validation report, use Stage `02b` canonical artifacts when discussing `BN - B`. Use Stage `05` for private post-attack, post-network, and final effect construction.

## Suggested Pipeline Hardening

- Consider making private baseline assessment canonical by `profile_id × opinion_leaf`, or explicitly mark repeated baseline rows as repeated stochastic elicitation.
- Add an explicit Stage `05` column for canonical `BN - B` from Stage `02b`, separate from any row-expanded comparison.
- Add grouped/clustered uncertainty estimates in final reports, because scenario rows are not independent.
- Treat vulnerability hub and resilience anchor labels as descriptive candidate labels from run-level percentile rankings, not causal evidence.
- Use condition-specific vulnerability planes for attack/opinion interpretation; the averaged profile plane is only a summary.
- Interpret sender-reach susceptibility alignment as descriptive placement of susceptibility on the empirical graph, not as a causal network effect estimate.
- For production, rerun this validation on a larger profile panel and compare role-level patterns across seeds.
