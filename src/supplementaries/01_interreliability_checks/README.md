# Supplementary Analysis 01: Interreliability Checks

This supplement contains the individual-layer reliability analyses for the
tiny controlled scenario subsets:

1. Test-retest repeated-run reproducibility with the same model and scenario IDs.
2. Cross-provider inter-rater reliability across five low-cost OpenRouter models.

The detailed methodology, fixed parameters, model benchmark metadata, results,
and figure interpretations are in:

- `individual_layer/README.md`

Main manuscript-facing PNGs:

- `individual_layer/04_images/02_test_retest/test_retest_reliability_main.png`
- `individual_layer/04_images/01_cross_provider/cross_provider_reliability_main.png`
- `individual_layer/04_images/04_rank_robustness/cross_provider_rank_robustness_main.png`
- `individual_layer/04_images/03_model_benchmarks/mmlu_pro_3d_specification_curve.png`

The scripts do not run network-layer analyses and do not make git commits or
pushes. The completed results are stored under `individual_layer/03_metrics/`,
`individual_layer/04_images/`, and `individual_layer/05_tables/`.
