# Network Exposure Run Analysis

This folder contains a compact validation and interpretation report for pipeline runs that include the empirical exposure-network phases.

The current report target is `evaluation/tests/run_2`. It validates whether the new network exposure measurements are functioning as intended and summarizes the first scientific insights they enable:

- `B`: private baseline opinion.
- `BN`: baseline opinion after empirical incoming peer baseline context.
- `P`: private post-attack opinion.
- `PN`: post-attack opinion after same-condition empirical incoming peer post-attack context.

The analysis is intentionally narrow. It does not replace the main pipeline, the workbench, or the exposure-network substrate report. It reads completed run artifacts and produces a colleague-facing HTML report with figures, compact tables, and explicit limitations.

## Rebuild

From the repository root:

```bash
env UV_CACHE_DIR=/tmp/uv-cache-cybermanipulation MPLCONFIGDIR=/tmp/matplotlib-cybermanipulation \
  uv run --with pandas --with numpy --with matplotlib --with seaborn \
  python src/backend/utils/network_exposure_run_analysis/scripts/build_run_network_exposure_report.py \
  --run-root evaluation/tests/run_2
```

## Outputs

```text
figures/
tables/
reports/run_2_network_exposure_validation.html
reports/LIMITATIONS_AND_PIPELINE_NOTES.md
reports/analysis_manifest.json
```

## Scope

The report is a validation and demonstration artifact for a 60-profile test run. It can show whether the network layer is wired correctly and whether the resulting variables behave meaningfully, but it should not be treated as final inferential evidence for the paper.
