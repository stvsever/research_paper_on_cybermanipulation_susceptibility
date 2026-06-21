#!/usr/bin/env python
"""Analyse production run 1 (10,000-scenario individual layer) from the stage-05 deltas.

Runs the SOTA family-wise profile-moderation analysis and the individual-layer
inferential tests, and renders the paper-ready figures plus one interactive HTML.
All outputs land in the run's own analysis/ and visuals/ subdirectories. Read-only
with respect to the raw deltas; safe to re-run.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]  # src/backend/utils/analysis/<file> -> repo root
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.backend.utils.analysis.production_moderation import run_production_moderation, drop_excluded_domains
from src.backend.utils.analysis.individual_layer_statistics import run_individual_layer_statistics
from src.backend.utils.figures.production_figures import generate_production_figures

RUN = ROOT / "evaluation" / "production" / "run_1"
SEM = RUN / "stage_outputs" / "05_compute_effectivity_deltas" / "sem_long_raw.csv"
ANALYSIS = RUN / "analysis"
FIGS = RUN / "visuals" / "paper_figures"
ANALYSIS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print(f"Loading deltas: {SEM}")
    df = pd.read_csv(SEM)
    print(f"  {len(df):,} leaf rows")

    # 1) Family-wise profile moderation (the headline individual-layer analysis).
    print("Running profile-moderation analysis ...")
    mod = run_production_moderation(df)
    mod["family_table"].to_csv(ANALYSIS / "moderation_family_table.csv", index=False)
    mod["within_family"].to_csv(ANALYSIS / "moderation_within_family.csv", index=False)
    mod["curated"].to_csv(ANALYSIS / "moderation_curated.csv", index=False)
    mod["by_domain"].to_csv(ANALYSIS / "moderation_by_domain.csv", index=False)
    import json
    (ANALYSIS / "variance_context.json").write_text(json.dumps(mod["variance_context"], indent=2))

    # 2) Inferential tests (attack works, domain, tactic, direction, heterogeneity).
    #    The single-leaf macroeconomic domain is excluded from every analysis.
    print("Running inferential tests ...")
    stats_df, stats_summary = run_individual_layer_statistics(drop_excluded_domains(df))
    if stats_df is not None and not stats_df.empty:
        stats_df.to_csv(ANALYSIS / "inferential_tests.csv", index=False)

    summary = mod["summary"] + "\n\n" + stats_summary
    (ANALYSIS / "ANALYSIS_SUMMARY.txt").write_text(summary)
    print("\n" + summary + "\n")

    # 3) Figures + interactive dashboard (dashboard writes to visuals/, PNGs to paper_figures/).
    print("Generating figures ...")
    files = generate_production_figures(str(SEM), mod, str(FIGS))
    for f in files:
        if Path(f).suffix == ".html":
            print(f"  dashboard -> {Path(f).relative_to(ROOT)}")
    pngs = sorted(FIGS.glob("*.png"))
    print(f"  {len(pngs)} PNG figures in {FIGS.relative_to(ROOT)}:")
    for p in pngs:
        print(f"    {p.name}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
