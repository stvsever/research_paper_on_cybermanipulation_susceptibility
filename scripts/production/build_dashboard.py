#!/usr/bin/env python
"""Generate the full legacy interactive dashboard for production run 1.

Drives the pipeline's rich `generate_research_visuals` (ontology explorer, profile
network, conditional-susceptibility and moderation views, etc.) using the stage-06
outputs computed on a no-macro subsample, and writes the result to
visuals/production_dashboard.html. The raw-data embedding is taken from a smaller
subsample so the single HTML stays portable.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.backend.utils.reporting.visualization_dashboard import generate_research_visuals

RUN = ROOT / "evaluation" / "production" / "run_1"
S06 = RUN / "stage_outputs" / "06_construct_structural_equation_model"
DIN = S06 / "_dash_input"
VIS = RUN / "visuals"
TMP = S06 / "_dashboard_out"


def main() -> None:
    sem_result = S06 / "sem_result.json"
    ols = S06 / "ols_robust_params.csv"
    if not sem_result.exists() or not ols.exists():
        sys.exit(f"stage-06 outputs missing ({sem_result.exists()=}, {ols.exists()=}); run stage 06 first")

    # Smaller embedding so the single HTML stays portable; aggregate views still come
    # from the full stage-06 outputs. Place it next to the stage-06 files so the
    # dashboard discovers profile_level_effectivity.csv / profile_sem_wide.csv there.
    full = pd.read_csv(DIN / "sem_long_encoded.csv")
    full["_scn"] = full["scenario_id"].astype(str).str.split("__", n=1).str[0]
    keep = pd.Series(full["_scn"].unique())
    keep = keep.sample(min(500, len(keep)), random_state=3)
    emb = full[full["_scn"].isin(set(keep))].drop(columns=["_scn"])
    emb_path = S06 / "_dashboard_embed.csv"
    emb.to_csv(emb_path, index=False)
    print(f"embedding {emb['scenario_id'].str.split('__', n=1).str[0].nunique()} scenarios into the dashboard")

    TMP.mkdir(parents=True, exist_ok=True)
    result = generate_research_visuals(
        sem_long_csv_path=str(emb_path),
        sem_result_json_path=str(sem_result),
        ols_params_csv_path=str(ols),
        output_dir=str(TMP),
        run_id="production_run_1",
    )
    dash = Path(result["dashboard_path"])
    VIS.mkdir(parents=True, exist_ok=True)
    target = VIS / "production_dashboard.html"
    shutil.copy2(dash, target)
    size_mb = target.stat().st_size / 1e6
    print(f"dashboard -> {target.relative_to(ROOT)}  ({size_mb:.1f} MB)")

    # tidy: keep visuals clean, drop the scratch dirs/files
    for p in (TMP, DIN, emb_path):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()
    print("cleaned scratch inputs")


if __name__ == "__main__":
    main()
