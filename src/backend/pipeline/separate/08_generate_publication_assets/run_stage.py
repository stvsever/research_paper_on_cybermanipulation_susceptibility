from __future__ import annotations

import argparse
import logging
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.utils.io import abs_path, ensure_dir, stage_manifest_path, write_json
from src.backend.utils.logging_utils import setup_logging
from src.backend.utils.publication_assets import generate_publication_assets
from src.backend.utils.schemas import StageArtifactManifest, StageConfig


LOGGER = logging.getLogger(__name__)


class Stage08Config(StageConfig):
    sem_result_path: str
    ols_params_path: str
    bootstrap_params_path: str
    exploratory_comparison_path: str
    config_path: str
    ontology_catalog_path: str
    assumptions_path: str
    critiques_path: str
    report_assets_root: str


def run_stage(input_path: str, output_dir: str, config: Stage08Config) -> StageArtifactManifest:
    ensure_dir(output_dir)

    result = generate_publication_assets(
        sem_long_csv_path=input_path,
        sem_result_json_path=config.sem_result_path,
        ols_params_csv_path=config.ols_params_path,
        bootstrap_params_csv_path=config.bootstrap_params_path,
        exploratory_comparison_csv_path=config.exploratory_comparison_path,
        config_json_path=config.config_path,
        ontology_catalog_path=config.ontology_catalog_path,
        assumptions_json_path=config.assumptions_path,
        critiques_json_path=config.critiques_path,
        output_dir=output_dir,
        report_assets_root=config.report_assets_root,
        run_id=config.run_id,
        paper_title=config.paper_title or "",
    )

    manifest = StageArtifactManifest(
        stage_id="08",
        stage_name="generate_publication_assets",
        input_path=abs_path(input_path),
        primary_output_path=result["manifest_path"],
        output_files=result["visual_files"] + result["table_files"] + result["snapshot_files"] + [result["manifest_path"]],
        record_count=0,
        metadata={
            "copied_figures": result["copied_figures"],
            "copied_tables": result["copied_tables"],
        },
    )
    write_json(stage_manifest_path(output_dir), manifest.model_dump())
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 08 - Generate publication assets")
    parser.add_argument("--input-path", required=True, help="Path to sem_long_encoded.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="run_1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paper-title", required=True)
    parser.add_argument("--report-assets-root", required=True)
    parser.add_argument("--sem-result-path", required=True)
    parser.add_argument("--ols-params-path", required=True)
    parser.add_argument("--bootstrap-params-path", required=True)
    parser.add_argument("--exploratory-comparison-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--ontology-catalog-path", required=True)
    parser.add_argument("--assumptions-path", required=True)
    parser.add_argument("--critiques-path", required=True)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file, args.log_level)

    config = Stage08Config(
        stage_name="generate_publication_assets",
        run_id=args.run_id,
        seed=args.seed,
        paper_title=args.paper_title,
        report_assets_root=args.report_assets_root,
        sem_result_path=args.sem_result_path,
        ols_params_path=args.ols_params_path,
        bootstrap_params_path=args.bootstrap_params_path,
        exploratory_comparison_path=args.exploratory_comparison_path,
        config_path=args.config_path,
        ontology_catalog_path=args.ontology_catalog_path,
        assumptions_path=args.assumptions_path,
        critiques_path=args.critiques_path,
    )

    manifest = run_stage(args.input_path, args.output_dir, config)
    LOGGER.info("Stage 08 completed: %s", manifest.primary_output_path)


if __name__ == "__main__":
    main()
