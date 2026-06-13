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
from src.backend.utils.schemas import StageArtifactManifest, StageConfig
from src.backend.utils.visualization_dashboard import generate_research_visuals


LOGGER = logging.getLogger(__name__)


class Stage07Config(StageConfig):
    sem_result_path: str
    ols_params_path: str


def run_stage(input_path: str, output_dir: str, config: Stage07Config) -> StageArtifactManifest:
    ensure_dir(output_dir)

    result = generate_research_visuals(
        sem_long_csv_path=input_path,
        sem_result_json_path=config.sem_result_path,
        ols_params_csv_path=config.ols_params_path,
        output_dir=output_dir,
        run_id=config.run_id,
    )

    manifest = StageArtifactManifest(
        stage_id="07",
        stage_name="generate_research_visuals",
        input_path=abs_path(input_path),
        primary_output_path=result["dashboard_path"],
        output_files=result["visual_files"],
        record_count=0,
        metadata={"summary_cards": result["summary_cards"]},
    )
    write_json(stage_manifest_path(output_dir), manifest.model_dump())
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 07 - Generate research visuals")
    parser.add_argument("--input-path", required=True, help="Path to sem_long_encoded.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sem-result-path", required=True)
    parser.add_argument("--ols-params-path", required=True)
    parser.add_argument("--run-id", default="run_1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file, args.log_level)

    config = Stage07Config(
        stage_name="generate_research_visuals",
        run_id=args.run_id,
        seed=args.seed,
        sem_result_path=args.sem_result_path,
        ols_params_path=args.ols_params_path,
    )

    manifest = run_stage(args.input_path, args.output_dir, config)
    LOGGER.info("Stage 07 completed: dashboard=%s", manifest.primary_output_path)


if __name__ == "__main__":
    main()
