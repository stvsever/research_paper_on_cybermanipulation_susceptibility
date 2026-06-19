"""
User-ontology CLI entry point
==============================
Allows cybersecurity analysts to run the full pipeline with custom
PROFILE × ATTACK × OPINION ontologies supplied as three JSON files,
without modifying any pipeline source code.

Usage
-----
python -m src.backend.ontology.utils.cli \\
    --profile-json path/to/profile.json \\
    --attack-json  path/to/attack.json  \\
    --opinion-json path/to/opinion.json \\
    [options]

Key options
-----------
--output-root DIR          Where to write run artefacts   (default: evaluation/user_run_1)
--run-id STR               Short identifier for this run  (default: user_run_1)
--n-profiles INT           How many profiles to simulate  (default: 40)
--attack-leaves CSV        Comma-01_separated ATTACK leaves  (default: all leaves)
--max-opinion-leaves INT   Cap on opinion leaves per run  (default: all)
--openrouter-model STR     LLM model slug                 (default: mistralai/mistral-small-3.2-24b-instruct)
--validate-only            Validate ontologies and exit without running
--dry-run                  Print resolved config and exit

This module delegates to run_full_pipeline.py after injecting the utils-
supplied ontology root into a temporary directory and passing --ontology-root.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.ontology.utils.validator import validate_ontology_triplet


LOGGER = logging.getLogger(__name__)
_DEFAULT_MODEL = "mistralai/mistral-small-3.2-24b-instruct"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cyber-manipulation susceptibility pipeline with custom ontologies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Ontology inputs ───────────────────────────────────────────────────────
    ont = parser.add_argument_group("Ontology inputs (required)")
    ont.add_argument("--profile-json", required=True, help="Path to PROFILE ontology JSON")
    ont.add_argument("--attack-json",  required=True, help="Path to ATTACK ontology JSON")
    ont.add_argument("--opinion-json", required=True, help="Path to OPINION ontology JSON")

    # ── Run configuration ─────────────────────────────────────────────────────
    run = parser.add_argument_group("Run configuration")
    run.add_argument("--output-root",      default="evaluation/user_run_1")
    run.add_argument("--run-id",           default="user_run_1")
    run.add_argument("--n-profiles",       type=int,   default=40)
    run.add_argument("--seed",             type=int,   default=42)
    run.add_argument("--attack-leaves",    default=None,
                     help="Comma-01_separated ATTACK leaf labels to simulate (default: all)")
    run.add_argument("--focus-opinion-domain", default=None,
                     help="Restrict simulation to one top-level OPINION domain")
    run.add_argument("--max-opinion-leaves", type=int, default=None)

    # ── Model / LLM ───────────────────────────────────────────────────────────
    mdl = parser.add_argument_group("LLM model")
    mdl.add_argument("--openrouter-model",   default=_DEFAULT_MODEL)
    mdl.add_argument("--temperature",        type=float, default=0.15)
    mdl.add_argument("--max-repair-iter",    type=int,   default=2)
    mdl.add_argument("--max-concurrency",    type=int,   default=10)
    mdl.add_argument("--timeout-sec",        type=int,   default=90)

    # ── Pipeline control ──────────────────────────────────────────────────────
    ctl = parser.add_argument_group("Pipeline control")
    ctl.add_argument("--resume-from-stage",  default="01",
                     choices=["01","02","03","04","05","06","07","08","09"])
    ctl.add_argument("--stop-after-stage",   default="09",
                     choices=["01","02","03","04","05","06","07","08","09"])
    ctl.add_argument("--no-visuals",         action="store_true")
    ctl.add_argument("--no-report",          action="store_true")
    ctl.add_argument("--bootstrap-samples",  type=int, default=200)

    # ── Dev helpers ───────────────────────────────────────────────────────────
    dev = parser.add_argument_group("Dev helpers")
    dev.add_argument("--validate-only",  action="store_true",
                     help="Validate ontologies and exit (no simulation)")
    dev.add_argument("--dry-run",        action="store_true",
                     help="Print resolved config and exit (no simulation)")
    dev.add_argument("--log-level",      default="INFO")

    return parser.parse_args()


def _stage_report(report) -> None:
    print(report.summary())
    if not report.is_valid:
        sys.exit(1)
    if report.warnings:
        print(f"\n  {len(report.warnings)} warning(s) above — proceeding anyway.\n")


def _build_tmp_ontology_root(
    profile_json: Path,
    attack_json: Path,
    opinion_json: Path,
    tmp_dir: Path,
) -> Path:
    """Copy ontology files into a temporary directory with the expected structure:
        <tmp>/PROFILE/profile.json
        <tmp>/ATTACK/attack.json
        <tmp>/OPINION/opinion.json
    """
    for subdir, src, dest_name in [
        ("PROFILE", profile_json, "profile.json"),
        ("ATTACK",  attack_json,  "attack.json"),
        ("OPINION", opinion_json, "opinion.json"),
    ]:
        target = tmp_dir / subdir
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target / dest_name)
    return tmp_dir


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)-8s %(message)s")

    profile_json = Path(args.profile_json).resolve()
    attack_json  = Path(args.attack_json).resolve()
    opinion_json = Path(args.opinion_json).resolve()

    # ── Step 1: validate ──────────────────────────────────────────────────────
    print("\n=== Validating ontologies ===")
    report = validate_ontology_triplet(profile_json, attack_json, opinion_json)
    _stage_report(report)

    if args.validate_only:
        print("Validation passed. Exiting (--validate-only).")
        return

    # ── Step 2: build resolved config ────────────────────────────────────────
    output_root = str(PROJECT_ROOT / args.output_root)
    full_pipeline_script = str(
        PROJECT_ROOT / "src" / "backend" / "pipeline" / "full" / "run_full_pipeline.py"
    )

    with tempfile.TemporaryDirectory(prefix="cog_user_ont_") as tmp:
        tmp_path = Path(tmp)
        ont_root = _build_tmp_ontology_root(profile_json, attack_json, opinion_json, tmp_path)

        # Determine which attack leaves to use
        attack_leaves_arg: str | None = args.attack_leaves
        if attack_leaves_arg is None:
            # Auto-discover all attack leaves from the utils's JSON
            import json as _json
            with open(attack_json) as fh:
                attack_tree = _json.load(fh)
            try:
                from src.backend.utils.ontology_utils import flatten_leaf_paths
                raw_leaves = flatten_leaf_paths(attack_tree)
                # Use just the leaf label (last component) for the --attack-leaves arg
                leaf_labels = [p.split(">")[-1].strip() for p in raw_leaves]
                attack_leaves_arg = ",".join(leaf_labels)
            except Exception:
                attack_leaves_arg = None

        cmd = [
            sys.executable, full_pipeline_script,
            "--output-root",    output_root,
            "--run-id",         args.run_id,
            "--n-profiles",     str(args.n_profiles),
            "--seed",           str(args.seed),
            "--ontology-root",  str(ont_root),
            "--use-test-ontology",
            "--openrouter-model",   args.openrouter_model,
            "--temperature",        str(args.temperature),
            "--max-repair-iter",    str(args.max_repair_iter),
            "--max-concurrency",    str(args.max_concurrency),
            "--timeout-sec",        str(args.timeout_sec),
            "--resume-from-stage",  args.resume_from_stage,
            "--stop-after-stage",   args.stop_after_stage,
            "--bootstrap-samples",  str(args.bootstrap_samples),
            "--profile-generation-mode", "deterministic",
            "--self-supervise-attack-realism",
            "--self-supervise-opinion-coherence",
            "--log-level",          args.log_level,
        ]

        if attack_leaves_arg:
            cmd += ["--attack-leaves", attack_leaves_arg]
        if args.focus_opinion_domain:
            cmd += ["--focus-opinion-domain", args.focus_opinion_domain]
        if args.max_opinion_leaves is not None:
            cmd += ["--max-opinion-leaves", str(args.max_opinion_leaves)]
        if not args.no_visuals:
            cmd += ["--generate-visuals", "--export-static-figures"]
        else:
            cmd += ["--no-generate-visuals", "--no-export-static-figures"]
        if not args.no_report:
            cmd += ["--build-report"]
        else:
            cmd += ["--no-build-report"]

        config_preview = {
            "run_id": args.run_id,
            "output_root": output_root,
            "ontology_root": str(ont_root),
            "profile_json": str(profile_json),
            "attack_json": str(attack_json),
            "opinion_json": str(opinion_json),
            "attack_leaves": attack_leaves_arg,
            "n_profiles": args.n_profiles,
            "openrouter_model": args.openrouter_model,
        }

        print("\n=== Resolved configuration ===")
        print(json.dumps(config_preview, indent=2))

        if args.dry_run:
            print("\nDry run. Pipeline command would be:")
            print("  " + " ".join(cmd))
            return

        # ── Step 3: run pipeline ──────────────────────────────────────────────
        print("\n=== Running pipeline ===")
        LOGGER.info("Launching: %s", " ".join(cmd))
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if result.returncode != 0:
            LOGGER.error("Pipeline exited with code %s", result.returncode)
            sys.exit(result.returncode)

    print(f"\n=== Done. Outputs in: {output_root} ===")


if __name__ == "__main__":
    main()
