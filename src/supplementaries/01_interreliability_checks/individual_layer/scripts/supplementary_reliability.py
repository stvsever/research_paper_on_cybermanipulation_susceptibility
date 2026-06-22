#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import os
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers 3D projection
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SUPPLEMENT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CONFIG_DIR = SUPPLEMENT_ROOT / "config"
INPUT_DIR = SUPPLEMENT_ROOT / "01_inputs"
RUNS_DIR = SUPPLEMENT_ROOT / "02_runs"
METRICS_DIR = SUPPLEMENT_ROOT / "03_metrics"
IMAGES_DIR = SUPPLEMENT_ROOT / "04_images"
TABLES_DIR = SUPPLEMENT_ROOT / "05_tables"
LOGS_DIR = SUPPLEMENT_ROOT / "logs"

ANALYSIS_CONFIG_PATH = CONFIG_DIR / "analysis_config.json"
MODELS_CONFIG_PATH = CONFIG_DIR / "models.json"
MEAN_ABS_DELTA_PLOT_CAP = 75.0

LOGGER = logging.getLogger("supplementary_reliability")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


@dataclass(frozen=True)
class ModelSpec:
    provider_key: str
    display_name: str
    openrouter_model: str
    family: str
    parameters_total_b: Optional[float]
    parameters_active_b: Optional[float]
    mmlu: Optional[float]
    mmlu_metric: Optional[str]
    mmlu_pro: Optional[float]
    mmlu_pro_metric: Optional[str]
    benchmark_source: str
    openrouter_source: str

    @property
    def latent_benchmark_score(self) -> Optional[float]:
        return self.mmlu_pro if self.mmlu_pro is not None else self.mmlu

    @property
    def latent_benchmark_type(self) -> str:
        return "MMLU-Pro" if self.mmlu_pro is not None else "MMLU"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False)


def _setup_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOGS_DIR / "supplementary_reliability.log", mode="a", encoding="utf-8"),
        ],
    )


def _load_stage_module(stage_dir: str, alias: str):
    path = PROJECT_ROOT / "src" / "backend" / "pipeline" / "separate" / stage_dir / "run_stage.py"
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load stage module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def load_config() -> Dict[str, Any]:
    return _read_json(ANALYSIS_CONFIG_PATH)


def load_models() -> List[ModelSpec]:
    raw = _read_json(MODELS_CONFIG_PATH)
    return [ModelSpec(**item) for item in raw]


def model_by_key(models: Sequence[ModelSpec]) -> Dict[str, ModelSpec]:
    return {m.provider_key: m for m in models}


def _patch_openrouter_client(config: Dict[str, Any]) -> None:
    """Patch the existing project client only inside this process.

    The main pipeline client disables reasoning for DeepSeek V4 Flash because
    reasoning-only responses break strict JSON scoring. The supplementary panel
    extends that same policy to Qwen3 and Nemotron, and constrains GPT-OSS
    reasoning tokens because reasoning is mandatory for that route.
    """
    import src.backend.agentic_framework.openrouter_client as client_mod

    cls = client_mod.OpenRouterClient
    if getattr(cls, "_supplementary_reliability_patched", False):
        return

    llm_params = config["llm_parameters"]
    reasoning_policy = llm_params.get("reasoning_policy", {})
    provider_routing = llm_params.get("provider_routing", {})
    top_p = llm_params.get("top_p")

    def patched_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1000,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.referer,
            "X-Title": self.app_title,
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if top_p is not None:
            payload["top_p"] = float(top_p)

        if self.model in reasoning_policy:
            payload["reasoning"] = reasoning_policy[self.model]
        elif self.model == "deepseek/deepseek-v4-flash":
            payload["reasoning"] = {"enabled": False, "exclude": True}
        if self.model in provider_routing:
            payload["provider"] = provider_routing[self.model]

        if response_format is not None:
            payload["response_format"] = response_format

        with httpx.Client(timeout=self.timeout_sec) as http_client:
            response: Optional[httpx.Response] = None
            for transport_attempt in range(1, 6):
                response = http_client.post(self.BASE_URL, headers=headers, json=payload)
                if response.status_code != 429:
                    break
                retry_after = response.headers.get("retry-after")
                if retry_after:
                    try:
                        sleep_s = min(45.0, float(retry_after))
                    except ValueError:
                        sleep_s = 5.0 * transport_attempt
                else:
                    sleep_s = min(45.0, 4.0 * transport_attempt)
                LOGGER.warning(
                    "OpenRouter 429 for %s; sleeping %.1fs before transport retry %d/5",
                    self.model,
                    sleep_s,
                    transport_attempt,
                )
                time.sleep(sleep_s)
            assert response is not None
            response.raise_for_status()
            data = response.json()

        content = cls._extract_content(data)
        return content, data

    cls.chat = patched_chat
    cls._supplementary_reliability_patched = True


def _patch_cluster_token_budget(config: Dict[str, Any]) -> None:
    import src.backend.agentic_framework.agents as agents_mod

    budget = config["llm_parameters"].get("cluster_output_token_budget", {})
    base = int(budget.get("base_tokens", 900))
    per_leaf = int(budget.get("per_leaf_tokens", 420))
    minimum = int(budget.get("min_tokens", 1800))
    maximum = int(budget.get("max_tokens", 9000))

    def supplementary_cluster_token_budget(n_leaves: int) -> int:
        return int(min(maximum, max(minimum, base + per_leaf * max(1, int(n_leaves)))))

    agents_mod._cluster_token_budget = supplementary_cluster_token_budget


def prepare_samples(force: bool = False) -> None:
    config = load_config()
    stage01 = _load_stage_module("01_create_scenarios", "supp_stage01")

    source_path = PROJECT_ROOT / config["scenario_source"]
    ontology_root = PROJECT_ROOT / config["ontology_root"]
    if not source_path.exists():
        raise RuntimeError(f"Scenario source not found: {source_path}")

    sample_specs = [
        (
            "cross_provider",
            INPUT_DIR / "01_cross_provider",
            int(config["cross_provider"]["n_scenarios"]),
            int(config["cross_provider"]["seed"]),
        ),
        (
            "test_retest",
            INPUT_DIR / "02_test_retest",
            int(config["test_retest"]["n_scenarios"]),
            int(config["test_retest"]["seed"]),
        ),
    ]

    manifest_rows: List[Dict[str, Any]] = []
    for panel_name, out_dir, n_scenarios, seed in sample_specs:
        scenarios_path = out_dir / "scenarios.jsonl"
        if scenarios_path.exists() and not force:
            LOGGER.info("Sample exists for %s: %s", panel_name, scenarios_path)
        else:
            LOGGER.info("Preparing %s sample: n=%d seed=%d", panel_name, n_scenarios, seed)
            cfg = stage01.Stage01Config(
                stage_name="create_scenarios",
                run_id=f"supplementary_{panel_name}_sample",
                seed=seed,
                use_test_ontology=False,
                ontology_root=str(ontology_root),
                n_scenarios=n_scenarios,
                integrated_scenarios_path=str(source_path),
                attack_ratio=1.0,
                profile_generation_mode="deterministic",
                enforce_compatibility_rules=False,
            )
            stage01.run_stage("", str(out_dir), cfg)

        rows = _read_jsonl(scenarios_path)
        id_rows = []
        for row in rows:
            cluster = row.get("opinion_cluster") or {}
            id_rows.append(
                {
                    "panel": panel_name,
                    "scenario_id": row.get("scenario_id"),
                    "profile_id": (row.get("profile") or {}).get("profile_id"),
                    "opinion_domain": cluster.get("parent_name"),
                    "n_opinion_leaves": cluster.get("n_leaves"),
                    "attack_leaf": row.get("attack_leaf"),
                    "attack_signal_total": (row.get("metadata") or {}).get("attack_signal_total"),
                    "attack_complexity_tier": (row.get("metadata") or {}).get("attack_complexity_tier"),
                }
            )
        id_path = out_dir / "scenario_ids.csv"
        pd.DataFrame(id_rows).to_csv(id_path, index=False)
        manifest_rows.append(
            {
                "panel": panel_name,
                "sample_dir": str(out_dir.relative_to(SUPPLEMENT_ROOT)),
                "scenario_jsonl": str(scenarios_path.relative_to(SUPPLEMENT_ROOT)),
                "scenario_ids_csv": str(id_path.relative_to(SUPPLEMENT_ROOT)),
                "n_scenarios": len(rows),
                "seed": seed,
            }
        )

    _write_json(
        INPUT_DIR / "sampling_manifest.json",
        {
            "scenario_source": config["scenario_source"],
            "sampling_logic": "Stage 01 integrated stratified-domain sampler, deterministic seeds",
            "samples": manifest_rows,
        },
    )


def _stage_done(output_dir: Path) -> bool:
    return (output_dir / "manifest.json").exists()


def _summarize_raw_usage(raw_dir: Path) -> Dict[str, Any]:
    total_cost = 0.0
    total_prompt = 0
    total_completion = 0
    total_reasoning = 0
    n_files = 0
    for path in raw_dir.glob("*.json"):
        try:
            payload = _read_json(path)
            usage = ((payload.get("raw_response") or {}).get("usage") or {})
        except Exception:
            continue
        n_files += 1
        total_cost += float(usage.get("cost") or 0.0)
        total_prompt += int(usage.get("prompt_tokens") or 0)
        total_completion += int(usage.get("completion_tokens") or 0)
        details = usage.get("completion_tokens_details") or {}
        total_reasoning += int(details.get("reasoning_tokens") or 0)
    return {
        "raw_response_files": n_files,
        "total_cost_usd": total_cost,
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "reasoning_tokens": total_reasoning,
    }


def _read_stage_summary(path: Path) -> Dict[str, Any]:
    return _read_json(path) if path.exists() else {}


def _run_single_model_panel(
    panel_name: str,
    run_label: str,
    input_scenarios: Path,
    model: ModelSpec,
    force: bool,
    allow_fallbacks: bool,
    max_concurrency: Optional[int],
) -> Dict[str, Any]:
    config = load_config()
    _patch_openrouter_client(config)
    _patch_cluster_token_budget(config)

    stage02 = _load_stage_module("02_assess_baseline_opinions", f"supp_stage02_{panel_name}_{run_label}")
    stage03 = _load_stage_module("03_run_opinion_attacks", f"supp_stage03_{panel_name}_{run_label}")
    stage04 = _load_stage_module("04_assess_post_attack_opinions", f"supp_stage04_{panel_name}_{run_label}")
    stage05 = _load_stage_module("05_compute_effectivity_deltas", f"supp_stage05_{panel_name}_{run_label}")

    params = config["llm_parameters"]
    ontology_root = PROJECT_ROOT / config["ontology_root"]
    run_dir = RUNS_DIR / panel_name / run_label
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    stage_root = run_dir / "stage_outputs"
    raw_dir = run_dir / "raw_llm"
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"supp_{panel_name}_{run_label}"
    concurrency = int(max_concurrency or params["max_concurrency"])
    common = {
        "run_id": run_id,
        "seed": int(config["cross_provider"]["seed"]),
        "openrouter_model": model.openrouter_model,
        "temperature": float(params["temperature"]),
        "max_repair_iter": int(params["max_repair_iter"]),
        "save_raw_llm": True,
        "raw_llm_dir": str(raw_dir),
        "timeout_sec": int(params["timeout_sec"]),
        "max_concurrency": concurrency,
    }

    t0 = time.time()
    LOGGER.info("[%s/%s] model=%s", panel_name, run_label, model.openrouter_model)

    out02 = stage_root / "02_assess_baseline_opinions"
    if force or not _stage_done(out02):
        LOGGER.info("[%s/%s] stage 02 baseline", panel_name, run_label)
        cfg02 = stage02.Stage02Config(
            stage_name="assess_baseline_opinions",
            self_supervise_opinion_coherence=False,
            coherence_threshold=0.72,
            **common,
        )
        stage02.run_stage(str(input_scenarios), str(out02), cfg02)

    out03 = stage_root / "03_run_opinion_attacks"
    if force or not _stage_done(out03):
        LOGGER.info("[%s/%s] stage 03 deterministic attack specs", panel_name, run_label)
        cfg03 = stage03.Stage03Config(
            stage_name="run_opinion_attacks",
            ontology_root=str(ontology_root),
            self_supervise_attack_realism=False,
            realism_threshold=0.72,
            **common,
        )
        stage03.run_stage(str(out02 / "scenarios_with_baseline.jsonl"), str(out03), cfg03)

    out04 = stage_root / "04_assess_post_attack_opinions"
    if force or not _stage_done(out04):
        LOGGER.info("[%s/%s] stage 04 post attack", panel_name, run_label)
        cfg04 = stage04.Stage04Config(
            stage_name="assess_post_attack_opinions",
            ontology_root=str(ontology_root),
            self_supervise_opinion_coherence=False,
            coherence_threshold=0.72,
            **common,
        )
        stage04.run_stage(str(out03 / "scenarios_with_attack_spec.jsonl"), str(out04), cfg04)

    out05 = stage_root / "05_compute_effectivity_deltas"
    if force or not _stage_done(out05):
        LOGGER.info("[%s/%s] stage 05 deltas", panel_name, run_label)
        cfg05 = stage05.Stage05Config(
            stage_name="compute_effectivity_deltas",
            run_id=run_id,
            seed=int(config["cross_provider"]["seed"]),
            use_test_ontology=False,
            ontology_root=str(ontology_root),
            primary_moderator="posthoc_profile_susceptibility_index",
        )
        stage05.run_stage(str(out04 / "scenarios_with_post.jsonl"), str(out05), cfg05)

    baseline_summary = _read_stage_summary(out02 / "baseline_summary.json")
    post_summary = _read_stage_summary(out04 / "post_attack_summary.json")
    raw_usage = _summarize_raw_usage(raw_dir)

    baseline_fallbacks = int(baseline_summary.get("fallback_count") or 0)
    post_fallbacks = int(post_summary.get("fallback_count") or 0)
    total_fallbacks = baseline_fallbacks + post_fallbacks
    status = {
        "panel": panel_name,
        "run_label": run_label,
        "provider_key": model.provider_key,
        "display_name": model.display_name,
        "openrouter_model": model.openrouter_model,
        "run_dir": str(run_dir.relative_to(SUPPLEMENT_ROOT)),
        "n_baseline_scenarios": int(baseline_summary.get("n_scenarios") or 0),
        "n_post_scenarios": int(post_summary.get("n_scenarios") or 0),
        "n_baseline_leaf_scores": int(baseline_summary.get("n_leaf_scores") or 0),
        "n_post_leaf_scores": int(post_summary.get("n_leaf_scores") or 0),
        "baseline_fallback_count": baseline_fallbacks,
        "post_fallback_count": post_fallbacks,
        "total_fallback_count": total_fallbacks,
        "duration_sec": round(time.time() - t0, 3),
        **raw_usage,
    }
    _write_json(run_dir / "run_status.json", status)

    if total_fallbacks and not allow_fallbacks:
        raise RuntimeError(
            f"{panel_name}/{run_label} had {total_fallbacks} deterministic fallback assessments. "
            f"Inspect {run_dir / 'run_status.json'}."
        )
    return status


def run_panels(
    panel: str,
    force: bool = False,
    allow_fallbacks: bool = False,
    model_keys: Optional[Sequence[str]] = None,
    max_concurrency: Optional[int] = None,
) -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set in .env or environment")

    config = load_config()
    models = load_models()
    by_key = model_by_key(models)
    selected_models = models
    if model_keys:
        selected_models = [by_key[key] for key in model_keys]

    statuses: List[Dict[str, Any]] = []
    if panel in {"cross_provider", "both"}:
        input_scenarios = INPUT_DIR / "01_cross_provider" / "scenarios.jsonl"
        if not input_scenarios.exists():
            prepare_samples(force=False)
        for model in selected_models:
            statuses.append(
                _run_single_model_panel(
                    panel_name="cross_provider",
                    run_label=model.provider_key,
                    input_scenarios=input_scenarios,
                    model=model,
                    force=force,
                    allow_fallbacks=allow_fallbacks,
                    max_concurrency=max_concurrency,
                )
            )

    if panel in {"test_retest", "both"}:
        input_scenarios = INPUT_DIR / "02_test_retest" / "scenarios.jsonl"
        if not input_scenarios.exists():
            prepare_samples(force=False)
        primary = by_key[config["primary_model"].replace("/", "_")] if config["primary_model"].replace("/", "_") in by_key else None
        if primary is None:
            primary = next(m for m in models if m.openrouter_model == config["primary_model"])
        for iteration in config["test_retest"]["iterations"]:
            statuses.append(
                _run_single_model_panel(
                    panel_name="test_retest",
                    run_label=iteration,
                    input_scenarios=input_scenarios,
                    model=primary,
                    force=force,
                    allow_fallbacks=allow_fallbacks,
                    max_concurrency=max_concurrency,
                )
            )

    if statuses:
        status_path = METRICS_DIR / "run_status_summary.csv"
        existing = pd.read_csv(status_path) if status_path.exists() else pd.DataFrame()
        combined = pd.concat([existing, pd.DataFrame(statuses)], ignore_index=True)
        combined = combined.drop_duplicates(subset=["panel", "run_label"], keep="last")
        combined.to_csv(status_path, index=False)


def _item_id_from_scenario_id(value: str) -> str:
    return str(value)


def _source_scenario_id(value: str) -> str:
    text = str(value)
    return text.split("__", 1)[0]


def _load_sem_table(panel_name: str, run_label: str, model: Optional[ModelSpec] = None) -> pd.DataFrame:
    path = RUNS_DIR / panel_name / run_label / "stage_outputs" / "05_compute_effectivity_deltas" / "sem_long_raw.csv"
    if not path.exists():
        raise RuntimeError(f"Missing SEM table: {path}")
    df = pd.read_csv(path)
    df["item_id"] = df["scenario_id"].map(_item_id_from_scenario_id)
    df["source_scenario_id"] = df["scenario_id"].map(_source_scenario_id)
    df["run_label"] = run_label
    if model is not None:
        df["provider_key"] = model.provider_key
        df["display_name"] = model.display_name
        df["openrouter_model"] = model.openrouter_model
    return df


def _collect_run_statuses() -> pd.DataFrame:
    rows = []
    for path in sorted(RUNS_DIR.glob("*/*/run_status.json")):
        try:
            rows.append(_read_json(path))
        except Exception as exc:
            LOGGER.warning("Could not read run status %s: %s", path, exc)
    status = pd.DataFrame(rows)
    if not status.empty:
        status.to_csv(METRICS_DIR / "run_status_summary.csv", index=False)
    return status


def _icc_absolute_agreement(matrix: pd.DataFrame) -> Dict[str, float]:
    """Two-way random-effects absolute-agreement ICC(A,1) and ICC(A,k)."""
    x = matrix.dropna(axis=0, how="any").to_numpy(dtype=float)
    n, k = x.shape if x.ndim == 2 else (0, 0)
    if n < 2 or k < 2:
        return {"n_items": float(n), "n_raters": float(k), "icc_a1": np.nan, "icc_ak": np.nan}

    grand = np.mean(x)
    row_means = np.mean(x, axis=1)
    col_means = np.mean(x, axis=0)
    ss_rows = k * np.sum((row_means - grand) ** 2)
    ss_cols = n * np.sum((col_means - grand) ** 2)
    ss_total = np.sum((x - grand) ** 2)
    ss_err = ss_total - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))

    denom_a1 = ms_rows + (k - 1) * ms_err + (k * (ms_cols - ms_err) / n)
    denom_ak = ms_rows + ((ms_cols - ms_err) / n)
    icc_a1 = (ms_rows - ms_err) / denom_a1 if denom_a1 else np.nan
    icc_ak = (ms_rows - ms_err) / denom_ak if denom_ak else np.nan
    return {
        "n_items": float(n),
        "n_raters": float(k),
        "ms_rows": float(ms_rows),
        "ms_cols": float(ms_cols),
        "ms_error": float(ms_err),
        "icc_a1": float(icc_a1),
        "icc_ak": float(icc_ak),
    }


def _corr_pair(x: pd.Series, y: pd.Series, method: str) -> float:
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return np.nan
    if method == "pearson":
        return float(stats.pearsonr(x[mask], y[mask]).statistic)
    if method == "spearman":
        return float(stats.spearmanr(x[mask], y[mask]).statistic)
    if method == "kendall":
        return float(stats.kendalltau(x[mask], y[mask]).statistic)
    raise ValueError(method)


def _safe_ci(series: pd.Series) -> Tuple[float, float]:
    values = series.dropna().to_numpy(dtype=float)
    if len(values) < 2:
        return (np.nan, np.nan)
    mean = float(np.mean(values))
    sem = float(stats.sem(values))
    half = 1.96 * sem
    return (mean - half, mean + half)


def analyze() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ["01_cross_provider", "02_test_retest", "03_model_benchmarks", "04_rank_robustness"]:
        (IMAGES_DIR / sub).mkdir(parents=True, exist_ok=True)

    models = load_models()
    config = load_config()
    status = _collect_run_statuses()

    model_rows = []
    for m in models:
        model_rows.append(
            {
                **m.__dict__,
                "latent_benchmark_score": m.latent_benchmark_score,
                "latent_benchmark_type": m.latent_benchmark_type,
            }
        )
    model_df = pd.DataFrame(model_rows)
    model_df.to_csv(TABLES_DIR / "model_benchmark_metadata.csv", index=False)

    cross_frames = []
    for model in models:
        run_path = RUNS_DIR / "cross_provider" / model.provider_key
        if (run_path / "stage_outputs" / "05_compute_effectivity_deltas" / "sem_long_raw.csv").exists():
            cross_frames.append(_load_sem_table("cross_provider", model.provider_key, model))
    if not cross_frames:
        raise RuntimeError("No cross-provider runs found. Run the panel first.")
    cross = pd.concat(cross_frames, ignore_index=True)
    cross.to_csv(METRICS_DIR / "cross_provider_long_scores.csv", index=False)

    retest_frames = []
    for iteration in config["test_retest"]["iterations"]:
        path = RUNS_DIR / "test_retest" / iteration
        if (path / "stage_outputs" / "05_compute_effectivity_deltas" / "sem_long_raw.csv").exists():
            df = _load_sem_table("test_retest", iteration)
            df["iteration"] = iteration
            retest_frames.append(df)
    if not retest_frames:
        raise RuntimeError("No test-retest runs found. Run the panel first.")
    retest = pd.concat(retest_frames, ignore_index=True)
    retest.to_csv(METRICS_DIR / "test_retest_long_scores.csv", index=False)

    outcomes = [
        "baseline_score",
        "post_score",
        "delta_score",
        "abs_delta_score",
        "adversarial_effectivity",
    ]
    cross_icc_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    for outcome in outcomes:
        pivot = cross.pivot_table(index="item_id", columns="provider_key", values=outcome, aggfunc="mean")
        icc = _icc_absolute_agreement(pivot)
        cross_icc_rows.append({"outcome": outcome, **icc})
        providers = list(pivot.columns)
        for i, left in enumerate(providers):
            for right in providers[i + 1 :]:
                pair_rows.append(
                    {
                        "outcome": outcome,
                        "provider_left": left,
                        "provider_right": right,
                        "pearson_r": _corr_pair(pivot[left], pivot[right], "pearson"),
                        "spearman_rho": _corr_pair(pivot[left], pivot[right], "spearman"),
                        "mae": float((pivot[left] - pivot[right]).abs().mean()),
                        "rmse": float(np.sqrt(((pivot[left] - pivot[right]) ** 2).mean())),
                        "mean_difference_left_minus_right": float((pivot[left] - pivot[right]).mean()),
                        "n_items": int((pivot[left].notna() & pivot[right].notna()).sum()),
                    }
                )
    cross_icc = pd.DataFrame(cross_icc_rows)
    pairwise = pd.DataFrame(pair_rows)
    cross_icc.to_csv(METRICS_DIR / "cross_provider_icc.csv", index=False)
    pairwise.to_csv(METRICS_DIR / "cross_provider_pairwise_reliability.csv", index=False)

    provider_summary = (
        cross.groupby(["provider_key", "display_name"], as_index=False)
        .agg(
            n_items=("item_id", "nunique"),
            mean_baseline_score=("baseline_score", "mean"),
            mean_post_score=("post_score", "mean"),
            mean_delta_score=("delta_score", "mean"),
            mean_abs_delta_score=("abs_delta_score", "mean"),
            mean_adversarial_effectivity=("adversarial_effectivity", "mean"),
            sd_adversarial_effectivity=("adversarial_effectivity", "std"),
            median_adversarial_effectivity=("adversarial_effectivity", "median"),
        )
        .merge(model_df.drop(columns=["display_name"]), on="provider_key", how="left")
    )
    ci_rows = []
    for provider_key, group in cross.groupby("provider_key"):
        lo, hi = _safe_ci(group["adversarial_effectivity"])
        ci_rows.append({"provider_key": provider_key, "ae_ci_low": lo, "ae_ci_high": hi})
    provider_summary = provider_summary.merge(pd.DataFrame(ci_rows), on="provider_key", how="left")
    if not status.empty:
        provider_status = status[status["panel"] == "cross_provider"][
            ["run_label", "total_fallback_count", "total_cost_usd", "raw_response_files", "reasoning_tokens"]
        ].rename(columns={"run_label": "provider_key"})
        provider_summary = provider_summary.merge(provider_status, on="provider_key", how="left")
    provider_reliability = _compute_provider_weighted_reliability(cross)
    provider_reliability.to_csv(METRICS_DIR / "cross_provider_weighted_reliability_by_model.csv", index=False)
    provider_summary = provider_summary.merge(provider_reliability, on="provider_key", how="left")
    provider_summary.to_csv(METRICS_DIR / "cross_provider_summary_by_model.csv", index=False)

    domain_provider = (
        cross.groupby(["opinion_domain", "provider_key"], as_index=False)
        .agg(
            mean_adversarial_effectivity=("adversarial_effectivity", "mean"),
            mean_abs_delta_score=("abs_delta_score", "mean"),
            n_items=("item_id", "nunique"),
        )
    )
    domain_provider.to_csv(METRICS_DIR / "cross_provider_domain_summary.csv", index=False)

    retest_summary_rows: List[Dict[str, Any]] = []
    retest_pair_rows = []
    if len(retest["iteration"].unique()) >= 2:
        it1, it2 = list(config["test_retest"]["iterations"])[:2]
        for outcome in outcomes:
            wide = retest.pivot_table(index="item_id", columns="iteration", values=outcome, aggfunc="mean")
            if it1 not in wide.columns or it2 not in wide.columns:
                continue
            diff = wide[it2] - wide[it1]
            icc = _icc_absolute_agreement(wide[[it1, it2]])
            retest_summary_rows.append(
                {
                    "outcome": outcome,
                    **icc,
                    "pearson_r": _corr_pair(wide[it1], wide[it2], "pearson"),
                    "spearman_rho": _corr_pair(wide[it1], wide[it2], "spearman"),
                    "mean_iteration_delta": float(diff.mean()),
                    "mae": float(diff.abs().mean()),
                    "rmse": float(np.sqrt((diff**2).mean())),
                    "n_items": int((wide[it1].notna() & wide[it2].notna()).sum()),
                }
            )
            tmp = wide[[it1, it2]].reset_index()
            tmp["outcome"] = outcome
            tmp["iteration_delta"] = tmp[it2] - tmp[it1]
            tmp["mean"] = (tmp[it2] + tmp[it1]) / 2
            retest_pair_rows.append(tmp)
    retest_summary = pd.DataFrame(retest_summary_rows)
    retest_pairs = pd.concat(retest_pair_rows, ignore_index=True) if retest_pair_rows else pd.DataFrame()
    retest_summary.to_csv(METRICS_DIR / "test_retest_reliability.csv", index=False)
    retest_pairs.to_csv(METRICS_DIR / "test_retest_paired_scores.csv", index=False)

    retest_domain = (
        retest.groupby(["opinion_domain", "iteration"], as_index=False)
        .agg(
            mean_adversarial_effectivity=("adversarial_effectivity", "mean"),
            mean_abs_delta_score=("abs_delta_score", "mean"),
            n_items=("item_id", "nunique"),
        )
    )
    retest_domain.to_csv(METRICS_DIR / "test_retest_domain_summary.csv", index=False)

    robustness_summary, robustness_pairwise = _compute_cross_provider_robustness(cross)
    robustness_summary.to_csv(METRICS_DIR / "cross_provider_rank_robustness.csv", index=False)
    robustness_pairwise.to_csv(METRICS_DIR / "cross_provider_rank_pairwise_reliability.csv", index=False)

    _make_cross_provider_figures(cross, provider_summary, pairwise, domain_provider)
    _make_cross_provider_robustness_figures(cross, provider_summary, robustness_summary, robustness_pairwise)
    _make_test_retest_figures(retest, retest_summary, retest_pairs, retest_domain, config)
    _write_readme(provider_summary, cross_icc, pairwise, retest_summary, robustness_summary)


def _provider_order(provider_summary: pd.DataFrame) -> List[str]:
    ordered = provider_summary.sort_values("latent_benchmark_score", ascending=True)["provider_key"].tolist()
    return ordered


def _display_lookup(provider_summary: pd.DataFrame) -> Dict[str, str]:
    return dict(zip(provider_summary["provider_key"], provider_summary["display_name"]))


def _provider_short_lookup() -> Dict[str, str]:
    return {
        "deepseek_v4_flash": "DeepSeek",
        "gpt_oss_120b": "GPT-OSS",
        "qwen3_32b": "Qwen3",
        "nemotron3_nano_30b": "Nemotron",
        "llama33_70b": "Llama 3.3",
    }


def _short_domain(value: str) -> str:
    mapping = {
        "Critical_Infrastructure_And_Energy_Sovereignty": "Infrastructure / energy",
        "Defense_And_National_Security": "Defense / security",
        "Democratic_Resilience_And_Institutions": "Democratic resilience",
        "Foreign_Policy_And_Geopolitics": "Foreign policy",
        "Information_Integrity_And_Platforms": "Information integrity",
        "Macroeconomic_And_Fiscal_Policy": "Macroeconomics",
        "Supranational_And_Regional_Integration": "Regional integration",
    }
    return mapping.get(str(value), str(value).replace("_", " "))


def _add_linear_fit(
    ax: plt.Axes,
    x: pd.Series,
    y: pd.Series,
    *,
    color: str = "#d04a02",
    label: str = "Linear fit",
    linewidth: float = 2.1,
    linestyle: str = "-",
    alpha: float = 0.95,
) -> None:
    values = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
        }
    ).replace([np.inf, -np.inf], np.nan)
    values = values.dropna()
    if len(values) < 3 or values["x"].nunique() < 2:
        return
    slope, intercept = np.polyfit(values["x"].to_numpy(), values["y"].to_numpy(), 1)
    xmin, xmax = ax.get_xlim()
    grid = np.linspace(xmin, xmax, 160)
    ax.plot(
        grid,
        intercept + slope * grid,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        label=label,
        zorder=8,
    )


def _padded_limits(values: Sequence[float], pad_fraction: float = 0.10, lower_floor: Optional[float] = None) -> Tuple[float, float]:
    arr = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return (0.0, 1.0)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    span = hi - lo
    if not np.isfinite(span) or span == 0:
        span = max(abs(hi), 1.0)
    pad = pad_fraction * span
    lo -= pad
    hi += pad
    if lower_floor is not None:
        lo = max(float(lower_floor), lo)
    return lo, hi


def _pairwise_matrix(pairwise: pd.DataFrame, value_col: str, order: Sequence[str]) -> pd.DataFrame:
    matrix = pd.DataFrame(np.eye(len(order)), index=order, columns=order, dtype=float)
    for _, row in pairwise.iterrows():
        left = row["provider_left"]
        right = row["provider_right"]
        if left in matrix.index and right in matrix.columns:
            matrix.loc[left, right] = row[value_col]
            matrix.loc[right, left] = row[value_col]
    return matrix


def _compute_cross_provider_robustness(cross: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    baseline = cross.pivot_table(index="item_id", columns="provider_key", values="baseline_score", aggfunc="mean")
    post = cross.pivot_table(index="item_id", columns="provider_key", values="post_score", aggfunc="mean")
    raw_mean_pb = (baseline + post) / 2
    raw_ae = cross.pivot_table(index="item_id", columns="provider_key", values="adversarial_effectivity", aggfunc="mean")
    transforms = {
        "raw_mean_pb": raw_mean_pb,
        "raw_ae": raw_ae,
        "percentile_mean_pb": raw_mean_pb.rank(axis=0, pct=True, method="average"),
        "percentile_ae": raw_ae.rank(axis=0, pct=True, method="average"),
    }

    summary_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    for transform_name, matrix in transforms.items():
        matrix = matrix.dropna(axis=0, how="any")
        if matrix.shape[1] < 2:
            continue
        icc = _icc_absolute_agreement(matrix)
        pair_values = []
        providers = list(matrix.columns)
        for i, left in enumerate(providers):
            for right in providers[i + 1 :]:
                pearson = _corr_pair(matrix[left], matrix[right], "pearson")
                spearman = _corr_pair(matrix[left], matrix[right], "spearman")
                kendall = _corr_pair(matrix[left], matrix[right], "kendall")
                mae = float((matrix[left] - matrix[right]).abs().mean())
                rmse = float(np.sqrt(((matrix[left] - matrix[right]) ** 2).mean()))
                row = {
                    "transform": transform_name,
                    "provider_left": left,
                    "provider_right": right,
                    "pearson_r": pearson,
                    "spearman_rho": spearman,
                    "kendall_tau": kendall,
                    "mae": mae,
                    "rmse": rmse,
                    "n_items": int((matrix[left].notna() & matrix[right].notna()).sum()),
                }
                pair_rows.append(row)
                pair_values.append(row)
        pair_df = pd.DataFrame(pair_values)
        summary_rows.append(
            {
                "transform": transform_name,
                **icc,
                "mean_pairwise_pearson": float(pair_df["pearson_r"].mean()) if not pair_df.empty else np.nan,
                "mean_pairwise_spearman": float(pair_df["spearman_rho"].mean()) if not pair_df.empty else np.nan,
                "mean_pairwise_kendall": float(pair_df["kendall_tau"].mean()) if not pair_df.empty else np.nan,
                "median_pairwise_spearman": float(pair_df["spearman_rho"].median()) if not pair_df.empty else np.nan,
                "mean_pairwise_mae": float(pair_df["mae"].mean()) if not pair_df.empty else np.nan,
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(pair_rows)


def _compute_provider_weighted_reliability(cross: pd.DataFrame) -> pd.DataFrame:
    """Provider-vs-leave-one-out consensus ICC averaged across key outcomes."""
    outcomes = {
        "baseline_score": "baseline_icc_vs_consensus",
        "post_score": "post_attack_icc_vs_consensus",
        "adversarial_effectivity": "delta_score_icc_vs_consensus",
    }
    provider_keys = sorted(cross["provider_key"].dropna().unique())
    rows: List[Dict[str, Any]] = []
    for provider in provider_keys:
        row: Dict[str, Any] = {"provider_key": provider}
        icc_values = []
        for outcome, metric_name in outcomes.items():
            pivot = cross.pivot_table(index="item_id", columns="provider_key", values=outcome, aggfunc="mean")
            if provider not in pivot.columns:
                row[metric_name] = np.nan
                continue
            other_cols = [col for col in pivot.columns if col != provider]
            if not other_cols:
                row[metric_name] = np.nan
                continue
            consensus = pivot[other_cols].mean(axis=1)
            icc = _icc_absolute_agreement(pd.DataFrame({"provider": pivot[provider], "consensus": consensus}))
            value = icc.get("icc_a1", np.nan)
            row[metric_name] = value
            if pd.notna(value):
                icc_values.append(float(value))
        row["weighted_reliability"] = float(np.mean(icc_values)) if icc_values else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _panel_label(ax: plt.Axes, label: str) -> None:
    """Bold APA-style panel tag (A, B, ...) in each panel's top-left corner.

    Uses ``text2D`` on 3D axes so the tag stays pinned to the panel frame rather
    than floating in data space."""
    placer = getattr(ax, "text2D", ax.text)
    placer(
        -0.04,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=17,
        fontweight="bold",
        va="top",
        ha="left",
        color="#111111",
        zorder=10,
    )


def _tag_panels(axes: Sequence[plt.Axes]) -> None:
    for ax, tag in zip(axes, "ABCDEFGH"):
        _panel_label(ax, tag)


def _make_cross_provider_figures(
    cross: pd.DataFrame,
    provider_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    domain_provider: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    palette = {
        "deepseek_v4_flash": "#1f77b4",
        "gpt_oss_120b": "#d62728",
        "qwen3_32b": "#2ca02c",
        "nemotron3_nano_30b": "#9467bd",
        "llama33_70b": "#ff7f0e",
    }
    order = _provider_order(provider_summary)
    display = _display_lookup(provider_summary)
    short = _provider_short_lookup()
    provider_label_order = [short.get(k, k) for k in order]
    provider_label_palette = {short.get(k, k): palette.get(k, "#333333") for k in order}
    cross = cross.copy()
    cross["provider_label"] = cross["provider_key"].map(short).fillna(cross["provider_key"])

    pivot_ae = cross.pivot_table(index="item_id", columns="provider_key", values="adversarial_effectivity", aggfunc="mean")
    density_rows = []
    for provider in pivot_ae.columns:
        other_cols = [col for col in pivot_ae.columns if col != provider]
        consensus = pivot_ae[other_cols].mean(axis=1)
        tmp = pd.DataFrame(
            {
                "provider_key": provider,
                "provider_label": short.get(provider, provider),
                "loo_consensus_ae": consensus,
                "provider_ae": pivot_ae[provider],
            }
        )
        tmp["provider_minus_consensus"] = tmp["provider_ae"] - tmp["loo_consensus_ae"]
        tmp["provider_consensus_mean"] = (tmp["provider_ae"] + tmp["loo_consensus_ae"]) / 2
        density_rows.append(tmp)
    density = pd.concat(density_rows, ignore_index=True)

    fig = plt.figure(figsize=(23, 14), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0], projection="3d")
    ps = provider_summary.sort_values("latent_benchmark_score")
    xs = ps["latent_benchmark_score"].astype(float).to_numpy()
    ys = ps["mean_abs_delta_score"].clip(upper=MEAN_ABS_DELTA_PLOT_CAP).astype(float).to_numpy()
    zs = ps["weighted_reliability"].astype(float).to_numpy()
    colors = [palette.get(k, "#333333") for k in ps["provider_key"]]
    ax0.plot(xs, ys, zs, color="#4c566a", linewidth=1.8, alpha=0.75)
    ax0.scatter(xs, ys, zs, s=130, c=colors, edgecolor="white", linewidth=1.0, depthshade=True)
    for _, row in ps.iterrows():
        true_abs = float(row["mean_abs_delta_score"])
        ax0.text(
            float(row["latent_benchmark_score"]),
            float(min(true_abs, MEAN_ABS_DELTA_PLOT_CAP)),
            float(row["weighted_reliability"]),
            " " + short.get(str(row["provider_key"]), str(row["display_name"]).replace(" Instruct", "")),
            fontsize=8,
        )
    ax0.set_xlabel("MMLU-Pro", labelpad=8)
    ax0.set_ylabel("Mean abs delta_score", labelpad=8)
    ax0.set_zlabel("Weighted reliability", labelpad=8)
    ax0.set_xlim(*_padded_limits(xs))
    ax0.set_ylim(*_padded_limits(ys, lower_floor=0))
    ax0.set_zlim(*_padded_limits(zs, lower_floor=0))
    ax0.set_title("MMLU-Pro reliability specification curve")
    ax0.view_init(elev=25, azim=-48)

    ax1 = fig.add_subplot(gs[0, 1])
    pearson = pairwise[pairwise["outcome"] == "adversarial_effectivity"]
    corr_mat = pd.DataFrame(np.eye(len(order)), index=order, columns=order, dtype=float)
    for _, row in pearson.iterrows():
        corr_mat.loc[row["provider_left"], row["provider_right"]] = row["pearson_r"]
        corr_mat.loc[row["provider_right"], row["provider_left"]] = row["pearson_r"]
    corr_mat = corr_mat.rename(index=short, columns=short)
    sns.heatmap(
        corr_mat,
        ax=ax1,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        cbar_kws={"label": "Pearson r"},
    )
    ax1.set_title("Pairwise provider agreement")
    ax1.tick_params(axis="x", labelrotation=35, labelsize=10)
    ax1.tick_params(axis="y", labelrotation=0, labelsize=10)

    ax2 = fig.add_subplot(gs[0, 2])
    sns.violinplot(
        data=cross,
        y="provider_label",
        x="adversarial_effectivity",
        hue="provider_label",
        order=provider_label_order,
        hue_order=provider_label_order,
        palette=provider_label_palette,
        legend=False,
        inner=None,
        cut=0,
        linewidth=1,
        ax=ax2,
    )
    sns.boxplot(
        data=cross,
        y="provider_label",
        x="adversarial_effectivity",
        order=provider_label_order,
        width=0.22,
        showcaps=True,
        showfliers=False,
        boxprops={"facecolor": "white", "zorder": 3},
        whiskerprops={"linewidth": 1.2},
        ax=ax2,
    )
    ax2.axvline(0, color="#222222", linewidth=1, linestyle="--")
    ax2.set_xscale("symlog", linthresh=20)
    ax2.set_title("Effectivity distributions")
    ax2.set_xlabel("Adversarial effectivity")
    ax2.set_ylabel("")

    ax3 = fig.add_subplot(gs[1, 0])
    hb = ax3.hexbin(
        density["loo_consensus_ae"],
        density["provider_ae"],
        gridsize=55,
        cmap="mako",
        mincnt=1,
        linewidths=0,
    )
    fig.colorbar(hb, ax=ax3, label="Provider-item density")
    lim_values = pd.concat([density["loo_consensus_ae"], density["provider_ae"]]).dropna()
    lo, hi = float(lim_values.quantile(0.01)), float(lim_values.quantile(0.99))
    pad = 0.05 * max(1.0, hi - lo)
    ax3.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#111111", linestyle="--", linewidth=1)
    ax3.set_xlim(lo - pad, hi + pad)
    ax3.set_ylim(lo - pad, hi + pad)
    _add_linear_fit(
        ax3,
        density["loo_consensus_ae"],
        density["provider_ae"],
        color="#e4572e",
        label="Linear fit",
    )
    ax3.set_title("Provider scores against leave-one-out consensus")
    ax3.set_xlabel("Consensus excluding focal provider")
    ax3.set_ylabel("Provider effectivity")

    ax4 = fig.add_subplot(gs[1, 1])
    for provider, group in density.groupby("provider_key"):
        ax4.scatter(
            group["provider_consensus_mean"],
            group["provider_minus_consensus"],
            s=18,
            alpha=0.33,
            color=palette.get(provider, "#333333"),
            label=short.get(provider, provider),
            linewidth=0,
        )
    ax4.axhline(0, color="#111111", linestyle="--", linewidth=1)
    _add_linear_fit(
        ax4,
        density["provider_consensus_mean"],
        density["provider_minus_consensus"],
        color="#111111",
        label="Linear fit",
        linewidth=1.8,
    )
    ax4.set_title("Provider bias against consensus")
    ax4.set_xlabel("Provider-consensus mean")
    ax4.set_ylabel("Provider - consensus")
    legend = ax4.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        title="Provider",
        fontsize=9,
        title_fontsize=9,
        markerscale=1.5,
        borderaxespad=0,
    )
    legend.get_frame().set_alpha(0.92)

    ax5 = fig.add_subplot(gs[1, 2])
    heat = domain_provider.pivot_table(
        index="opinion_domain",
        columns="provider_key",
        values="mean_adversarial_effectivity",
        aggfunc="mean",
    )
    heat = heat[order].rename(columns=short)
    heat.index = [_short_domain(idx) for idx in heat.index]
    sns.heatmap(
        heat,
        ax=ax5,
        cmap="Spectral_r",
        center=0,
        robust=True,
        annot=False,
        cbar_kws={"label": "Mean AE"},
    )
    ax5.set_title("Domain x provider effectivity")
    ax5.set_xlabel("")
    ax5.set_ylabel("")
    ax5.tick_params(axis="x", labelrotation=35, labelsize=10)
    ax5.tick_params(axis="y", labelsize=10)

    _tag_panels([ax0, ax1, ax2, ax3, ax4, ax5])
    out = IMAGES_DIR / "01_cross_provider" / "cross_provider_reliability_main.png"
    fig.savefig(out, dpi=260, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    _make_3d_specification_figure(provider_summary, palette)


def _make_cross_provider_robustness_figures(
    cross: pd.DataFrame,
    provider_summary: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    robustness_pairwise: pd.DataFrame,
) -> None:
    if robustness_summary.empty or robustness_pairwise.empty:
        return

    sns.set_theme(style="whitegrid", context="talk")
    palette = {
        "deepseek_v4_flash": "#1f77b4",
        "gpt_oss_120b": "#d62728",
        "qwen3_32b": "#2ca02c",
        "nemotron3_nano_30b": "#9467bd",
        "llama33_70b": "#ff7f0e",
    }
    order = _provider_order(provider_summary)
    short = _provider_short_lookup()
    label_map = {
        "raw_mean_pb": "Raw mean P/B",
        "raw_ae": "Raw AE",
        "percentile_mean_pb": "Percentile mean P/B",
        "percentile_ae": "Percentile AE",
    }
    transform_order = [
        "raw_mean_pb",
        "raw_ae",
        "percentile_mean_pb",
        "percentile_ae",
    ]

    pivot = cross.pivot_table(index="item_id", columns="provider_key", values="adversarial_effectivity", aggfunc="mean")
    rank = pivot.rank(axis=0, pct=True, method="average")
    density_rows = []
    for provider in rank.columns:
        other_cols = [col for col in rank.columns if col != provider]
        consensus = rank[other_cols].mean(axis=1)
        tmp = pd.DataFrame(
            {
                "item_id": rank.index,
                "provider_key": provider,
                "provider_label": short.get(provider, provider),
                "loo_consensus_rank": consensus,
                "provider_rank": rank[provider],
            }
        )
        tmp["provider_rank_minus_consensus"] = tmp["provider_rank"] - tmp["loo_consensus_rank"]
        density_rows.append(tmp)
    density = pd.concat(density_rows, ignore_index=True)

    domain_lookup = cross[["item_id", "opinion_domain"]].drop_duplicates()
    rank_long = rank.reset_index().melt(id_vars="item_id", var_name="provider_key", value_name="provider_rank")
    rank_long = rank_long.merge(domain_lookup, on="item_id", how="left")

    fig = plt.figure(figsize=(26, 14))
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.90, wspace=0.92, hspace=0.34)
    gs = GridSpec(2, 8, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0:2])
    summary_plot = robustness_summary[robustness_summary["transform"].isin(transform_order)].copy()
    summary_plot["transform_label"] = summary_plot["transform"].map(label_map)
    summary_long = summary_plot.melt(
        id_vars=["transform", "transform_label"],
        value_vars=["icc_a1", "mean_pairwise_spearman", "mean_pairwise_kendall"],
        var_name="metric",
        value_name="value",
    )
    metric_labels = {
        "icc_a1": "ICC(A,1)",
        "mean_pairwise_spearman": "Mean Spearman",
        "mean_pairwise_kendall": "Mean Kendall",
    }
    summary_long["metric_label"] = summary_long["metric"].map(metric_labels)
    sns.barplot(
        data=summary_long,
        y="transform_label",
        x="value",
        hue="metric_label",
        order=[label_map[t] for t in transform_order if t in set(summary_plot["transform"])],
        palette="Set2",
        ax=ax0,
    )
    ax0.axvline(0, color="#222222", linewidth=1)
    ax0.set_xlim(-0.10, 1.0)
    ax0.set_title("Reliability across raw and percentile metrics")
    ax0.set_xlabel("Agreement")
    ax0.set_ylabel("")
    ax0.legend(title="")

    raw_order = [key for key in order if key in set(pivot.columns)]

    ax1 = fig.add_subplot(gs[0, 2:5])
    percentile_mean_pairs = robustness_pairwise[robustness_pairwise["transform"] == "percentile_mean_pb"]
    percentile_mean_mat = _pairwise_matrix(percentile_mean_pairs, "pearson_r", raw_order).rename(index=short, columns=short)
    sns.heatmap(
        percentile_mean_mat,
        ax=ax1,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        cbar_kws={"label": "Pearson r"},
    )
    ax1.set_title("Pairwise percentile mean P/B agreement")
    ax1.tick_params(axis="x", labelrotation=35, labelsize=10)
    ax1.tick_params(axis="y", labelrotation=0, labelsize=10)

    ax2 = fig.add_subplot(gs[0, 5:8])
    raw_pairs = robustness_pairwise[robustness_pairwise["transform"] == "raw_ae"]
    kendall_mat = _pairwise_matrix(raw_pairs, "kendall_tau", raw_order).rename(index=short, columns=short)
    sns.heatmap(
        kendall_mat,
        ax=ax2,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        cbar_kws={"label": "Kendall tau"},
    )
    ax2.set_title("Pairwise rank-order agreement")
    ax2.tick_params(axis="x", labelrotation=35, labelsize=10)
    ax2.tick_params(axis="y", labelrotation=0, labelsize=10)

    ax3 = fig.add_subplot(gs[1, 0:2])
    hb = ax3.hexbin(
        density["loo_consensus_rank"],
        density["provider_rank"],
        gridsize=44,
        cmap="mako",
        mincnt=1,
        linewidths=0,
        extent=(0, 1, 0, 1),
    )
    fig.colorbar(hb, ax=ax3, label="Provider-item density")
    ax3.plot([0, 1], [0, 1], color="#111111", linestyle="--", linewidth=1)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    _add_linear_fit(
        ax3,
        density["loo_consensus_rank"],
        density["provider_rank"],
        color="#e4572e",
        label="Linear fit",
    )
    ax3.set_title("Provider percentile rank vs leave-one-out rank consensus")
    ax3.set_xlabel("Consensus percentile rank")
    ax3.set_ylabel("Provider percentile rank")

    ax4 = fig.add_subplot(gs[1, 2:5])
    provider_label_order = [short.get(k, k) for k in raw_order]
    sns.boxenplot(
        data=density,
        y="provider_label",
        x="provider_rank_minus_consensus",
        order=provider_label_order,
        color="#c9d8ed",
        linewidth=1,
        showfliers=False,
        ax=ax4,
    )
    sns.stripplot(
        data=density,
        y="provider_label",
        x="provider_rank_minus_consensus",
        order=provider_label_order,
        hue="provider_key",
        palette=palette,
        size=2.2,
        alpha=0.23,
        jitter=0.24,
        legend=False,
        ax=ax4,
    )
    ax4.axvline(0, color="#111111", linestyle="--", linewidth=1.1)
    ax4.set_xlim(-1, 1)
    ax4.set_title("Provider rank residuals against consensus")
    ax4.set_xlabel("Provider rank - consensus rank")
    ax4.set_ylabel("")

    ax5 = fig.add_subplot(gs[1, 5:8])
    heat = rank_long.pivot_table(
        index="opinion_domain",
        columns="provider_key",
        values="provider_rank",
        aggfunc="mean",
    )
    heat = heat[[key for key in raw_order if key in heat.columns]].rename(columns=short)
    heat.index = [_short_domain(idx) for idx in heat.index]
    sns.heatmap(
        heat,
        ax=ax5,
        cmap="Spectral_r",
        vmin=0,
        vmax=1,
        center=0.5,
        annot=False,
        cbar_kws={"label": "Mean percentile rank"},
    )
    ax5.set_title("Domain x provider rank profile")
    ax5.set_xlabel("")
    ax5.set_ylabel("")
    ax5.tick_params(axis="x", labelrotation=35, labelsize=10)
    ax5.tick_params(axis="y", labelsize=10)

    _tag_panels([ax0, ax1, ax2, ax3, ax4, ax5])
    out = IMAGES_DIR / "04_rank_robustness" / "cross_provider_rank_robustness_main.png"
    fig.savefig(out, dpi=260, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _make_3d_specification_figure(provider_summary: pd.DataFrame, palette: Dict[str, str]) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    ps = provider_summary.sort_values("latent_benchmark_score").copy()
    fig = plt.figure(figsize=(14, 10))
    fig.subplots_adjust(left=0.03, right=0.88, bottom=0.08, top=0.90)
    ax = fig.add_subplot(111, projection="3d")
    xs = ps["latent_benchmark_score"].astype(float).to_numpy()
    ys = ps["mean_abs_delta_score"].clip(upper=MEAN_ABS_DELTA_PLOT_CAP).astype(float).to_numpy()
    zs = ps["weighted_reliability"].astype(float).to_numpy()
    colors = [palette.get(k, "#333333") for k in ps["provider_key"]]
    ax.plot(xs, ys, zs, color="#2f3b52", linewidth=2.2, alpha=0.85)
    ax.scatter(xs, ys, zs, s=180, c=colors, edgecolor="white", linewidth=1.2)
    for _, row in ps.iterrows():
        label = _provider_short_lookup().get(str(row["provider_key"]), str(row["display_name"]).replace(" Instruct", ""))
        true_abs = float(row["mean_abs_delta_score"])
        display_abs = min(true_abs, MEAN_ABS_DELTA_PLOT_CAP)
        ax.text(
            float(row["latent_benchmark_score"]),
            float(display_abs),
            float(row["weighted_reliability"]),
            f"  {label} |d|={true_abs:.1f}",
            fontsize=9,
        )
    ax.set_xlabel("MMLU-Pro", labelpad=12)
    ax.set_ylabel("Mean abs delta_score", labelpad=14)
    ax.set_zlabel("Weighted reliability", labelpad=18)
    ax.set_xlim(*_padded_limits(xs))
    ax.set_ylim(*_padded_limits(ys, lower_floor=0))
    ax.set_zlim(*_padded_limits(zs, lower_floor=0))
    ax.set_title("3D specification curve: MMLU-Pro, movement, and reliability")
    ax.view_init(elev=24, azim=-58)
    legacy = IMAGES_DIR / "03_model_benchmarks" / "mmlu_3d_specification_curve.png"
    for old in [legacy, legacy.with_suffix(".pdf")]:
        old.unlink(missing_ok=True)
    out = IMAGES_DIR / "03_model_benchmarks" / "mmlu_pro_3d_specification_curve.png"
    fig.savefig(out, dpi=280, bbox_inches="tight", pad_inches=0.45)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.45)
    plt.close(fig)


def _make_test_retest_figures(
    retest: pd.DataFrame,
    retest_summary: pd.DataFrame,
    retest_pairs: pd.DataFrame,
    retest_domain: pd.DataFrame,
    config: Dict[str, Any],
) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    iterations = list(config["test_retest"]["iterations"])[:2]
    if len(iterations) < 2 or retest_pairs.empty:
        return
    it1, it2 = iterations
    ae = retest_pairs[retest_pairs["outcome"] == "adversarial_effectivity"].copy()
    baseline = retest_pairs[retest_pairs["outcome"] == "baseline_score"].copy()
    post = retest_pairs[retest_pairs["outcome"] == "post_score"].copy()

    fig = plt.figure(figsize=(21, 13), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)

    def plot_iteration_agreement(ax: plt.Axes, data: pd.DataFrame, title: str, cmap: str) -> None:
        hb = ax.hexbin(data[it1], data[it2], gridsize=42, cmap=cmap, mincnt=1, linewidths=0)
        fig.colorbar(hb, ax=ax, label="Item density")
        vals = pd.concat([data[it1], data[it2]]).dropna()
        lo, hi = float(vals.quantile(0.01)), float(vals.quantile(0.99))
        pad = 0.05 * max(1.0, hi - lo)
        ax.plot(
            [lo - pad, hi + pad],
            [lo - pad, hi + pad],
            color="#111111",
            linestyle="--",
            linewidth=1.2,
            label="Identity",
        )
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        _add_linear_fit(ax, data[it1], data[it2], color="#e4572e", label="Linear fit")
        ax.set_title(title)
        ax.set_xlabel("Iteration 1")
        ax.set_ylabel("Iteration 2")

    ax0 = fig.add_subplot(gs[0, 0])
    plot_iteration_agreement(ax0, baseline, "Baseline scores: iteration 1 vs 2", "magma")

    ax1 = fig.add_subplot(gs[0, 1])
    plot_iteration_agreement(ax1, post, "Post-attack scores: iteration 1 vs 2", "plasma")

    ax2 = fig.add_subplot(gs[0, 2])
    plot_iteration_agreement(ax2, ae, "delta_score: iteration 1 vs 2", "viridis")

    ax3 = fig.add_subplot(gs[1, 0])
    summary_plot = retest_summary[retest_summary["outcome"].isin(["baseline_score", "post_score", "adversarial_effectivity"])].copy()
    summary_plot["outcome_label"] = summary_plot["outcome"].map(
        {
            "baseline_score": "baseline_score",
            "post_score": "post_score",
            "adversarial_effectivity": "delta_score",
        }
    )
    summary_long = summary_plot.melt(
        id_vars=["outcome", "outcome_label"],
        value_vars=["pearson_r", "icc_a1", "icc_ak"],
        var_name="metric",
        value_name="value",
    )
    sns.barplot(
        data=summary_long,
        y="outcome_label",
        x="value",
        hue="metric",
        order=["baseline_score", "post_score", "delta_score"],
        ax=ax3,
        palette="Set2",
    )
    ax3.set_xlim(-0.05, 1.05)
    ax3.set_title("Retest reliability by outcome")
    ax3.set_xlabel("Reliability")
    ax3.set_ylabel("")
    ax3.legend(title="")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(ae["mean"], ae["iteration_delta"], s=22, alpha=0.45, color="#3b6fb6", linewidth=0)
    mean_diff = float(ae["iteration_delta"].mean())
    sd_diff = float(ae["iteration_delta"].std())
    ax4.axhline(mean_diff, color="#111111", linewidth=1.4, label="Mean")
    ax4.axhline(mean_diff + 1.96 * sd_diff, color="#8c2d04", linestyle="--", linewidth=1.1, label="95% limits")
    ax4.axhline(mean_diff - 1.96 * sd_diff, color="#8c2d04", linestyle="--", linewidth=1.1)
    _add_linear_fit(ax4, ae["mean"], ae["iteration_delta"], color="#e4572e", label="Linear fit")
    ax4.set_title("Bland-Altman: delta_score")
    ax4.set_xlabel("Mean of two iterations")
    ax4.set_ylabel("iteration_delta")
    ax4.legend(title="", fontsize=8, loc="upper left")

    ax5 = fig.add_subplot(gs[1, 2])
    heat = retest_domain.pivot_table(
        index="opinion_domain",
        columns="iteration",
        values="mean_adversarial_effectivity",
        aggfunc="mean",
    )
    if it1 in heat.columns and it2 in heat.columns:
        heat["iteration_delta"] = heat[it2] - heat[it1]
    heat.index = [_short_domain(idx) for idx in heat.index]
    sns.heatmap(
        heat[[c for c in [it1, it2, "iteration_delta"] if c in heat.columns]],
        ax=ax5,
        cmap="Spectral_r",
        center=0,
        robust=True,
        cbar_kws={"label": "Mean delta_score"},
    )
    ax5.set_title("Domain-level retest stability")
    ax5.set_xlabel("")
    ax5.set_ylabel("")
    ax5.tick_params(axis="x", labelrotation=40, labelsize=10)
    ax5.tick_params(axis="y", labelsize=10)
    for label in ax5.get_xticklabels():
        label.set_horizontalalignment("right")

    _tag_panels([ax0, ax1, ax2, ax3, ax4, ax5])
    out = IMAGES_DIR / "02_test_retest" / "test_retest_reliability_main.png"
    fig.savefig(out, dpi=260, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _fmt_float(value: Any, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "NA"


def _write_readme(
    provider_summary: pd.DataFrame,
    cross_icc: pd.DataFrame,
    pairwise: pd.DataFrame,
    retest_summary: pd.DataFrame,
    robustness_summary: pd.DataFrame,
) -> None:
    ae_icc = cross_icc[cross_icc["outcome"] == "adversarial_effectivity"]
    ae_icc_a1 = ae_icc["icc_a1"].iloc[0] if len(ae_icc) else np.nan
    ae_icc_ak = ae_icc["icc_ak"].iloc[0] if len(ae_icc) else np.nan
    retest_ae = retest_summary[retest_summary["outcome"] == "adversarial_effectivity"]
    retest_r = retest_ae["pearson_r"].iloc[0] if len(retest_ae) else np.nan
    retest_icc = retest_ae["icc_a1"].iloc[0] if len(retest_ae) else np.nan
    retest_delta_mean = retest_ae["mean_iteration_delta"].iloc[0] if len(retest_ae) else np.nan

    def robustness_value(transform: str, metric: str) -> float:
        if robustness_summary.empty:
            return np.nan
        row = robustness_summary[robustness_summary["transform"] == transform]
        if row.empty or metric not in row.columns:
            return np.nan
        return row[metric].iloc[0]

    raw_pb_icc = robustness_value("raw_mean_pb", "icc_a1")
    raw_ae_icc = robustness_value("raw_ae", "icc_a1")
    raw_ae_mean_spearman = robustness_value("raw_ae", "mean_pairwise_spearman")
    raw_ae_mean_kendall = robustness_value("raw_ae", "mean_pairwise_kendall")
    percentile_pb_icc = robustness_value("percentile_mean_pb", "icc_a1")
    percentile_pb_mean_pearson = robustness_value("percentile_mean_pb", "mean_pairwise_pearson")
    percentile_ae_icc = robustness_value("percentile_ae", "icc_a1")
    percentile_ae_mean_spearman = robustness_value("percentile_ae", "mean_pairwise_spearman")
    nemotron_row = provider_summary[provider_summary["provider_key"] == "nemotron3_nano_30b"]
    non_nemotron_abs = provider_summary[provider_summary["provider_key"] != "nemotron3_nano_30b"][
        "mean_abs_delta_score"
    ]
    nemotron_abs = nemotron_row["mean_abs_delta_score"].iloc[0] if len(nemotron_row) else np.nan
    non_nemotron_abs_low = non_nemotron_abs.min() if len(non_nemotron_abs) else np.nan
    non_nemotron_abs_high = non_nemotron_abs.max() if len(non_nemotron_abs) else np.nan

    model_table = provider_summary[
        [
            "display_name",
            "openrouter_model",
            "latent_benchmark_score",
            "latent_benchmark_type",
            "mean_abs_delta_score",
            "weighted_reliability",
            "n_items",
        ]
    ].rename(
        columns={
            "latent_benchmark_score": "mmlu_pro_score",
            "latent_benchmark_type": "benchmark_type",
        }
    ).sort_values("mmlu_pro_score", ascending=False)

    table_md = model_table.to_markdown(index=False, floatfmt=".3f")
    pair_ae = pairwise[pairwise["outcome"] == "adversarial_effectivity"].copy()
    best_pair = pair_ae.sort_values("pearson_r", ascending=False).head(1)
    weakest_pair = pair_ae.sort_values("pearson_r", ascending=True).head(1)
    best_text = "NA"
    weak_text = "NA"
    if len(best_pair):
        row = best_pair.iloc[0]
        best_text = f"{row['provider_left']} vs {row['provider_right']} (r={_fmt_float(row['pearson_r'])})"
    if len(weakest_pair):
        row = weakest_pair.iloc[0]
        weak_text = f"{row['provider_left']} vs {row['provider_right']} (r={_fmt_float(row['pearson_r'])})"

    readme = f"""# Supplementary Analysis 01: Interreliability Checks

This directory contains two tiny, controlled reliability checks for the
integrated individual-layer opinion simulation. No git commit or push is
performed by these scripts.

## Design

The two supplementary panels are:

1. **Test-retest repeated-run reproducibility** across two independent runs of
   the same primary model (`deepseek/deepseek-v4-flash`) on the same 30
   integrated scenarios.
2. **Cross-provider inter-rater reliability** across five low-cost OpenRouter
   models on the same 20 integrated scenarios.

Scenarios are sampled from
`src/backend/pipeline/separate/01_create_scenarios/samples/02_integrated/integrated_scenarios_10000.jsonl`
with the existing Stage 01 integrated stratified-domain sampler. The
test-retest panel uses 30 scenarios; both iterations receive exactly the same
scenario IDs. The cross-provider panel uses 20 scenarios; each provider receives
exactly the same scenario IDs.

Each selected scenario is scored with the existing cluster-batched individual
layer:

- Stage 02: baseline opinion cluster assessment.
- Stage 03: deterministic DISARM Plan/Prepare/Execute attack-vector
  specification; zero LLM calls.
- Stage 04: post-attack opinion cluster assessment.
- Stage 05: expansion to per-leaf deltas and adversarial effectivity.

The nominal LLM call budgets are:

- Test-retest: `30 scenarios x 2 phases x 2 iterations = 120 calls`.
- Cross-provider: `20 scenarios x 2 phases x 5 providers = 200 calls`.

Repair retries can add calls only when a model returns invalid JSON. The run
status table records raw response file counts, token counts, cost, reasoning
tokens, and deterministic fallback counts.

## Controlled Parameters

The supplementary runner uses the same prompt files and schema validators as the
main project. Parameters are stored in `config/analysis_config.json`.

- `temperature = 0.15`
- `top_p = 1.0`
- `max_repair_iter = 2`
- `timeout_sec = 180`
- `max_concurrency = 4` unless overridden on the CLI
- `response_format = {{"type": "json_object"}}`
- cluster output token cap is deterministic by leaf count:
  `min(9000, max(1800, 900 + 420 * n_leaves))`
- self-supervised coherence rewrites disabled to keep the nominal call design
  fixed
- no exposure-network stages are run

Reasoning is controlled per route to preserve strict JSON behavior:

- DeepSeek V4 Flash, Qwen3-32B, and Nemotron 3 Nano: reasoning disabled and
  excluded.
- GPT-OSS 120B: reasoning is mandatory, constrained to a small reasoning-token
  budget and excluded from returned content.
- Llama 3.3 70B Instruct: no explicit reasoning control.

## Main Figures

### Test-retest reliability

![Test-retest reliability](04_images/02_test_retest/test_retest_reliability_main.png)

Interpretation: baseline and post-attack scores are stable across repeated
runs, but `delta_score` is substantially less stable. This is a repeated-run
reproducibility check, not evidence of temporal drift.

### Cross-provider reliability

![Cross-provider reliability](04_images/01_cross_provider/cross_provider_reliability_main.png)

Interpretation: continuous cross-provider agreement is weak for `delta_score`.
Nemotron produces a much larger movement profile than the other providers,
which appears as a scale outlier in the distribution, consensus, and domain
panels.

### Cross-provider percentile and rank reliability

![Cross-provider percentile and rank reliability](04_images/04_rank_robustness/cross_provider_rank_robustness_main.png)

Interpretation: the four summary rows compare raw mean baseline/post scores,
raw `delta_score`, and their provider-wise percentile equivalents. Percentile
mean P/B agreement is higher than raw AE agreement, while AE rank-order
agreement remains weak. This separates general score-level comparability from
agreement about attack-induced movement.

### MMLU-Pro specification curve

![MMLU-Pro specification curve](04_images/03_model_benchmarks/mmlu_pro_3d_specification_curve.png)

Interpretation: the benchmark axis is consistently MMLU-Pro for all five
models. The movement axis is capped at `{_fmt_float(MEAN_ABS_DELTA_PLOT_CAP, 0)}`
for display, while raw values remain uncapped in the CSV files. The
non-Nemotron mean absolute `delta_score` range is `{_fmt_float(non_nemotron_abs_low)}`
to `{_fmt_float(non_nemotron_abs_high)}`, while Nemotron is
`{_fmt_float(nemotron_abs)}`. Weighted reliability is the equal-weight average
of provider-vs-consensus ICC(A,1) for baseline, post-attack, and `delta_score`.

## Current Results

Test-retest repeated-run `delta_score` reproducibility, where `delta_score`
is the de-duplicated adversarial-effectivity / absolute-delta outcome:
Pearson r = `{_fmt_float(retest_r)}`, ICC(A,1) = `{_fmt_float(retest_icc)}`,
and mean `iteration_delta` = `{_fmt_float(retest_delta_mean)}`.

Cross-provider continuous `delta_score` agreement remains low:
ICC(A,1) = `{_fmt_float(ae_icc_a1)}` and ICC(A,k) =
`{_fmt_float(ae_icc_ak)}`. Strongest pairwise provider correlation:
{best_text}. Weakest pairwise provider correlation: {weak_text}.

Percentile and rank diagnostics are saved as a separate sensitivity analysis,
not as a replacement for the continuous result. Raw mean P/B ICC(A,1) =
`{_fmt_float(raw_pb_icc)}`; raw AE ICC(A,1) = `{_fmt_float(raw_ae_icc)}`;
percentile mean P/B ICC(A,1) = `{_fmt_float(percentile_pb_icc)}` with mean
pairwise Pearson r = `{_fmt_float(percentile_pb_mean_pearson)}`; percentile AE
ICC(A,1) = `{_fmt_float(percentile_ae_icc)}` with mean pairwise Spearman rho =
`{_fmt_float(percentile_ae_mean_spearman)}`. Raw AE mean pairwise Spearman rho =
`{_fmt_float(raw_ae_mean_spearman)}` and mean Kendall tau =
`{_fmt_float(raw_ae_mean_kendall)}`.

## Model Benchmark Metadata

MMLU-Pro values are stored in `config/models.json` and exported to
`05_tables/model_benchmark_metadata.csv`. Original MMLU is retained as metadata
when available, but the specification curve uses MMLU-Pro for every provider.
GPT-OSS uses the Vals AI MMLU-Pro value because the OpenAI model card reports
MMLU but not MMLU-Pro.

{table_md}

Sources:

- DeepSeek V4 Flash: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
- GPT-OSS 120B MMLU-Pro: https://www.vals.ai/comparison?modelA=fireworks%2Fgpt-oss-120b
- GPT-OSS 120B model card: https://arxiv.org/html/2508.10925v1
- Qwen3 32B: https://arxiv.org/html/2505.09388v1
- Nemotron 3 Nano 30B-A3B: https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b/modelcard
- Llama 3.3 70B Instruct: https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/MODEL_CARD.md

## File Layout

- `config/`: fixed model list, benchmark metadata, and run parameters.
- `scripts/supplementary_reliability.py`: end-to-end runner and analysis code.
- `01_inputs/`: sampled scenario IDs and Stage 01 scenario manifests.
- `02_runs/`: per-provider and per-iteration stage outputs plus raw LLM
  provenance.
- `03_metrics/`: combined long tables and reliability metrics.
- `05_tables/`: clean metadata tables for manuscript use.
- `04_images/`: PNG/PDF figures organized by analysis.

## Re-run

From the repository root:

```bash
python src/supplementaries/01_interreliability_checks/scripts/supplementary_reliability.py all
```

Useful narrower commands:

```bash
python src/supplementaries/01_interreliability_checks/scripts/supplementary_reliability.py prepare
python src/supplementaries/01_interreliability_checks/scripts/supplementary_reliability.py run --panel cross_provider
python src/supplementaries/01_interreliability_checks/scripts/supplementary_reliability.py run --panel test_retest
python src/supplementaries/01_interreliability_checks/scripts/supplementary_reliability.py analyze
```
"""
    (SUPPLEMENT_ROOT / "README.md").write_text(readme, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supplementary interreliability checks")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="Sample tiny scenario panels")
    p_prepare.add_argument("--force", action="store_true")

    p_run = sub.add_parser("run", help="Run LLM panels")
    p_run.add_argument("--panel", choices=["cross_provider", "test_retest", "both"], default="both")
    p_run.add_argument("--force", action="store_true")
    p_run.add_argument("--allow-fallbacks", action="store_true")
    p_run.add_argument("--models", default=None, help="Comma-separated provider keys for cross-provider runs")
    p_run.add_argument("--max-concurrency", type=int, default=None)

    p_analyze = sub.add_parser("analyze", help="Analyze completed panel outputs and regenerate figures")
    p_analyze.set_defaults()

    p_all = sub.add_parser("all", help="Prepare, run, and analyze")
    p_all.add_argument("--force", action="store_true")
    p_all.add_argument("--allow-fallbacks", action="store_true")
    p_all.add_argument("--max-concurrency", type=int, default=None)

    return parser.parse_args()


def main() -> None:
    _setup_logging()
    args = parse_args()

    if args.command == "prepare":
        prepare_samples(force=args.force)
    elif args.command == "run":
        model_keys = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None
        run_panels(
            panel=args.panel,
            force=args.force,
            allow_fallbacks=args.allow_fallbacks,
            model_keys=model_keys,
            max_concurrency=args.max_concurrency,
        )
    elif args.command == "analyze":
        analyze()
    elif args.command == "all":
        prepare_samples(force=args.force)
        run_panels(
            panel="both",
            force=args.force,
            allow_fallbacks=args.allow_fallbacks,
            max_concurrency=args.max_concurrency,
        )
        analyze()
    else:
        raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
