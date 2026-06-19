from __future__ import annotations

import json
import re
import sys
from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

from profile_network_workbench_api.schemas import AttackOption, AttackOptionsResponse, OntologyMode, OpinionLeafOption
from profile_network_workbench_api.settings import load_settings


class WorkbenchNotFoundError(FileNotFoundError):
    pass


def safe_run_id(run_id: str) -> str:
    if not re.fullmatch(r"run_[A-Za-z0-9_-]+", run_id):
        raise WorkbenchNotFoundError(f"Unsupported run id: {run_id}")
    return run_id


def pipeline_config_path_for_run(run_id: str) -> Path:
    settings = load_settings()
    safe_id = safe_run_id(run_id)
    candidates = [
        settings.evaluation_path / safe_id / "config" / "pipeline_config.json",
        settings.evaluation_path / "tests" / safe_id / "config" / "pipeline_config.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def read_pipeline_config(run_id: str) -> dict[str, Any]:
    path = pipeline_config_path_for_run(run_id)
    if not path.exists():
        raise WorkbenchNotFoundError(f"Pipeline config not found: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return dict(payload or {})


def mode_from_config(payload: dict[str, Any]) -> OntologyMode:
    return "test" if bool(payload.get("use_test_ontology", True)) else "production"


def ontology_root(mode: OntologyMode) -> Path:
    settings = load_settings()
    return settings.test_ontology_root if mode == "test" else settings.production_ontology_root


def _ensure_backend_path() -> None:
    lab_root = load_settings().lab_root
    if str(lab_root) not in sys.path:
        sys.path.insert(0, str(lab_root))


@lru_cache(maxsize=1)
def load_stage01_helpers() -> dict[str, Any]:
    _ensure_backend_path()
    path = load_settings().lab_root / "src" / "backend" / "pipeline" / "separate" / "01_create_scenarios" / "run_stage.py"
    spec = spec_from_file_location("profile_network_stage01", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Stage 01 helpers from {path}")
    module = module_from_spec(spec)
    sys.modules.setdefault("profile_network_stage01", module)
    spec.loader.exec_module(module)
    return {
        "Stage01Config": module.Stage01Config,
        "_allocate_profiles": module._allocate_profiles,
        "_resolve_attack_leaves": module._resolve_attack_leaves,
        "_scenario_attack_leaves": module._scenario_attack_leaves,
        "_select_opinion_leaves": module._select_opinion_leaves,
        "_target_profile_count": module._target_profile_count,
    }


def load_backend_helpers() -> dict[str, Any]:
    _ensure_backend_path()
    from src.backend.utils.scenario.compatibility_rules import (
        evaluate_scenario_admissibility,
        load_attack_metadata_index,
        load_opinion_metadata_index,
    )
    from src.backend.utils.ontology_utils import flatten_leaf_paths, load_adversarial_directions_from_opinion, load_ontology_triplet
    from src.backend.utils.scenario.scenario_realism import build_attack_context, extract_leaf_label, extract_opinion_domain

    return {
        "build_attack_context": build_attack_context,
        "evaluate_scenario_admissibility": evaluate_scenario_admissibility,
        "extract_leaf_label": extract_leaf_label,
        "extract_opinion_domain": extract_opinion_domain,
        "flatten_leaf_paths": flatten_leaf_paths,
        "load_adversarial_directions_from_opinion": load_adversarial_directions_from_opinion,
        "load_attack_metadata_index": load_attack_metadata_index,
        "load_ontology_triplet": load_ontology_triplet,
        "load_opinion_metadata_index": load_opinion_metadata_index,
    }


def load_ontologies(mode: OntologyMode) -> dict[str, Any]:
    root = ontology_root(mode)
    if not root.exists():
        raise WorkbenchNotFoundError(f"Ontology root not found: {root}")
    return load_backend_helpers()["load_ontology_triplet"](root)


def split_csv(value: Any) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def stage01_config(payload: dict[str, Any], mode: OntologyMode) -> Any:
    helpers = load_stage01_helpers()
    config_payload = dict(payload)
    config_payload["stage_name"] = "create_scenarios"
    config_payload["ontology_root"] = str(ontology_root(mode))
    config_payload["use_test_ontology"] = mode == "test"
    config_payload["profile_generation_mode"] = "deterministic"
    return helpers["Stage01Config"](**config_payload)


def selected_opinion_leaves(ontologies: dict[str, Any], config: Any) -> list[str]:
    helpers = load_stage01_helpers()
    backend = load_backend_helpers()
    opinion_leaves = list(backend["flatten_leaf_paths"](ontologies["OPINION"]))
    opinion_index = dict(backend["load_opinion_metadata_index"](ontologies["OPINION"]))

    candidate_leaves = list(opinion_leaves)
    if getattr(config, "drop_direction_neutral_opinions", False):
        directional = [
            leaf
            for leaf in candidate_leaves
            if leaf in opinion_index and int(opinion_index[leaf].adversarial_direction) != 0
        ]
        if directional:
            candidate_leaves = directional

    if getattr(config, "opinion_leaves", None):
        selected: list[str] = []
        for requested in split_csv(config.opinion_leaves):
            requested_lower = requested.lower()
            matched = [leaf for leaf in candidate_leaves if requested_lower in leaf.lower()]
            if not matched:
                raise WorkbenchNotFoundError(f"Opinion leaf not found among available leaves: {requested}")
            selected.append(matched[0])
        return selected

    return helpers["_select_opinion_leaves"](
        opinion_leaves=candidate_leaves,
        focus_opinion_domain=getattr(config, "focus_opinion_domain", None),
        max_opinion_leaves=getattr(config, "max_opinion_leaves", None),
    )


def reconstruct_profile_bundles(run_id: str) -> tuple[dict[str, Any], Any, dict[str, Any], list[dict[str, Any]], list[str]]:
    payload = read_pipeline_config(run_id)
    mode = mode_from_config(payload)
    ontologies = load_ontologies(mode)
    config = stage01_config(payload, mode)
    helpers = load_stage01_helpers()

    attack_leaves = helpers["_scenario_attack_leaves"](ontologies["ATTACK"])
    selected_attacks = helpers["_resolve_attack_leaves"](
        attack_leaves,
        getattr(config, "attack_leaves", None),
        getattr(config, "attack_leaf", None),
    )
    opinions = selected_opinion_leaves(ontologies, config)
    target_profiles = helpers["_target_profile_count"](config, opinions, len(selected_attacks))
    profile_bundles = helpers["_allocate_profiles"](
        profile_tree=ontologies["PROFILE"],
        config=config,
        llm_generator=lambda *args, **kwargs: None,
        target_profiles=target_profiles,
    )
    return payload, config, ontologies, profile_bundles, opinions


def selected_attack_leaves(ontologies: dict[str, Any], config: Any) -> list[str]:
    helpers = load_stage01_helpers()
    attack_leaves = helpers["_scenario_attack_leaves"](ontologies["ATTACK"])
    return helpers["_resolve_attack_leaves"](
        attack_leaves,
        getattr(config, "attack_leaves", None),
        getattr(config, "attack_leaf", None),
    )


def canonical_leaf_path(value: str, leaves: list[str]) -> str | None:
    if value in leaves:
        return value
    value_lower = value.lower().strip()
    matches = [
        leaf
        for leaf in leaves
        if leaf.lower() == value_lower or leaf.split(">")[-1].strip().lower() == value_lower
    ]
    if len(matches) == 1:
        return matches[0]
    suffix_matches = [leaf for leaf in leaves if leaf.lower().endswith(value_lower)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    return None


def opinion_options(opinions: list[str]) -> list[OpinionLeafOption]:
    extract_domain = load_backend_helpers()["extract_opinion_domain"]
    options: list[OpinionLeafOption] = []
    for path in opinions:
        parts = [part.strip() for part in path.split(">") if part.strip()]
        options.append(
            OpinionLeafOption(
                path=path,
                label=parts[-1] if parts else path,
                domain=str(extract_domain(path) or "") or None,
            )
        )
    return options


def attack_options(run_id: str, opinion_leaf: str) -> AttackOptionsResponse:
    payload, config, ontologies, profile_bundles, opinions = reconstruct_profile_bundles(run_id)
    canonical_opinion_leaf = canonical_leaf_path(opinion_leaf, opinions)
    if canonical_opinion_leaf is None:
        raise WorkbenchNotFoundError("Opinion leaf must be one of the run-configured opinion leaves")
    opinion_leaf = canonical_opinion_leaf

    backend = load_backend_helpers()
    attack_index = dict(backend["load_attack_metadata_index"](ontologies["ATTACK"]))
    opinion_index = dict(backend["load_opinion_metadata_index"](ontologies["OPINION"]))
    profile = profile_bundles[0]["profile_result"].profile if profile_bundles else None
    opinion_meta = opinion_index.get(opinion_leaf)
    options: list[AttackOption] = []

    for attack_leaf in selected_attack_leaves(ontologies, config):
        meta = attack_index.get(attack_leaf)
        notes: list[str] = []
        compatible = True
        if meta is not None and opinion_meta is not None and profile is not None:
            admissibility = backend["evaluate_scenario_admissibility"](
                profile=profile,
                attack_meta=meta,
                opinion_meta=opinion_meta,
            )
            compatible = bool(admissibility.admissible)
            notes = list(admissibility.notes) + list(admissibility.excluded_reasons)
        elif meta is None:
            notes = ["Attack metadata unavailable for this ontology leaf."]

        label = attack_leaf.split(">")[-1].strip()
        options.append(
            AttackOption(
                path=attack_leaf,
                label=label,
                family=str(getattr(meta, "family", "")) if meta is not None else "",
                complexity_tier=str(getattr(meta, "complexity_tier", "")) if meta is not None else "",
                temporal_horizon=str(getattr(meta, "temporal_horizon", "")) if meta is not None else "",
                epistemic_target=str(getattr(meta, "epistemic_target", "")) if meta is not None else "",
                compatible=compatible,
                notes=notes,
            )
        )

    return AttackOptionsResponse(run_id=run_id, opinion_leaf=opinion_leaf, attack_options=options)
