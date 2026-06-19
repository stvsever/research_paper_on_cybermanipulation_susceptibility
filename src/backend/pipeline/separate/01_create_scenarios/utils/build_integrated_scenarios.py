from __future__ import annotations

"""
Integrated scenario set (stage 01 → 02_integrated)
==================================================
Combines the three separated samples into the final 10,000-scenario design.

    one scenario = 1 profile configuration
                 + 1 DISARM-red Plan/Prepare/Execute attack triplet
                 + 1 opinion parent cluster (several leaf opinion nodes)

Construction keeps each design factor at (or near) maximal entropy while
PRESERVING the source distributions, so the joint set introduces no range
restriction or confounding:

  profiles        bijection: each of the 10,000 profiles is used exactly once
                  (perfectly uniform; preserves every balanced demographic margin)
  opinion cluster balanced allocation across all 69 parent clusters
                  (uniform = maximal cluster-layer entropy)
  attack triplet  entropy/coverage-preserving 10,000 subsample of the 48,991
                  filtered triplets, found by a multi-hop stochastic swap search
  pairing         the three factors are shuffled independently and zipped, so the
                  scenario factors are statistically independent (no confounding)

Outputs (in samples/02_integrated/):
  integrated_scenarios_10000.jsonl          the 10,000 scenarios (one per line)
  integrated_scenarios_10000.summary.json   entropy + independence + provenance
  integrated_scenarios_examples_3.json      three fully-resolved example scenarios

Run by file path:
  python .../utils/build_integrated_scenarios.py
"""

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[6]
for _p in (str(PROJECT_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from production_profile_sampling import load_population  # noqa: E402

STAGE = PROJECT_ROOT / "src" / "backend" / "pipeline" / "separate" / "01_create_scenarios"
SEP = STAGE / "samples" / "01_separated"
OUT = STAGE / "samples" / "02_integrated"
N_SCENARIOS = 10_000
SEED = 42
PHASES = ("Plan", "Prepare", "Execute")
# curated attacks force-included so the worked examples carry known formulations
CURATED_SRC = (42781, 58567, 14767)


def norm_entropy(counts) -> float:
    """Normalised Shannon entropy H/log(K) of a count vector over its support K."""
    c = np.asarray([v for v in counts if v > 0], dtype=float)
    if c.size <= 1:
        return 1.0 if c.size == 1 else 0.0
    p = c / c.sum()
    return float(-(p * np.log(p)).sum() / math.log(len(c)))


def tv_distance(sub_counts: Counter, full_p: dict) -> float:
    n = sum(sub_counts.values()) or 1
    keys = set(sub_counts) | set(full_p)
    return 0.5 * sum(abs(sub_counts.get(k, 0) / n - full_p.get(k, 0.0)) for k in keys)


def cramers_v(a, b) -> float:
    """Bias-corrected Cramér's V between two equal-length categorical sequences."""
    cats_a = {v: i for i, v in enumerate(dict.fromkeys(a))}
    cats_b = {v: i for i, v in enumerate(dict.fromkeys(b))}
    table = np.zeros((len(cats_a), len(cats_b)))
    for x, y in zip(a, b):
        table[cats_a[x], cats_b[y]] += 1
    n = table.sum()
    if n == 0 or table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    row = table.sum(1, keepdims=True); col = table.sum(0, keepdims=True)
    expected = row @ col / n
    chi2 = np.nansum((table - expected) ** 2 / np.where(expected == 0, np.nan, expected))
    phi2 = chi2 / n
    r, k = table.shape
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(math.sqrt(phi2c / denom)) if denom > 0 else 0.0


# ── attack subsample: entropy/coverage-preserving multi-hop search ───────────
def select_attacks(configs, leaf_by_id, rng, force_idx):
    """Pick N_SCENARIOS attack rows that preserve the source marginals and
    maximise leaf coverage, via a stochastic swap search (with multi-hop moves)."""
    n = len(configs)
    sig = np.array([c["opinion_manipulation_evidence"]["signal_total"] for c in configs])
    edges = np.quantile(sig, np.linspace(0, 1, 11))
    edges[-1] += 1e-6

    def feats(i):
        c = configs[i]
        lv = c["leaves"]
        return {
            "plan": leaf_by_id[lv["Plan"]]["secondary"],
            "prepare": leaf_by_id[lv["Prepare"]]["secondary"],
            "execute": leaf_by_id[lv["Execute"]]["secondary"],
            "route": c["opinion_manipulation_evidence"]["inclusion_route"],
            "sigbin": int(np.clip(np.searchsorted(edges, sig[i], "right") - 1, 0, 9)),
        }
    F = [feats(i) for i in range(n)]
    STRATA = ("plan", "prepare", "execute", "route", "sigbin")
    full_p = {s: {k: v / n for k, v in Counter(F[i][s] for i in range(n)).items()} for s in STRATA}
    leaves_of = [tuple(configs[i]["leaves"][p] for p in PHASES) for i in range(n)]
    total_leaves = len({x for tpl in leaves_of for x in tpl})

    chosen = set(force_idx)
    pool = list(set(range(n)) - chosen)
    rng.shuffle(pool)
    chosen |= set(pool[: N_SCENARIOS - len(chosen)])
    chosen = set(list(chosen)[:N_SCENARIOS])

    strat_counts = {s: Counter(F[i][s] for i in chosen) for s in STRATA}
    leaf_use = Counter(x for i in chosen for x in leaves_of[i])
    distinct = sum(1 for v in leaf_use.values() if v > 0)

    def cost():
        tv = sum(tv_distance(strat_counts[s], full_p[s]) for s in STRATA)
        return tv + 0.6 * (1 - distinct / total_leaves)

    chosen_list = list(chosen)
    pos = {i: p for p, i in enumerate(chosen_list)}
    out_pool = list(set(range(n)) - chosen)
    forced = set(force_idx)
    cur = cost()

    def apply_swap(old, new):
        nonlocal distinct
        for s in STRATA:
            strat_counts[s][F[old][s]] -= 1
            strat_counts[s][F[new][s]] += 1
        for x in leaves_of[old]:
            leaf_use[x] -= 1
            if leaf_use[x] == 0:
                distinct -= 1
        for x in leaves_of[new]:
            if leaf_use[x] == 0:
                distinct += 1
            leaf_use[x] += 1

    iters, improved = 60_000, 0
    for t in range(iters):
        oi = chosen_list[rng.integers(len(chosen_list))]
        if oi in forced:
            continue
        nj = out_pool[rng.integers(len(out_pool))]
        apply_swap(oi, nj)
        new_cost = cost()
        # occasional multi-hop: chain a second swap before deciding
        if new_cost >= cur and t % 7 == 0:
            ok = chosen_list[rng.integers(len(chosen_list))]
            if ok not in forced and ok != oi:
                nl = out_pool[rng.integers(len(out_pool))]
                apply_swap(ok, nl)
                c2 = cost()
                if c2 < cur:
                    p = pos[oi]; chosen_list[p] = nj; pos[nj] = p; del pos[oi]
                    q = pos[ok]; chosen_list[q] = nl; pos[nl] = q; del pos[ok]
                    out_pool.remove(nj); out_pool.remove(nl)
                    out_pool.append(oi); out_pool.append(ok)
                    cur = c2; improved += 1
                    continue
                apply_swap(nl, ok)  # revert second
        if new_cost < cur:
            p = pos[oi]; chosen_list[p] = nj; pos[nj] = p; del pos[oi]
            out_pool.remove(nj); out_pool.append(oi)
            cur = new_cost; improved += 1
        else:
            apply_swap(nj, oi)  # revert

    report = {
        "method": "entropy/coverage-preserving subsample via multi-hop stochastic swap search",
        "iterations": iters, "accepted_moves": improved,
        "final_cost": round(cur, 5),
        "leaf_coverage_of_filtered_pool": round(distinct / total_leaves, 4),
        "distinct_attack_leaves_used": distinct,
        "normalised_entropy_subsample": {s: round(norm_entropy(list(strat_counts[s].values())), 4) for s in STRATA},
        "normalised_entropy_full_filtered": {
            s: round(norm_entropy(list(Counter(F[i][s] for i in range(n)).values())), 4) for s in STRATA},
        "total_variation_to_full": {s: round(tv_distance(strat_counts[s], full_p[s]), 4) for s in STRATA},
    }
    return chosen_list, report


def full_attack(c, leaf_by_id):
    """Self-contained attack block: the triplet leaves resolved to full paths,
    labels and per-phase opinion-signal scores (no bare leaf ids left dangling)."""
    ev = c["opinion_manipulation_evidence"]
    triplet = {}
    for ph in PHASES:
        lid = c["leaves"][ph]
        lf = leaf_by_id[lid]
        ps = ev["phase_scores"][ph]
        triplet[ph] = {
            "leaf_id": lid,
            "secondary": lf["secondary"],
            "label": lf["label"],
            "path": lf["path"],
            "signal_score": ps["signal_score"],
            "confidence": ps["confidence"],
            "criteria": ps["criteria"],
        }
    return {
        "config_id": c["id"],
        "source_config_id": c["source_config_id"],
        "signal_total": ev["signal_total"],
        "inclusion_route": ev["inclusion_route"],
        "criteria": ev["criteria"],
        "triplet": triplet,
    }


def full_opinion(ck, cl):
    """Self-contained opinion block: every leaf of the parent cluster with its
    full path and adversarial direction (+1 amplify / -1 erode; 0 never sampled)."""
    leaves = [{"leaf": lf["leaf"], "path": lf["path"],
               "adversarial_direction": lf["adversarial_direction"]} for lf in cl["leaves"]]
    return {
        "key": ck,
        "family": cl["family"],
        "parent_name": cl["parent_name"],
        "n_leaves": cl["n_leaves"],
        "direction_summary": {
            "amplify_+1": sum(1 for lf in cl["leaves"] if lf["adversarial_direction"] == 1),
            "erode_-1": sum(1 for lf in cl["leaves"] if lf["adversarial_direction"] == -1),
        },
        "leaves": leaves,
    }


def main():
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    prof_bulk = json.loads((SEP / "profiles" / "production_profiles_maxent_10000.json").read_text())
    profiles = load_population(prof_bulk)
    opinion = json.loads((SEP / "opinions" / "opinion_targets_maxent_1000.json").read_text())
    clusters = opinion["clusters"]
    cluster_keys = list(clusters.keys())
    attack = json.loads((SEP / "attacks" / "red_plan_prepare_execute_opinion_effect_filtered.json").read_text())
    configs = attack["configurations"]
    leaf_by_id = {l["id"]: l for l in attack["leaf_catalog"]}
    by_src = {c["source_config_id"]: idx for idx, c in enumerate(configs)}

    assert len(profiles) >= N_SCENARIOS, "need >= 10K profiles"
    print(f"Loaded {len(profiles)} profiles · {len(clusters)} opinion clusters · {len(configs)} attack triplets")

    # 1. profiles — bijection (each used exactly once)
    prof_order = rng.permutation(len(profiles))[:N_SCENARIOS]

    # 2. opinion clusters — balanced uniform allocation over all clusters
    base, rem = divmod(N_SCENARIOS, len(cluster_keys))
    alloc = []
    for i, k in enumerate(cluster_keys):
        alloc += [k] * (base + (1 if i < rem else 0))
    rng.shuffle(alloc)

    # 3. attacks — entropy/coverage-preserving subsample (multi-hop search)
    force_idx = [by_src[s] for s in CURATED_SRC if s in by_src]
    print("Selecting attack subsample (multi-hop stochastic search)…")
    attack_idx, attack_report = select_attacks(configs, leaf_by_id, rng, force_idx)
    rng.shuffle(attack_idx)

    # 4. zip into scenarios (independent factors → joint independence)
    scenarios = []
    for i in range(N_SCENARIOS):
        prof = profiles[int(prof_order[i])]
        c = configs[attack_idx[i]]
        ck = alloc[i]
        cl = clusters[ck]
        scenarios.append({
            "scenario_id": f"scenario_{i + 1:05d}",
            "profile": prof,                          # FULL profile configuration
            "attack": full_attack(c, leaf_by_id),     # resolved Plan/Prepare/Execute triplet
            "opinion_cluster": full_opinion(ck, cl),  # all leaves + directions
        })

    # ── entropy + independence report ────────────────────────────────────────
    prof_region = []
    for i in range(N_SCENARIOS):
        p = profiles[int(prof_order[i])]
        prof_region.append(next((v for k, v in p["categorical_attributes"].items()
                                 if "broad_region" in k), "NA"))
    age_band = []
    for i in range(N_SCENARIOS):
        a = profiles[int(prof_order[i])]["demographics"]["age_years"]
        age_band.append(f"{(a // 16) * 16}-{(a // 16) * 16 + 15}")
    fam_seq = [s["opinion_cluster"]["family"] for s in scenarios]
    clu_seq = [s["opinion_cluster"]["parent_name"] for s in scenarios]
    exec_sec = [s["attack"]["triplet"]["Execute"]["secondary"] for s in scenarios]
    route_seq = [s["attack"]["inclusion_route"] for s in scenarios]

    cluster_usage = Counter(s["opinion_cluster"]["key"] for s in scenarios)
    family_usage = Counter(fam_seq)
    prof_usage = Counter(s["profile"]["profile_id"] for s in scenarios)

    summary = {
        "title": "Integrated opinion cognitive-warfare scenario set",
        "n_scenarios": N_SCENARIOS,
        "scenario_definition": "1 profile configuration + 1 DISARM-red Plan/Prepare/Execute "
                               "attack triplet + 1 opinion parent cluster (multiple leaf nodes)",
        "seed": SEED,
        "sources": {
            "profiles": {"file": "samples/01_separated/profiles/production_profiles_maxent_10000.json",
                         "n_available": len(profiles)},
            "opinions": {"file": "samples/01_separated/opinions/opinion_targets_maxent_1000.json",
                         "n_clusters": len(clusters),
                         "subtree": "Issue_Position_Taxonomy (issue-position opinion targets)"},
            "attacks": {"file": "samples/01_separated/attacks/red_plan_prepare_execute_opinion_effect_filtered.json",
                        "n_available": len(configs),
                        "ontology_note": "EXTERNAL DISARM-red attack ontology (not this repo's attack ontology)"},
        },
        "construction": {
            "profile_assignment": "bijection (each of 10,000 profiles used exactly once)",
            "opinion_cluster_assignment": f"balanced uniform allocation across all {len(clusters)} clusters",
            "attack_selection": "entropy/coverage-preserving 10,000 subsample via multi-hop stochastic search",
            "pairing": "factors shuffled independently and zipped (statistical independence)",
            "curated_attacks_force_included": list(CURATED_SRC),
        },
        "entropy_report": {
            "profile_usage_normalised_entropy": round(norm_entropy(list(prof_usage.values())), 6),
            "profile_each_used_once": max(prof_usage.values()) == 1,
            "opinion_cluster_normalised_entropy": round(norm_entropy(list(cluster_usage.values())), 6),
            "opinion_cluster_min_max_usage": [min(cluster_usage.values()), max(cluster_usage.values())],
            "opinion_family_normalised_entropy": round(norm_entropy(list(family_usage.values())), 6),
            "attack_subsample": attack_report,
        },
        "independence_cramers_v": {
            "profile_region__x__opinion_cluster": round(cramers_v(prof_region, clu_seq), 4),
            "profile_age_band__x__attack_execute_tactic": round(cramers_v(age_band, exec_sec), 4),
            "opinion_cluster__x__attack_route": round(cramers_v(clu_seq, route_seq), 4),
            "note": "Cramér's V near 0 ⇒ scenario factors are not confounded (good internal validity).",
        },
        "join_back": {
            "profile_id": "join to the profile columnar/example files",
            "attack.config_id / leaves": "join to attack leaf_catalog for full Plan/Prepare/Execute paths",
            "opinion_cluster.key": "join to opinion clusters for the full leaf list + adversarial directions",
        },
    }

    # ── write outputs ────────────────────────────────────────────────────────
    jsonl = OUT / "integrated_scenarios_10000.jsonl"
    with jsonl.open("w") as fh:
        for s in scenarios:
            fh.write(json.dumps(s) + "\n")
    (OUT / "integrated_scenarios_10000.summary.json").write_text(json.dumps(summary, indent=2))

    # examples built around the curated attacks (rich, fully resolved)
    examples = build_examples(scenarios, profiles, clusters, leaf_by_id, by_src)
    (OUT / "integrated_scenarios_examples_3.json").write_text(json.dumps(examples, indent=2))

    print(f"\n  wrote {jsonl.name}  ({len(scenarios):,} scenarios)")
    print(f"  wrote integrated_scenarios_10000.summary.json")
    print(f"  wrote integrated_scenarios_examples_3.json")
    er = summary["entropy_report"]
    print(f"\n  profile usage entropy   : {er['profile_usage_normalised_entropy']}  (each once: {er['profile_each_used_once']})")
    print(f"  opinion cluster entropy : {er['opinion_cluster_normalised_entropy']}  usage {er['opinion_cluster_min_max_usage']}")
    print(f"  attack leaf coverage    : {attack_report['leaf_coverage_of_filtered_pool']}")
    print(f"  attack subsample entropy: {attack_report['normalised_entropy_subsample']}")
    print(f"  independence (Cramér V) : {summary['independence_cramers_v']}")
    return summary, examples


def _profile_brief(p):
    d = p["demographics"]
    bf = {k: v["pct"] for k, v in d.get("big_five", {}).items()}
    ca = p["categorical_attributes"]
    pick = lambda frag: next((v for k, v in ca.items() if frag in k and v is not None), None)  # noqa: E731
    return {
        "profile_id": p["profile_id"],
        "age_years": d["age_years"],
        "sex_assigned_at_birth": d.get("sex_assigned_at_birth"),
        "gender_identity": d.get("gender_identity"),
        "world_region": pick("broad_region"),
        "highest_education": pick("highest_education"),
        "big_five_pct": {k: round(v, 1) for k, v in bf.items()},
    }


def build_examples(scenarios, profiles, clusters, leaf_by_id, by_src):
    """Human-readable digest of three representative scenarios. The full detail
    lives in every row of the jsonl; this view trims the profile to its headline
    traits for quick reading."""
    picks = []
    for src in CURATED_SRC:
        sc = next((s for s in scenarios if s["attack"]["source_config_id"] == src), None)
        if sc:
            picks.append(sc)
    out = []
    for sc in picks:
        a = sc["attack"]
        out.append({
            "scenario_id": sc["scenario_id"],
            "profile": _profile_brief(sc["profile"]),
            "opinion_cluster": {
                "key": sc["opinion_cluster"]["key"],
                "family": sc["opinion_cluster"]["family"],
                "parent_name": sc["opinion_cluster"]["parent_name"],
                "n_leaves": sc["opinion_cluster"]["n_leaves"],
                "leaves": [{"leaf": lf["leaf"], "adversarial_direction": lf["adversarial_direction"]}
                           for lf in sc["opinion_cluster"]["leaves"]],
            },
            "attack": {
                "config_id": a["config_id"], "source_config_id": a["source_config_id"],
                "signal_total": a["signal_total"], "inclusion_route": a["inclusion_route"],
                "triplet_paths": {ph: a["triplet"][ph]["path"] for ph in PHASES},
            },
        })
    return {"_meta": {"note": "Three representative scenarios, profile trimmed to headline traits. "
                              "Each jsonl row holds the full profile, attack and opinion detail."},
            "examples": out}


if __name__ == "__main__":
    main()
