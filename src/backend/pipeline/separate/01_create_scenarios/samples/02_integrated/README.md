# Integrated scenario set (10,000 scenarios)

This folder contains the final integrated design for the opinion cognitive-warfare
study. It combines the three separated samples (profiles, opinions, attacks) into
one set of 10,000 scenarios that downstream stages consume directly.

## What one scenario is

Every scenario is a combination of three building blocks, one drawn from each
sub-ontology:

```
one scenario = 1 profile configuration
             + 1 DISARM-red Plan / Prepare / Execute attack triplet
             + 1 opinion parent cluster (an issue domain, several issue positions)
```

In plain terms, each scenario asks: for this kind of person, if an adversary runs
this specific attack, what happens to this related group of issue positions?

Two scoping choices shape this version. The profile ontology's
`Issue_Position_Taxonomy` (223 fine issue-position items) is excluded from profile
sampling, because issue positions are stances an adversary tries to shift, not
stable background traits. The opinions are then sampled only from the opinion
ontology's `Issue_Position_Taxonomy` subtree, giving 106 adversarially directional
issue positions grouped into 7 issue domains.

## Files in this folder

| File | What it holds |
|------|----------------|
| `integrated_scenarios_10000.jsonl` | The 10,000 scenarios, one JSON object per line. Each row is fully self-contained (see the schema below), so it is large (about 0.4 GB). |
| `integrated_scenarios_10000.summary.json` | Provenance, the entropy report, and the independence checks. Small, read this first. |
| `integrated_scenarios_examples_3.json` | Three representative scenarios with the profile trimmed to headline traits, for quick reading. |
| `README.md` | This document. |

Overview figures live in `../../overview/integrated/`, split into `pngs/` and
`htmls/`. The HTML versions are interactive. The flagship is
`integrated_factor_explorer_3d`, a single 3D view whose X, Y and Z axes are each
chosen from the real ontology hierarchy (a three-level cascade: ontology and
domain, then construct, then factor), covering every profile, attack and opinion
factor. Colour is the opinion issue domain, and the legend is click-toggleable.
See the overview README in that folder for the full guide.

## Each row stands on its own (schema)

Every row carries the full detail, so a single line is interpretable without
opening any other file. The leaf id numbers are kept and also resolved to their
human-readable paths, so nothing is left as a bare code.

```
{
  "scenario_id": "scenario_00001",

  "profile": {                      full profile configuration, exactly as sampled
    "profile_id": "...",
    "demographics": { "age_years": .., "sex_assigned_at_birth": "..",
                      "gender_identity": "..", "big_five": { ... pct + level ... } },
    "categorical_attributes": { ... 268 labelled traits ... },
    "numeric_attributes":     { ... 248 fine-grained scores (no issue positions) ... },
    "scalar_attributes": { ... }, "identifiers": { ... }
  },

  "attack": {                       DISARM-red Plan/Prepare/Execute triplet
    "config_id": .., "source_config_id": .., "signal_total": ..,
    "inclusion_route": "..", "criteria": [ .. opinion-manipulation mechanisms .. ],
    "triplet": {
      "Plan":    { "leaf_id": 3168, "secondary": "..", "label": "..",
                   "path": ["Plan", "..", ".."], "signal_score": .., "criteria": [..] },
      "Prepare": { ... same shape ... },
      "Execute": { ... same shape ... }
    }
  },

  "opinion_cluster": {              one issue domain, several issue positions
    "key": "Issue_Position_Taxonomy > <issue domain>",
    "family": "Issue_Position_Taxonomy", "parent_name": "<issue domain>", "n_leaves": ..,
    "direction_summary": { "amplify_+1": .., "erode_-1": .. },
    "leaves": [ { "leaf": "..", "path": "Issue_Position_Taxonomy > <issue domain> > <position>",
                  "adversarial_direction": 1 or -1 }, ... ]
  }
}
```

What the attack leaf numbers mean: `leaf_id` is the stable id of a leaf in the
external DISARM-red ontology, and `path` is that same leaf spelled out from the
phase down to the technique. For example `"Plan": {"leaf_id": 549, "path": ["Plan",
"Plan Objectives", ... , "Label dispute pressure within objective planning"]}`. The
id lets you join back to the raw sample, the path lets you read it directly.

## How the set was built

The guiding rule is maximal entropy with preservation, so the joint set stays
balanced and introduces no range restriction or confounding. Each factor is
handled with the method that best fits its role.

1. **Profiles (each used exactly once).** There are 10,000 profiles and 10,000
   scenarios, so every profile is used one time. This is a perfect bijection. It
   is the maximal-entropy outcome over the profile set (normalised entropy 1.0)
   and it preserves every demographic margin that the profile sampler balanced.

2. **Opinion clusters (balanced and uniform).** The 10,000 scenarios are spread
   evenly across the 7 issue-domain clusters (each used 1,428 or 1,429 times).
   Uniform usage is the maximal-entropy allocation at the cluster layer
   (normalised entropy 1.0), and it guarantees that all 7 issue domains and all
   106 adversarially directional issue positions appear in the set.

3. **Attack triplets (entropy and coverage preserving subsample).** There are
   48,991 filtered attack triplets and we need 10,000. A multi-hop stochastic
   swap search selects the 10,000 so that the per-phase tactic mix, the inclusion
   route mix, and the signal distribution match the full filtered set (total
   variation close to 0), while maximising how many distinct attack leaves are
   covered. We preserve the source distribution rather than forcing every margin
   to be flat, because forcing uniformity on a naturally skewed factor (for
   example the inclusion route) would distort the sample and harm validity.

4. **Independent pairing.** The three factor lists are shuffled independently and
   then zipped together, so the profile, the attack, and the opinion cluster of a
   scenario are statistically independent of one another. This protects internal
   validity.

The build is fully reproducible with seed 42. The script is
`../../utils/build_integrated_scenarios.py`.

## Entropy and validity guarantees

Normalised entropy is the Shannon entropy divided by the maximum possible for the
number of categories, so 1.0 means perfectly uniform.

| Factor | Normalised entropy in the 10K set |
|--------|-----------------------------------|
| Profile usage (each used once) | 1.000 |
| Opinion cluster (7 issue domains) | 1.000 |
| Attack Plan tactic | 0.988 |
| Attack Prepare tactic | 0.998 |
| Attack Execute tactic | 0.997 |
| Attack signal decile | 0.999 |

The attack marginals match the full filtered set almost exactly (total variation
distance of about 0 to 0.0001 on every stratum), so the subsample is
representative and preserves the source entropy rather than reshaping it.

**Independence (no confounding).** Pairwise Cramer's V between the scenario
factors is close to 0 on every off-diagonal pair, for example:

* profile region by opinion cluster: 0.000
* profile age band by attack Execute tactic: 0.000
* opinion cluster by attack inclusion route: 0.003

Values this small mean the factors are not entangled, so any later effect can be
attributed cleanly to the factor that drives it.

## Coverage relative to the state space

| Quantity | Covered by the 10K set |
|----------|------------------------|
| Profiles | 10,000 of 10,000 |
| Opinion issue domains (clusters) | 7 of 7 |
| Adversarially directional issue positions | 106 of 106 |
| Distinct attack leaves | 13,626 of 17,941 filtered (about 76 percent) |

## A note on opinion direction (and neutral leaves)

Each opinion leaf carries a baked adversarial direction:

* `+1` means the adversary wants to amplify or increase the construct,
* `-1` means the adversary wants to erode or decrease it,
* `0` means the leaf is not a general cognitive-warfare target.

The opinion sampling frame is built only from directional leaves (the 106
issue-position leaves of the Issue Position Taxonomy with direction `+1` or `-1`,
out of 261 total). Neutral leaves (`0`) are kept in the ontology for completeness,
but they are never an attack target, so they are excluded from the frame and never
appear inside a scenario cluster. The directional balance is heavily toward
erosion (about 780 erode versus 220 amplify draws in the panel), because the
adversary mostly tries to weaken support for defense, democratic resilience,
infrastructure and alliances. Two of the 7 issue domains mix amplify and erode
positions (Foreign Policy and Geopolitics, Information Integrity and Platforms).

## Three representative scenarios

The wording below reads each scenario as: who is targeted, which issue positions
are in view, and what the attack does to them.

### Scenario 02676: eroding support for national defense

* **Profile.** A 72 year old (male at birth, nonbinary) from Asia with no formal
  education. Very low openness (4.5) and conscientiousness (16.5), high
  extraversion (84) and neuroticism (72).
* **Issue positions in view.** The issue domain "Defense and National Security"
  (21 directional positions), all encoded `-1`: support for alliance commitments,
  allied cyber mutual assistance, anti-submarine warfare investment, conscription,
  counter hybrid warfare capability, cyber defense investment, defense industrial
  reshoring, procurement transparency, and more. The adversary goal is to erode
  support for national defense and alliances.
* **Attack on those positions** (signal 11.6). A platform labeling dispute
  objective (Plan), microtargeting through a persistent synthetic persona
  (Prepare), and priming through a legal filing evidence narrative (Execute).

### Scenario 04697: a second attack on the same defense consensus

* **Profile.** A 61 year old (intersex, trans woman) from South America with upper
  secondary education. Very high extraversion (93), very low agreeableness (6.7),
  low neuroticism (16).
* **Issue positions in view.** The same "Defense and National Security" domain (21
  positions, all `-1`), shown here against a very different person and attack. The
  goal is again to erode defense and alliance support.
* **Attack on those positions** (signal 7.8). Segmenting by a Manichean good versus
  evil worldview (Plan), AI generated text content (Prepare), and prepositioning a
  screenshot archive for later priming (Execute).

### Scenario 06536: shifting a young person toward accommodationist geopolitics

* **Profile.** A 17 year old (male, cis man) from Asia with primary education. High
  openness (76), conscientiousness (84) and agreeableness (90), very low
  neuroticism (9).
* **Issue positions in view.** The issue domain "Foreign Policy and Geopolitics"
  (22 positions, mixed direction). The adversary erodes (`-1`) support for
  democracy promotion, diplomacy first, export controls, multilateralism and
  sanctions, while amplifying (`+1`) acceptance of energy dependence on
  authoritarian suppliers, great power appeasement and spheres of influence. The
  combined goal is an accommodationist, authoritarian friendly stance.
* **Attack on those positions** (signal 6.5). A review score pressure objective
  (Plan), a citation source narrative (Prepare), and priming through an astroturfed
  think tank expert (Execute).

## Provenance and joining back

Each row is self-contained, so joining back is optional. If you do want the
original source records, the stable identifiers are still present:

* `profile.profile_id` matches the profile sample in `../01_separated/profiles/`.
* `attack.config_id` and the three `attack.triplet.*.leaf_id` match the attack
  `leaf_catalog` in
  `../01_separated/attacks/red_plan_prepare_execute_opinion_effect_filtered.json`.
* `opinion_cluster.key` matches the opinion clusters in
  `../01_separated/opinions/opinion_targets_maxent_1000.json`.

The attack sample comes from an external DISARM-red attack ontology, not from this
repository's attack ontology. The raw leaf ids are stable, so the joins above are
exact.
