# Integrated scenario set: overview figures and guide

This folder visualises the final integrated design, the 10,000 scenarios in
`../../samples/02_integrated/integrated_scenarios_10000.jsonl`. Every figure is
provided as a static `pngs/` image and an interactive `htmls/` version.

For the data schema, the build method and the join-back keys, see the data README
at `../../samples/02_integrated/README.md`. This document explains what the 10,000
scenarios are, how to read each figure, how to use the interactive explorer, and
it ends with three worked examples.

## What the 10,000 scenarios are

Each scenario combines one item from each of the three sub-ontologies:

```
one scenario = 1 profile configuration
             + 1 DISARM-red Plan / Prepare / Execute attack triplet
             + 1 opinion parent cluster (an issue domain, several issue positions)
```

Read as a question: for this kind of person, if an adversary runs this specific
attack, what happens to this related group of issue positions?

Two scoping choices shape this version:

* **Issue positions moved from the profile to the opinion side.** The profile
  ontology's `Issue_Position_Taxonomy` (223 fine issue-position items) was
  excluded from profile sampling, because issue positions are stances an
  adversary tries to shift, not stable background traits. They now define the
  opinion targets instead.
* **Opinions are sampled only from the opinion `Issue_Position_Taxonomy`
  subtree.** The opinion targets are the adversarially directional issue
  positions, grouped into 7 issue domains (Defense and National Security,
  Foreign Policy and Geopolitics, Democratic Resilience and Institutions,
  Information Integrity and Platforms, Critical Infrastructure and Energy
  Sovereignty, Macroeconomic and Fiscal Policy, Supranational and Regional
  Integration).

The set keeps each design factor at or near maximal entropy while preserving the
source distributions, and the three factors stay statistically independent:

* every one of the 10,000 profiles is used exactly once (uniform, entropy 1.00),
* the 7 issue-domain clusters are used evenly, 1,428 or 1,429 times each
  (uniform, entropy 1.00), so all 106 directional issue-position leaves appear,
* the 10,000 attack triplets are an entropy and coverage preserving subsample of
  the 48,991 filtered triplets,
* the factors are paired by independent shuffles, so pairwise Cramer's V between
  profile, attack and opinion features is close to 0.

The attack triplets come from an external DISARM-red attack ontology, not from
this repository's attack ontology, so the explorer uses the common root path
structure of the sampled triplets (phase, then tactic).

## The figures in this folder

| Figure | Type | What it shows |
|--------|------|----------------|
| `integrated_factor_explorer_3d` | interactive (flagship) | A 3D scatter of all 10,000 scenarios where each X, Y, Z axis is chosen from the real ontology hierarchy. Colour = opinion issue domain. |
| `integrated_composition` | descriptive | Marginals of the set: opinion issue domain, profile age, attack Execute tactic, and the attack opinion-signal density. |
| `integrated_entropy_retention` | methodological | Left: normalised entropy of each scenario factor against its source or theoretical maximum. Right: coverage relative to each ontology. |
| `integrated_independence_matrix` | methodological | Pairwise Cramer's V between the design factors. Off-diagonal near 0 means no confounding. |
| `integrated_flow_sankey` | descriptive | Flow from profile world region to opinion issue domain to dominant attack phase. |

## How to use the interactive explorer

Open `htmls/integrated_factor_explorer_3d.html` in a browser (Plotly is bundled,
so it works offline). There are three rows of controls, one per axis (X, Y, Z).
Each row is a three-level cascade that drills the real ontology:

```
ontology · domain   →   construct   →   factor
```

For example, for an axis you might pick
`Profile · Demographics and Identity` → `Political Profile › Ideological Dimensions
Two Axis Model › Economic Left Right` → `Redistribution and Inequality`, or
`Attack · DISARM-red` → `Tactics & route` → `Execute tactic`, or
`Opinion · Issue Position Taxonomy` → `Adversarial direction` → `Net direction`.

The profile branch is the real `profile.json` hierarchy (read from the variable
plan, so every factor keeps its true ontology path), the attack branch is the
DISARM-red sample structure, and the opinion branch is the Issue Position Taxonomy
structure. There are 479 factors across 20 ontology and domain groups.

Colour is fixed on the opinion issue domain (the 7 clusters). The legend is
click-toggleable, so you can isolate or hide any issue domain.

Each factor is drawn in the way that suits its measurement level: continuous and
interval factors (age, the Big Five, the numeric psychometric and political
scales, the attack signal) use a linear axis, ordinal factors (highest education)
use the construct order, and categorical factors (region, gender, attack tactic,
issue domain) are placed at evenly spaced positions with the real category labels
on the axis and a small jitter so the cloud is visible.

What "Attack signal" means. The factor `Opinion-manipulation signal (total)` is
the heuristic strength, on a scale of roughly 5 to 26, with which an attack
triplet relates to opinion change. It is preserved from the filtered attack pool,
not maximised.

## Three worked examples

Each example reads as: who is targeted, which issue positions are in view, and
what the attack does to them. The full versions are in
`../../samples/02_integrated/integrated_scenarios_examples_3.json`.

### Scenario 02676: eroding support for national defense

* **Profile.** A 72 year old (male at birth, nonbinary) from Asia with no formal
  education. Personality: very low openness (4.5) and conscientiousness (16.5),
  high extraversion (84) and neuroticism (72). A sociable, anxious, change-averse
  older person.
* **Issue positions in view.** The issue domain `Defense and National Security`
  (21 directional issue positions), all marked `-1`: support for alliance
  commitments, allied cyber mutual assistance, anti-submarine warfare investment,
  conscription, counter hybrid warfare capability, cyber defense investment,
  defense industrial reshoring, procurement transparency, and more. The
  adversary's goal is to erode support for national defense and alliances.
* **The attack.** Signal 11.6. In Plan a platform labeling dispute objective, in
  Prepare microtargeting through a persistent synthetic persona, in Execute
  priming with a legal filing evidence narrative. Visibility pressure plus
  persona based personalisation plus a credibility and evidence frame.

### Scenario 04697: a second attack on the same defense consensus

* **Profile.** A 61 year old (intersex, trans woman) from South America with upper
  secondary education. Personality: very high extraversion (93), very low
  agreeableness (6.7), low neuroticism (16). A highly outgoing, contrarian, calm
  person.
* **Issue positions in view.** The same `Defense and National Security` domain
  (21 issue positions, all `-1`). The goal is again to erode defense and alliance
  support, shown here against a very different person and attack.
* **The attack.** Signal 7.8. In Plan it segments by a Manichean good versus evil
  worldview, in Prepare it develops AI generated text, in Execute it prepositions
  a screenshot archive for later priming. Susceptibility segmentation plus
  automated content plus pre staged evidence.

### Scenario 06536: shifting a young person toward accommodationist geopolitics

* **Profile.** A 17 year old (male, cis man) from Asia with primary education.
  Personality: high openness (76), conscientiousness (84), agreeableness (90) and
  very low neuroticism (9). A young, open, disciplined, agreeable, emotionally
  stable person.
* **Issue positions in view.** The issue domain `Foreign Policy and Geopolitics`
  (22 directional issue positions, mixed direction). The adversary erodes (`-1`)
  support for democracy promotion, diplomacy first, export controls on sensitive
  technology, multilateralism and sanctions, while amplifying (`+1`) acceptance of
  energy dependence on authoritarian suppliers, great power appeasement and
  spheres of influence. The combined goal is to move the person toward an
  accommodationist, authoritarian friendly foreign policy stance.
* **The attack.** Signal 6.5. In Plan a review score pressure objective, in
  Prepare a citation source narrative, in Execute priming through an astroturfed
  think tank expert. Reputation pressure plus source framing plus fake expertise.

These three span a wide range: a 72 year old and a 17 year old, three world
regions, no formal through upper secondary education, pure erosion and mixed
direction goals, and strong through moderate attack signals. The first two show
two different attacks and people aimed at the same defense consensus, the third
shows a mixed amplify and erode goal on foreign policy.
