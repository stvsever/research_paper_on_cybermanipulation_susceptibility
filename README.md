<div align="center">

# Inter-individual Differences in Susceptibility to Cyber-manipulation of Political Opinions

### An Ontology-Based Multi-Agent Simulation Approach

[![License: MIT](https://img.shields.io/badge/License-MIT-2a9d8f.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-e9c46a.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](docker/)

**Stijn Van Severen<sup>1,*</sup> · Thomas De Schryver<sup>1</sup> · Mira Ostyn<sup>1</sup>**

<sup>1</sup> Ghent University · <sup>*</sup> Corresponding author

---

</div>

## 📋 Table of Contents

- [🧬 Abstract](#abstract)
- [🔄 Pipeline](#pipeline)
- [🔬 Pipeline Runs](#pipeline-runs)
- [🗂️ Repository Structure](#repository-structure)
- [⚙️ Setup](#setup)
- [🚀 Manual Run](#manual-run)
- [📚 Citation](#citation)
- [⚖️ License](#license)

---

<a id="abstract"></a>
## 🧬 Abstract

This repository contains the backend research pipeline and first testing-run artifacts for a study on how **inter-individual differences moderate susceptibility to cyber-manipulation of political opinions**.

The workflow represents `PROFILE`, `ATTACK`, and `OPINION` as explicit hierarchical ontologies, generates ontology-based profile by attack by opinion scenarios, elicits baseline and post-exposure opinions with structured LLM agents, audits response coherence, computes directional adversarial effectivity, and estimates moderation with scenario-level machine-learning and statistical diagnostics.

---

<a id="pipeline"></a>
## 🔄 Pipeline

The full workflow runs from ontology-based scenario construction through agentic measurement, directional effect construction, and inferential analysis.

<div align="center">
<img src="src/backend/pipeline/full/pipeline_visualization.png" width="1200" alt="Pipeline overview for ontology-based adversarial opinion susceptibility auditing.">
</div>

**I. Ontological State Spaces** — three independent input taxonomies define the admissible state space: `PROFILE` (individual-level attributes), `ATTACK` (manipulative intervention types), and `OPINION` (target belief dimensions, each leaf carrying an adversarial direction: `+1` toward the goal, `-1` away, `0` excluded).

**II. Scenario Construction** — a crossed scenario generator takes the cross product of admissible leaves, so each observation is a unique tuple `(profile i, attack j, opinion k)`.

**III. Agentic Measurement Pipeline** — per scenario: (1) elicit a baseline opinion score conditioned on the profile, (2) generate an attack artifact instantiating the attack in realistic format, (3) audit and repair the artifact's realism until acceptable, (4) re-elicit the same profile's opinion after exposure, and (5) check response coherence, bounds, and consistency with the reasoning trace, flagging or re-running on failure.

**IV. Effect Construction** — compute the signed, direction-aware effectivity `AE_ijk = (P_ijk − B_ijk) × d_k`, where `d_k ∈ {+1, −1, 0}` is the adversarial direction at opinion leaf `k`. `AE > 0` is movement toward the attacker's goal (success), `AE < 0` is resistance/backfire, `AE = 0` is excluded. This yields a structured repeated-outcome dataset (observations nested within profiles).

**V. Inferential Layer** — a multi-stage analysis over the repeated-outcome dataset: (1) repeated-outcome moderation models (path/SEM, mixed-effects, multilevel variance decomposition) testing which profile characteristics explain directional effectivity; (2) task-conditional regularized models (ridge / elastic net / LASSO per attack×opinion task) aggregated into a composite susceptibility index `CSI_i`; (3) uncertainty and rank stability via profile-cluster bootstrap (BCa intervals) and Bayesian rank stability; (4) a profile-feature dependency graph (signed correlation network, centrality, community detection, bridge variables).

**VI. Outputs & Interpretation** — identify which profile characteristics increase or decrease susceptibility and toward which opinions, provide profile-level susceptibility estimates with uncertainty (`CSI` scores and intervals), surface feature interdependencies and key bridge variables, and enable generalizable insight into adversarial persuasion under structured heterogeneity.

Cross-cutting **safeguards and traceability** run across all stages: ontology constraints define admissible states, realism audits constrain artifact validity, coherence checks ensure plausible responses, and full provenance logging is kept end-to-end.

> **Note (legacy description).** The figure and stages I to VI above describe the original crossed-factorial design used in run 1 (a dense `profile x attack x opinion` cross product). The current integrated production pipeline (runs 2 and 3) instead samples scenarios from a pre-built 10,000-row integrated set, where each scenario pairs one full high-resolution profile, one near-unique DISARM Plan/Prepare/Execute attack triplet, and one opinion parent cluster scored cluster-at-once. Run 3 additionally runs the empirical exposure-network layer (stages 01b, 02b, 04b) for the four-state B/BN/P/PN backbone. `#TODO: update this pipeline section and figure to the integrated cluster + exposure-network design.`

---

<a id="test-runs"></a>
<a id="pipeline-runs"></a>
## 🔬 Pipeline Runs

The evaluation record has two tiers: small **test runs** that validate methodology
end to end, and the full-scale **production run**. Each has its own README with the
complete configuration and headline results.

### Test Runs

Five test runs validate the pipeline and the exposure-network layer at small scale.

| Run | Design | Layers | Output |
|-----|--------|--------|--------|
| **Run 1** | 60 pseudoprofiles, 4 attack vectors, 3 opinions (crossed factorial, test ontology) | individual | [`evaluation/tests/run_1`](evaluation/tests/run_1) · [README](evaluation/tests/run_1/README.md) |
| **Run 2** | 100 scenarios from the 10,000-row integrated production set (full profiles, DISARM Plan/Prepare/Execute triplets, opinion clusters) | individual + exposure-network | [`evaluation/tests/run_2`](evaluation/tests/run_2) · [README](evaluation/tests/run_2/README.md) |
| **Run 3** | the run-2 production design on the current cluster pipeline + the integrated empirical exposure-network layer | individual + exposure-network (cluster) | [`evaluation/tests/run_3`](evaluation/tests/run_3) · [README](evaluation/tests/run_3/README.md) |
| **Run 4** | 200 scenarios concentrated into 2 issue domains so the empirical exposure network is dense, on a recalibrated exposure-network layer | individual + exposure-network (working) | [`evaluation/tests/run_4`](evaluation/tests/run_4) · [README](evaluation/tests/run_4/README.md) |
| **Run 5** | 60 scenarios in a single issue domain with a reduced (about 40 percent smaller) profile and per-attack-tactic conditional estimation, so the individual-layer moderators are interpretable | individual + exposure-network | [`evaluation/tests/run_5`](evaluation/tests/run_5) · [README](evaluation/tests/run_5/README.md) |

**Run 5 is the test reference.** It keeps the run-4 four-state backbone (B, BN, P, PN) and working exposure-network methodology and fixes the individual layer. The pre-built profiles carry ~526 traits across many overlapping taxonomies, which left the moderator models under-determined (run 4 had near-zero CV-R2 and no significant moderators). Stage 01 drops a curated portion of the profile (the redundant HEXACO/Eysenck/Hexad personality taxonomies and the low-relevance goals/values/safety/criminal/administrative subtrees) and keeps the research core; the filter applies in one place so the dropped traits leave both the agent prompt and every analysis. With the reduced profile the moderation is interpretable (the political-psychology block carries about 60 percent of the explained moderation; openness is a significant moderator, b = +2.78, p = 0.030), and the conditional susceptibility estimator carries per-DISARM-Execute-tactic tasks so a specific attack vector can be selected in the dashboard. Reproduce with `bash scripts/tests/run_5.sh` (add `--no-network` for the individual layer only, or `--verbose` for a live monitor).

### Production Runs

| Run | Design | Layers | Output |
|-----|--------|--------|--------|
| **Run 1** | all 10,000 integrated scenarios (stratified across the 7 issue domains, 151,448 opinion-leaf measurements), individual layer only. A 159-feature research-core profile: the full hierarchical Big Five, the core demographic markers, and the political-psychology / ideology / moral-foundations battery | individual only | [`evaluation/production/run_1`](evaluation/production/run_1) · [README](evaluation/production/run_1/README.md) |

**Production run 1 is the full-scale individual-layer measurement.** It runs stages 01 to 05 over the entire integrated set (about 20,000 LLM calls on `deepseek/deepseek-v4-flash`), with the empirical exposure-network layer off. The profile is reduced a second time, from 336 to about 159 features (a 53 percent further reduction, 71 percent versus the full 540-feature integrated profile), keeping the full hierarchical Big Five, the core demographic markers and the political-psychology / ideology / moral-foundations battery, and dropping the political-participation, socioeconomic and life-circumstance taxonomies and the over-detailed identity spectra. Storage is lean: stage 05 keeps only the compact CSV delta tables (every B, P, delta and effectivity score per scenario and leaf); the full source content of any scenario is recoverable by joining the integrated set on `scenario_id`.

The headline results (see the [run README](evaluation/production/run_1/README.md) and `evaluation/production/run_1/visuals/`): the attack reliably moves private opinions (88 percent of leaves, Cohen d_z = 1.23); what is attacked dominates who is attacked, with the issue domain explaining roughly 30 times more between-scenario variance than the entire 159-trait profile battery and the specific DISARM tactic barely mattering; and susceptibility is highly heterogeneous between individuals (ICC = 0.83) but only weakly along measured trait axes, with the Big Five the leading moderator family (openness more movable, conscientiousness more resistant, neuroticism more movable, all FDR-significant). The single-leaf macroeconomic domain is excluded from the analyses as a statistical outlier. Reproduce with:

```bash
bash scripts/production/run_1.sh --verbose            # the 10,000-scenario run (stages 01..05)
.venv/bin/python src/backend/utils/analysis/analyze_run_1.py  # family-wise moderation, inferential tests, figures
```

The launcher checks for `OPENROUTER_API_KEY` and verifies the projected OpenRouter budget before running.

---

<a id="repository-structure"></a>
## 🗂️ Repository Structure

```text
research_paper_on_cybermanipulation_susceptibility/
|-- README.md
|-- LICENSE
|-- CITATION.cff
|-- requirements.txt
|-- .env.example
|-- .gitignore
|-- docker/
|-- scripts/
|   |-- tests/                     (run_1.sh, run_2.sh, run_3.sh, run_4.sh, run_5.sh)
|   `-- production/                (run_1.sh: full 10,000-scenario individual-layer run)
|-- evaluation/
|   |-- tests/
|   |   |-- run_1/                 (individual layer; see run_1/README.md)
|   |   |-- run_2/                 (individual + exposure-network; see run_2/README.md)
|   |   |-- run_3/                 (prior integrated reference; see run_3/README.md)
|   |   |-- run_4/                 (dense 2-domain exposure-network reference; see run_4/README.md)
|   |   `-- run_5/                 (test reference, reduced profile; see run_5/README.md)
|   |       |-- config/            (run configuration)
|   |       |-- logs/              (per-stage logs)
|   |       |-- provenance/        (raw LLM calls + run manifest)
|   |       |-- stage_outputs/     (canonical per-stage data for post-hoc analysis, all B/BN/P/PN phases)
|   |       |-- analysis/          (datasets, SEM, moderation report)
|   |       |-- visuals/           (dashboard, figures, embeddings, network_exposure_analysis)
|   |       |-- publication/       (publication assets + paper)
|   |       `-- README.md
|   `-- production/
|       `-- run_1/                 (full 10,000-scenario individual-layer run; see run_1/README.md)
|           |-- config/            (run configuration)
|           |-- logs/              (console log)
|           |-- stage_outputs/     (stages 01..05: scenarios, baseline B, attack spec, post-attack P, effectivity deltas)
|           `-- README.md
`-- src/
    |-- data/                      (empirical exposure-network substrate)
    `-- backend/
        |-- agentic_framework/     (agents, factory, prompts 01-04 + opinion_coherence_review)
        |-- ontology/
        |-- pipeline/
        |   |-- separate/          (numbered stages 01..08 plus the network stages 01b/02b/04b/05b)
        |   `-- full/              (run_full_pipeline orchestrator)
        `-- utils/                 (core: io, schemas, ontology_utils, logging; grouped: figures/, reporting/, analysis/, scenario/, embeddings/, network_exposure/)
```

`research_report/`, local virtual environments, editor files, local frontends, and `.env` files are intentionally excluded from the repository.

---

<a id="setup"></a>
## ⚙️ Setup

```bash
git clone https://github.com/stvsever/research_paper_on_cybermanipulation_susceptibility.git
cd research_paper_on_cybermanipulation_susceptibility
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Add `OPENROUTER_API_KEY` to `.env` before running the pipeline.

---

<a id="manual-run"></a>
## 🚀 Manual Run

The launcher `scripts/tests/run_3.sh` is the recommended entry point. The equivalent direct invocation for the current integrated reference run (run 3, with the empirical exposure-network layer) is:

```bash
.venv/bin/python src/backend/pipeline/full/run_full_pipeline.py \
  --output-root evaluation/tests/run_3 \
  --run-id run_3 \
  --integrated-scenarios-path src/backend/pipeline/separate/01_create_scenarios/samples/02_integrated/integrated_scenarios_10000.jsonl \
  --n-scenarios 100 \
  --seed 120 \
  --attack-ratio 1.0 \
  --primary-moderator posthoc_profile_susceptibility_index \
  --bootstrap-samples 200 \
  --no-use-test-ontology \
  --ontology-root src/backend/ontology/separate/production \
  --no-enforce-compatibility-rules \
  --drop-direction-neutral-opinions \
  --openrouter-model deepseek/deepseek-v4-flash \
  --temperature 0.15 \
  --max-repair-iter 1 \
  --profile-generation-mode deterministic \
  --no-self-supervise-opinion-coherence \
  --no-self-supervise-attack-realism \
  --with-network-exposure \
  --exposure-network-root src/data/exposure_networks/politisky24_bluesky_v1 \
  --generate-visuals \
  --export-static-figures \
  --no-build-report \
  --resume-from-stage 01 \
  --stop-after-stage 08
```

Drop `--with-network-exposure` for the individual layer only. Run 1 and run 2 are reproduced with `bash scripts/tests/run_1.sh` and `bash scripts/tests/run_2.sh`.

---

<a id="citation"></a>
## 📚 Citation

### APA 7

> Van Severen, S., De Schryver, T., & Ostyn, M. (2026). *Inter-individual Differences in Susceptibility to Cyber-manipulation of Political Opinions: An Ontology-Based Multi-Agent Simulation Approach*. Ghent University. https://github.com/stvsever/research_paper_on_cybermanipulation_susceptibility

### BibTeX

```bibtex
@article{vanseveren2026cybermanipulationsusceptibility,
  title     = {Inter-individual Differences in Susceptibility to Cyber-manipulation of Political Opinions: An Ontology-Based Multi-Agent Simulation Approach},
  author    = {Van Severen, Stijn and De Schryver, Thomas and Ostyn, Mira},
  year      = {2026},
  institution = {Ghent University},
  url       = {https://github.com/stvsever/research_paper_on_cybermanipulation_susceptibility}
}
```

A machine-readable citation is also available in [`CITATION.cff`](CITATION.cff).

---

<a id="license"></a>
## ⚖️ License

This project is licensed under the **MIT License**; see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built at **Ghent University** for the course *Case Studies in the Analysis of Experimental Data*

</div>
