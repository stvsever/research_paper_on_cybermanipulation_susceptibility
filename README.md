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
- [🗂️ Repository Structure](#repository-structure)
- [🔄 Pipeline](#pipeline)
- [🔬 Test Runs](#test-runs)
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
## 🔬 Test Runs

The evaluation record spans three test runs. Each has its own README with the full configuration, methodology and headline results.

| Run | Design | Layers | Output |
|-----|--------|--------|--------|
| **Run 1** | 60 pseudoprofiles, 4 attack vectors, 3 opinions (crossed factorial, test ontology) | individual | [`evaluation/tests/run_1`](evaluation/tests/run_1) · [README](evaluation/tests/run_1/README.md) |
| **Run 2** | 100 scenarios from the 10,000-row integrated production set (full profiles, DISARM Plan/Prepare/Execute triplets, opinion clusters) | individual + exposure-network | [`evaluation/tests/run_2`](evaluation/tests/run_2) · [README](evaluation/tests/run_2/README.md) |
| **Run 3** | the run-2 production design on the current cluster pipeline + the integrated empirical exposure-network layer | individual + exposure-network (cluster) | [`evaluation/tests/run_3`](evaluation/tests/run_3) · [README](evaluation/tests/run_3/README.md) |

**Run 3 is the current integrated reference.** It produces a four-state, cluster-batched measurement backbone per opinion-cluster leaf: private baseline (B), network-exposure baseline (BN), private post-attack (P), and network-exposure post-attack (PN), at roughly 400 LLM calls for 100 scenarios. Reproduce it with:

```bash
bash scripts/tests/run_3.sh --network        # individual + exposure-network layer
bash scripts/tests/run_3.sh                   # individual layer only (~200 calls)
bash scripts/tests/run_3.sh --network --verbose   # add a live progress monitor
```

The interactive dashboard is at [`evaluation/tests/run_3/visuals/dashboard_results.html`](evaluation/tests/run_3/visuals/dashboard_results.html) and the comprehensive exposure-network report at [`evaluation/tests/run_3/visuals/network_exposure_analysis/reports`](evaluation/tests/run_3/visuals/network_exposure_analysis/reports). The launcher checks for `OPENROUTER_API_KEY` and verifies the projected OpenRouter budget before running.

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
|   `-- tests/                     (run_1.sh, run_2.sh, run_3.sh)
|-- evaluation/
|   `-- tests/
|       |-- run_1/                 (individual layer; see run_1/README.md)
|       |-- run_2/                 (individual + exposure-network; see run_2/README.md)
|       `-- run_3/                 (current integrated reference; see run_3/README.md)
|           |-- config/            (run configuration)
|           |-- logs/              (per-stage logs)
|           |-- provenance/        (raw LLM calls + run manifest)
|           |-- stage_outputs/     (canonical per-stage data for post-hoc analysis, all B/BN/P/PN phases)
|           |-- analysis/          (datasets, SEM, moderation report)
|           |-- visuals/           (dashboard, figures, embeddings, network_exposure_analysis)
|           |-- publication/       (publication assets + paper)
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
