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
- [🔬 Test Run 1](#test-run-1)
- [🔄 Pipeline](#pipeline)
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

The current public evaluation record is **test run 1**, located at `evaluation/tests/run_1`. It uses a focused subset of the testing ontology.

---

<a id="test-run-1"></a>
## 🔬 Test Run 1

Test run 1 is a focused factorial evaluation over 60 pseudoprofiles, 4 cyber-manipulation attack vectors, and 3 political opinion targets.

| Component | Value |
|-----------|-------|
| Output path | `evaluation/tests/run_1` |
| Profiles | 60 maximal-entropy pseudoprofiles |
| Attack vectors | Headline_And_Lede_Misframing, Personal_Safety_Fear_Appeal, Petition_Astroturf, Multi_Turn_Counter_Argument_Adaptation |
| Opinion leaves | Alliance_Commitment_Support, Trust_In_Mainstream_Journalism, Defense_Spending_Increase_Support |
| Ontology source | `src/backend/ontology/separate/test` |
| Simulation model | `deepseek/deepseek-v4-flash` through OpenRouter |
| Stages | 01 through 08 |
| Dashboard | `evaluation/tests/run_1/visuals/dashboard_results.html` |

Reproduce the testing run with:

```bash
bash scripts/tests/run_1.sh
```

The launcher checks for `OPENROUTER_API_KEY`, verifies the projected OpenRouter budget, and writes logs under `evaluation/tests/logs`.

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
|   `-- tests/
|       `-- run_1.sh
|-- evaluation/
|   |-- production/
|   `-- tests/
|       `-- run_1/
|           |-- config/
|           |-- datasets/
|           |-- embeddings/
|           |-- embeddings_production/
|           |-- publication_assets/
|           |-- sem/
|           |-- stage_outputs/
|           `-- visuals/
`-- src/
    `-- backend/
        |-- agentic_framework/
        |-- ontology/
        |-- pipeline/
        `-- utils/
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

```bash
.venv/bin/python src/backend/pipeline/full/run_full_pipeline.py \
  --output-root evaluation/tests/run_1 \
  --run-id run_1 \
  --paper-title "Inter-individual Differences in Susceptibility to Cyber-manipulation of Political Opinions: An Ontology-Based Multi-Agent Simulation Approach" \
  --n-profiles 60 \
  --seed 120 \
  --attack-ratio 1.0 \
  --attack-leaves "Headline_And_Lede_Misframing,Personal_Safety_Fear_Appeal,Petition_Astroturf,Multi_Turn_Counter_Argument_Adaptation" \
  --opinion-leaves "Alliance_Commitment_Support,Trust_In_Mainstream_Journalism,Defense_Spending_Increase_Support" \
  --profile-candidate-multiplier 8 \
  --primary-moderator posthoc_profile_susceptibility_index \
  --bootstrap-samples 600 \
  --use-test-ontology \
  --ontology-root src/backend/ontology/separate/test \
  --enforce-compatibility-rules \
  --drop-direction-neutral-opinions \
  --openrouter-model deepseek/deepseek-v4-flash \
  --temperature 0.15 \
  --max-repair-iter 2 \
  --profile-generation-mode deterministic \
  --self-supervise-opinion-coherence \
  --coherence-threshold 0.74 \
  --generate-visuals \
  --export-static-figures \
  --no-build-report \
  --resume-from-stage 01 \
  --stop-after-stage 08
```

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
