# Future Directions

This document outlines the research and engineering roadmap for the cyber-manipulation susceptibility pipeline. Items are grouped by theme and ordered roughly by priority.

---

## 1. Ablation Studies

A rigorous ablation programme is needed to isolate which pipeline components contribute genuine measurement value.

### 1.1 Simulation vs. direct one-shot elicitation *(most critical)*

The central methodological claim is that the staged multi-step simulation produces better-calibrated susceptibility estimates than a scalar judgment. Without this ablation, the claim remains asserted rather than demonstrated.

**Protocol:**
1. Recruit 50–100 real participants with matched profile features
2. Collect: (a) baseline opinion, (b) adversarial exposure, (c) post-exposure opinion on ontology-aligned items
3. Compute observed *AE* as criterion
4. Evaluate four estimators in parallel: demographics-only regression, Big Five regression, direct one-shot LLM rating ("how susceptible is this profile?"), simulation-derived CSI
5. Compare by MAE and Spearman rank correlation against observed *AE*

### 1.2 Attack ontology ablation

Hold profiles and opinions constant; remove one attack family at a time (cognitive reframing / emotional manipulation / coordinated amplification / authority heuristics / AI-based). Measure:
- Change in inter-profile discriminability of the susceptibility index (Δ variance explained)
- Absolute shift variance by profile group
- Moderator effect size stability

### 1.3 Opinion-domain ablation

Replace opinion domains with alternatives (health misinformation, financial fraud, civic trust erosion). If moderator patterns (Extraversion, Conscientiousness) remain stable across domains, a domain-general interpretation is supported.

### 1.4 Profile-feature ablation

Systematically remove feature groups (demographics-only → Big Five-only → full 85-dim set) and measure test-set rank correlation of the CSI. This tests whether any feature combination produces reliable cross-profile discrimination at the current *N_p*.

### 1.5 LLM backend sensitivity

Run the identical N_p × N_a × N_o factorial across multiple LLM backends (Mistral, LLaMA-3, GPT-4o). High cross-model CSI rank-order consistency supports construct validity; high inconsistency indicates the estimates reflect idiosyncratic model priors.

---

## 2. Non-linear Susceptibility Estimation

The current conditional susceptibility index uses **ridge regression** — a linear model with L2 regularisation. This has advantages (interpretability, reliable convergence at small *N_p*) but misses interaction and threshold effects.

### 2.1 Gradient-boosted trees (XGBoost / LightGBM)

- Replace the per-task ridge model with a GBT model behind the same scoring API
- Enables automatic modelling of trait × attack × opinion interactions
- Requires larger *N_p* for stable feature importance; cross-validate carefully
- Integrate SHAP TreeExplainer for global and per-profile feature attribution

### 2.2 Neural susceptibility model

- Architecture: profile feature vector → 3-layer MLP → predicted *AE* per task
- Multi-task head: one output per (attack, opinion) task *t ∈ T*
- Shared representation layer forces generalisation across tasks
- Useful when *N_p* > 500; apply dropout + weight decay to prevent overfitting

### 2.3 Gaussian process regression

- Natural uncertainty quantification: posterior predictive intervals on CSI without bootstrap
- Kernel choice: Matérn on continuous traits + categorical kernel on Sex / Education
- Scales poorly above *N_p* ≈ 2,000 without sparse approximations

---

## 3. Explainability (XAI)

### 3.1 SHAP (SHapley Additive exPlanations)

- **Global explanations**: mean |SHAP| per feature group across all *N_p* profiles and all tasks
- **Per-profile explanations**: explain why profile *i* scored high/low on the CSI → actionable intervention targets
- **Interaction plots**: SHAP interaction values between pairs of traits (e.g. Conscientiousness × Neuroticism)
- Implementation: `shap.TreeExplainer` for GBT models; `shap.KernelExplainer` for the ridge baseline

### 3.2 LIME (Local Interpretable Model-agnostic Explanations)

- Complement SHAP for the ridge baseline where TreeExplainer does not apply
- Perturb profile features locally; fit linear approximation; rank local feature contributions
- Useful for operational use cases: "why did this specific profile score in the 90th percentile?"

### 3.3 Permutation importance with grouped features

- Extend the existing leave-one-group-out marginal R² decomposition to the non-linear models
- Permute entire inventory groups (all Big Five facets simultaneously) to measure group-level importance rather than individual feature importance

### 3.4 Counterfactual profiles

- For high-CSI profiles, compute the minimal feature perturbation that would move them below the 50th percentile
- Useful for resilience intervention design: "what would it take to make this profile resilient?"
- Implementation: `dice-ml` library for diverse counterfactual generation

---

## 4. Simulation Realism: Multi-stage Internal Decision-making

The current pipeline uses a single-step opinion elicitation: the agent reads the attack and immediately gives a post-exposure score. This is a simplification of how real persuasion operates.

### 4.1 Internal reasoning chain (chain-of-thought simulation)

Before emitting a post-exposure opinion score, the agent should simulate:
1. **Source evaluation**: credibility assessment of the attack source
2. **Content processing**: identification of rhetorical strategies in the attack
3. **Prior belief integration**: how does this new information relate to existing beliefs?
4. **Emotional response**: affect triggered by the attack
5. **Belief update**: revised opinion score with justification

This multi-stage deliberation produces richer intermediate variables (source credibility, perceived threat, emotional valence) that can themselves be studied as moderator mediators.

### 4.2 Social context simulation

Add a social feed context before attack exposure: the agent first reads a plausible social media feed (2–3 posts, mix of agreeing/disagreeing content), then encounters the attack. This models the real context in which disinformation operates.

### 4.3 Memory and belief persistence

After post-exposure opinion collection, re-elicit the opinion after a simulated "delay" (prompt the agent to answer as if 24 hours have passed). Measures belief persistence vs. rebound — a key distinction between attitude change and momentary persuasion.

### 4.4 Counter-inoculation simulation

Before attack exposure, expose a subset of profiles to a pre-bunking message (brief warning about the manipulation tactic to follow). Enables studying inoculation × profile interactions within the same factorial design.

---

## 5. Scale-up and Validation

| Goal | Current | Target |
|------|---------|--------|
| N_p (profiles) | 80 | 500–1,000 |
| N_a (attack leaves) | 6 | 12–20 |
| N_o (opinion leaves) | 8 | 16–32 |
| Opinion domains | 4 | 6–8 |
| Profile dimensions | 85 | 120+ (add HEXACO, Dark Triad) |
| LLM concurrency | 20 | 100+ |

### 5.1 Human ground-truth validation

Collect a matched human experimental dataset (Prolific or MTurk) using the same PROFILE × ATTACK × OPINION operationalisation. Compare simulation-derived CSI ranks against observed human susceptibility ranks. This is the primary external validity test.

### 5.2 Longitudinal opinion tracking

Run the simulation across multiple time points with the same pseudoprofile. Measure stability of CSI rank over time and across opinion domain rotations.

---

## 6. Production Deployment

### 6.1 Artifact contract stabilisation

Define a versioned JSON schema for the `conditional_susceptibility_artifact.json` file. Add backward-compatibility checks so older artifacts can be scored by newer model versions.

### 6.2 Calibration and ranking diagnostics

- Add isotonic regression calibration layer to map raw CSI scores to probability-calibrated susceptibility estimates
- Compute Brier score and rank correlation against held-out profiles at each retraining
- Alert when out-of-distribution profiles are submitted to the scoring API

### 6.3 Caching and resumability

- Cache LLM responses at the (profile_hash, attack_leaf, opinion_leaf) level
- Support warm-start from any stage: if stage *k* outputs exist, skip stages 1..k
- Add per-scenario retry budget with exponential backoff

### 6.4 Interactive frontend

Build a browser-based dashboard (Streamlit or Next.js) that allows:
- Uploading custom PROFILE × ATTACK × OPINION JSON ontologies
- Running a mini-simulation (N_p = 10–20) live in the browser via WebSocket
- Visualising CSI rankings and SHAP explanations interactively

---

## 7. Ethics and Dual-use Considerations

- All adversarial attack content is generated for defensive research purposes only
- Production deployments must gate access to the scoring API behind institutional review
- CSI scores should never be used to target real individuals — only to study population-level susceptibility patterns
- Maintain a transparency log of which ontologies and LLM backends were used in each run
- Consider adding a differential-privacy noise layer to the CSI scores before releasing group-level summaries

---

*Last updated: 2026-04-16 · See [README.md](../README.md) for the current study design.*
