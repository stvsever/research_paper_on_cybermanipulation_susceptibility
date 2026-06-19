# Test Run 2 — Individual + empirical exposure-network layer (production set)

Test run 2 was executed on the pre-built integrated production scenario set: 100 scenarios sampled from the 10,000-row integrated set, where each scenario pairs one full high-resolution profile, one real DISARM-red Plan/Prepare/Execute attack triplet, and one opinion issue-domain cluster. It additionally runs the empirical exposure-network layer (profile-to-PolitiSky24-position assignment, then network-exposed baseline and post-attack re-elicitation through the directed exposure graph).

| Component | Value |
|-----------|-------|
| Output path | `evaluation/tests/run_2` |
| Scenarios | 100 (sampled from the 10,000-row integrated production set) |
| Ontology source | `src/backend/ontology/separate/production` |
| Simulation model | `deepseek/deepseek-v4-flash` through OpenRouter |
| Layers | individual + empirical exposure-network |
| Dashboard | `visuals/dashboard_results.html` |
| Network analysis | `network_exposure_analysis/` |

Reproduce:

```bash
bash scripts/tests/run_2.sh
```

Run 2 is the first run of the empirical exposure-network layer. Run 3 re-runs the same production design on the current cluster pipeline, with the exposure-network layer re-implemented onto the cluster-batched, DISARM-triplet logic; see `evaluation/tests/run_3/README.md`.
