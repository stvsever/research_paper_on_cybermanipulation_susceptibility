# Test Run 1 — Individual layer (crossed factorial, test ontology)

Test run 1 is a focused factorial evaluation over 60 pseudoprofiles, 4 cyber-manipulation attack vectors, and 3 political opinion targets. It uses a focused subset of the testing ontology and exercises the individual layer only (no exposure-network layer).

| Component | Value |
|-----------|-------|
| Output path | `evaluation/tests/run_1` |
| Profiles | 60 maximal-entropy pseudoprofiles |
| Attack vectors | Headline_And_Lede_Misframing, Personal_Safety_Fear_Appeal, Petition_Astroturf, Multi_Turn_Counter_Argument_Adaptation |
| Opinion leaves | Alliance_Commitment_Support, Trust_In_Mainstream_Journalism, Defense_Spending_Increase_Support |
| Ontology source | `src/backend/ontology/separate/test` |
| Simulation model | `deepseek/deepseek-v4-flash` through OpenRouter |
| Stages | 01 through 08 |
| Dashboard | `visuals/dashboard_results.html` |

Reproduce:

```bash
bash scripts/tests/run_1.sh
```

The launcher checks for `OPENROUTER_API_KEY`, verifies the projected OpenRouter budget, and writes logs under the run's `logs/`. Open `visuals/dashboard_results.html` in a browser for the interactive overview.

Run 1 is the original crossed-factorial design (a dense `profile x attack x opinion` cross product). The later runs (2 and 3) move to the integrated production scenario set; see the root `README.md` and the run 2 / run 3 READMEs.
