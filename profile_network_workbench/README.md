# Profile Network Experiment Workbench

Clean local workbench for inspecting the profile-affinity network behind the cognitive sovereignty simulation and running baseline plus post-exposure opinion elicitation across the profile panel.

The app is intentionally narrow:

- nodes are deterministic pipeline profiles
- edges are profile affinity, not real social ties
- baseline and post-exposure results are node overlays, not inputs to the network
- post-exposure uses deterministic ontology-derived attack vector specs, matching Stage 03/04 semantics
- no multi-agent propagation or multi-stage contagion simulation logic is included

## API

```bash
cd 02_LOCAL_LAB/profile_network_workbench/api
uv run uvicorn profile_network_workbench_api.main:app --reload --port 8013
```

## Web

```bash
cd 02_LOCAL_LAB/profile_network_workbench/web
npm install
npm run dev -- --host 127.0.0.1 --port 5176
```

The web app expects `VITE_API_BASE_URL=http://127.0.0.1:8013`.

## Checks

```bash
cd 02_LOCAL_LAB/profile_network_workbench/api
uv run pytest

cd ../web
npm run typecheck
npm run build
```
