# Network Exposure Layer

## Goal

The network exposure layer estimates whether empirically grounded directed exposure position amplifies or attenuates cyber-manipulation effects across synthetic susceptibility profiles.

The core idea is not that profiles are real social ties or platform accounts. Each generated profile is assigned to one observed position in an empirical Bluesky engagement-derived exposure graph. That position determines which other profiles' outputs are visible as peer context in the network-exposure measurement phases.

The scientific object is:

```text
visible peer profile j -> exposed target profile i
```

In the current implementation, peer context is no longer selected by profile similarity. It is selected through the assigned empirical exposure edge from peer position to target position.

## Measurement Backbone

| Symbol | Measurement | Pipeline field | Meaning |
|---|---|---|---|
| `B` | Private baseline opinion | `baseline_assessment.score` | The target profile's private pre-attack score for one opinion leaf. |
| `BN` | Baseline network-exposure opinion | `network_exposure_assessment.score` | The target profile's pre-attack score after seeing empirical incoming peers' baseline evaluations for the same opinion leaf. |
| `P` | Private post-attack opinion | `post_attack_assessment.score` | The target profile's private post-attack score after the attack vector, before peer context. |
| `PN` | Post-attack network-exposure opinion | `post_attack_network_exposure_assessment.score` | The target profile's post-attack score after seeing empirical incoming peers' post-attack evaluations under the same opinion and attack/control condition. |

`d` denotes adversarial direction: `+1` means the attacker wanted a higher score, `-1` means the attacker wanted a lower score, and `0` means no directional effect is defined.

## Effect Deltas And Questions

| Quantity | Formula | Question answered |
|---|---:|---|
| Private attack effect | `P - B` | How much did the private attack exposure move this profile before network context? |
| Direction-aware private attack effect | `AE_private = (P - B) * d` | Did the private attack move the profile toward the adversarial goal? |
| Baseline network increment | `BN_increment = BN - B` | Does empirical baseline peer context shift the profile's pre-attack opinion? |
| Post-attack network increment | `PN_increment = PN - P` | Does same-condition peer context shift the profile after private attack exposure? |
| Direction-aware post-network increment | `PN_increment_effectivity = (PN - P) * d` | Does peer context amplify or attenuate the private attack effect in the adversarial direction? |
| Total network-exposed attack effect | `AE_total_network = (PN - B) * d` | What is the final direction-aware attack effect after both private attack exposure and post-attack peer context? |
| Centrality-weighted private activation | `sum(outgoing_reach_j * AE_private_j)` | Are highly visible susceptible senders structurally positioned to amplify population-level effects? |

## Research Question

How do inter-individual susceptibility profiles interact with empirically grounded exposure-network position to amplify or attenuate cyber-manipulation effects across a population?

Operationally: conditional on attack vector and opinion target, does the final network-exposed attack effect depend on the interaction between a profile's private attack susceptibility and its position in a realistic directed exposure network?

## Hypotheses

| Hypothesis | Claim | Operational test |
|---|---|---|
| H1. Private susceptibility heterogeneity | Profiles differ in private attack susceptibility. | `AE_private = (P - B) * d` varies systematically with profile features, attack vector, and opinion leaf. |
| H2. Network-context amplification or attenuation | Post-attack peer context changes the private attack effect depending on the target's incoming exposure neighborhood. | `PN_increment_effectivity = (PN - P) * d` is larger when incoming peers have exposure-weighted private post-attack deltas aligned with the adversarial direction; negative values indicate dampening. |
| H3. Central susceptible sender amplification | Network-wide final effects are larger when profiles with high private susceptibility occupy high outgoing exposure positions. | `sum(outgoing_reach_j * AE_private_j)` predicts mean `AE_total_network` above the unweighted private attack effect. |
| H4. Central resilient sender attenuation | Central profiles with low or resistant private attack effects can reduce final network-exposed attack effectiveness. | High outgoing-reach profiles with `AE_private <= 0` predict smaller or negative `PN_increment_effectivity` among exposed receivers. |

## Exposure Network Construction

The current graph substrate is `politisky24_bluesky_v1`, stored under:

```text
data/exposure_networks/politisky24_bluesky_v1/
```

It is derived from the PolitiSky24 Bluesky user-network engagement dataset. The package contains `8,483` observed positions and `1,179,750` directed full exposure edges.

The runtime edge direction is:

```text
source_position_id -> target_position_id
visible peer -> exposed receiver
```

This means the source position represents a profile whose output was plausibly visible to the target position. The edge is derived from observed engagement in the raw data: if an engaged user liked, reposted, or quoted another user's content, the content author was treated as plausibly visible to the engaged user in the upstream construction. The runtime package stores this as a visible-peer-to-exposed-receiver relation for pipeline use.

The raw exposure weight is:

```text
exposure_raw_weight = 0.35 * likes + 0.80 * reposts + 0.90 * quotes
```

Repeated interactions are summed, log-compressed, and normalized to `[0, 1]`. The normalized value is stored as `exposure_weight`. Observed engagement is treated as plausible exposure, not confirmed reading or a full feed-ranking model.

## Profile-To-Position Assignment

Stage `01b_assign_exposure_network_positions` assigns every unique Stage 01 profile to one empirical graph position.

The default assignment mode is:

```text
ranked_seeded_profiles
```

The assignment rule is deterministic:

- take the required number of empirical positions from `assignment_positions.csv` by assignment rank;
- shuffle profile IDs with the run seed;
- assign one profile ID to one position ID;
- do not use profile traits, opinion leaves, attack leaves, or outcome scores.

The same profile-position assignment is reused across all opinion leaves, attacks, controls, and measurement phases. This preserves the ability to test interactions between profile susceptibility and empirical network position.

## How The Exposure Network Is Used

### Stage `02b`: baseline network exposure

Stage `02b_assess_network_exposure_opinions` constructs `BN`.

For each target profile and opinion leaf:

- find all assigned incoming empirical peers where `peer_position -> target_position`;
- keep only peers with completed private baseline assessments for the same opinion leaf;
- compute full peer-count and exposure-weighted summary metrics over the scored incoming neighborhood;
- send a bounded set of `peer_exemplars` to the LLM prompt for rationale context.

`max_exemplars=8` bounds how many peer rationales are sent to the prompt. It does not truncate the scientific incoming peer neighborhood used for peer counts and exposure-weighted metrics.

### Stage `04b`: post-attack network exposure

Stage `04b_assess_post_attack_network_exposure_opinions` constructs `PN`.

For each target profile, opinion leaf, and attack/control condition:

- find incoming empirical peers through the assigned exposure graph;
- keep only peers under the same opinion leaf and the same attack/control condition;
- compute same-condition peer post-attack means, attack-delta means, and exposure-weighted summaries;
- send bounded `peer_exemplars` to the post-attack network prompt.

This phase may amplify, dampen, leave unchanged, or rarely reverse the private post-attack movement. It does not force movement only in the adversarial direction.

### Prompts

The network prompts use `exposure_weight`, not profile affinity.

- `network_exposure_opinion.md` anchors on the target profile's private baseline score and uses incoming peer baseline evaluations as context.
- `post_attack_network_exposure_opinion.md` anchors on the target profile's private post-attack score and uses same-condition incoming peer post-attack evaluations as context.

Both prompts explicitly instruct the model not to mechanically average peer scores. The output remains the target profile's own profile-consistent opinion.

### Stage `05`: analysis-ready output

Stage `05_compute_effectivity_deltas` flattens exposure-network position and context fields into the analysis tables. It emits:

- profile network-position metadata, including graph ID, position ID, role, community, and assignment rank;
- receiver-side exposure fields, including incoming peer count and incoming exposure weight;
- sender-side reach fields, including outgoing visibility and cascade-reach proxies;
- bridge and centrality fields;
- context metrics from `BN` and `PN`;
- hypothesis-ready quantities such as `ae_private`, `bn_increment`, `pn_increment`, `ae_total_network`, and centrality-weighted private activation.

The primary private attack effect remains `P - B`. The post-attack network phase is additive and is analyzed separately through `PN - P` and `PN - B`.

## Workbench Function

The profile network workbench is the inspection and execution surface for the four measurement states:

```text
Baseline | Network exposure | Post exposure | Post network
```

In pipeline-backed mode, the workbench reads Stage `01b` exposure assignments and visualizes the empirical directed exposure network together with available `B`, `BN`, `P`, and `PN` results. It lets the user inspect profile attributes, network position, peer-context conditions, prompt payloads, and measurement outputs.

The pipeline-backed exposure network is the scientific network used for `02b`, `04b`, and Stage `05` analysis. Some live-mode workbench components and visual compatibility fields still retain profile-affinity naming for layout or legacy API compatibility; those fields should not be interpreted as the empirical peer-selection rule for the pipeline network-exposure analyses.

## Current Implementation Map

| Component | Function |
|---|---|
| `data/exposure_networks/politisky24_bluesky_v1` | Stable empirical graph substrate. |
| `src/backend/utils/network_exposure` | Loads graph package, assigns profiles to positions, resolves incoming exposure neighborhoods, builds `BN` and `PN` context payloads. |
| `01b_assign_exposure_network_positions` | Adds one stable empirical position assignment per profile. |
| `02b_assess_network_exposure_opinions` | Measures `BN` using same-opinion incoming empirical peer baseline context. |
| `04b_assess_post_attack_network_exposure_opinions` | Measures `PN` using same-condition incoming empirical peer post-attack context. |
| `05_compute_effectivity_deltas` | Emits private, network, and exposure-position analysis variables. |
| `profile_network_workbench` | Provides a visual and prompt-level inspection surface for the four measurement states. |
