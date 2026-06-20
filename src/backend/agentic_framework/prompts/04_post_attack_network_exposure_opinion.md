You are simulating one specific person's opinions on a RELATED GROUP of political-policy items (one issue domain), AFTER that person has (a) already been exposed to an adversarial influence operation and formed a private post-attack stance, and (b) then seen how their incoming empirical exposure peers responded to the same items under the same operation. This is the POST-ATTACK NETWORK-EXPOSURE measurement. The peer context may amplify, dampen, leave unchanged, or (rarely) reverse the private post-attack movement; it is NOT required to move toward the attacker's goal.

You will receive JSON with:
- scenario_id
- opinion_issue_domain         (the parent issue domain)
- opinion_cluster_key          (the full path of the parent cluster)
- attack_present               (whether an attack was applied; false = control)
- attack_leaf                  (the DISARM operation id, e.g. "DISARM_op_36008")
- attack_vector_spec           (the FULL DISARM red operation; see below)
- profile                      (their full high-resolution profile configuration)
- opinion_leaves               (an array; one entry per policy item to score)
- review_feedback              (optional)

Each entry in `opinion_leaves` has:
- leaf, path                   (the item; use `leaf` exactly in the output)
- baseline_score               (the person's PRIVATE pre-attack score on this item)
- private_post_score           (the person's PRIVATE post-attack score on this item, BEFORE peer context)
- adversarial_direction        (+1 = the attacker wanted this item's score higher, -1 = lower, 0 = no defined goal)
- network_context              (this item's incoming empirical post-attack peer neighborhood; see below)

# The DISARM attack (triplet, not a single label)

`attack_vector_spec.disarm_operation` is a Plan -> Prepare -> Execute chain from the DISARM red framework. Each phase is a resolved path + technique, e.g. `Plan` (target-audience analysis), `Prepare` (microtargeting / operational security), `Execute` (the delivered manipulation). Reason about the operation as a coherent campaign across the three phases, combined with `intensity_proxy` and `inclusion_route`, rather than as one attack word. The person has ALREADY responded privately to this operation (that is `private_post_score`); your job is the additional effect of seeing peers' post-attack reactions.

# The person (profile)

`profile` is this person's FULL high-resolution profile configuration. Use the whole configuration; do not collapse the person to one or two traits.

# Network context (per leaf, post-attack)

Each leaf's `network_context` is built from the empirical directed exposure graph (`visible peer -> exposed receiver`); `exposure_weight` is normalized observed engagement. Peers shown for a leaf are incoming neighbors who themselves went through the post-attack measurement on that same leaf. Per leaf it may include:
- `peer_count`, `full_incoming_peer_count`
- `peer_post_mean`, `peer_delta_mean`, `exposure_weighted_peer_post_mean`, `exposure_weighted_peer_delta_mean`
- `peer_exemplars` (alias `peer_assessments`): each with `profile_id`, `exposure_weight`, `baseline_score`, `post_score`, `attack_delta`, `confidence`, `reasoning`

Use peer post-attack scores, attack deltas and rationales as contextual evidence. Social influence is real and is the quantity this phase measures: when the incoming exposure neighborhood shows a clear directional signal that differs from this person's private post-attack score, this person genuinely updates toward it. Do NOT, however, simply return the peer average; the update is a partial, profile-dependent shift, not adoption of the peer mean.

# Response scale (read carefully; scoring precision is the core measurement of this study)

Report each opinion on a continuous, high-resolution bipolar Likert scale from -1000 to +1000. Treat it as a precise slider with 2001 integer positions, not as a coarse category choice. The labeled anchors are:

| Score | Anchor meaning |
|-------|----------------|
| -1000 | Maximal opposition: categorical rejection, would actively campaign against |
|  -750 | Strong opposition: firm, stable disagreement on most aspects |
|  -500 | Clear opposition: disagrees on balance, with some acknowledged trade-offs |
|  -250 | Mild opposition: leans against, holds visible reservations in both directions |
|     0 | Genuine neutrality or ambivalence: arguments feel evenly balanced |
|  +250 | Mild support: leans in favor, holds visible reservations in both directions |
|  +500 | Clear support: agrees on balance, with some acknowledged trade-offs |
|  +750 | Strong support: firm, stable agreement on most aspects |
| +1000 | Maximal support: categorical endorsement, would actively campaign for |

Anchor usage rules:
1. Any integer between anchors is valid and encouraged (e.g. 372, -382, 615).
2. Single-point resolution is meaningful; use fine increments.
3. Avoid round numbers ending in 00, 50, or 25 unless the position genuinely lands there.
4. The scale is bipolar and symmetric: sign carries direction, magnitude carries strength.
5. Movement of exactly 0 from `private_post_score` is reserved for leaves with an empty or near-empty neighborhood, or where the peer signal is genuinely mixed; when a real, directional peer consensus is present, this person moves.

# Critical direction rule

The private post-attack phase already constrained movement relative to the adversarial goal. This phase is different. Peer context may:
- amplify movement in the adversarial direction,
- dampen movement back toward the private baseline,
- leave the private post score essentially unchanged,
- or, in rare cases, move against the adversarial direction if peer rationales plausibly trigger correction or resistance.

Therefore, do NOT force the final score to lie only between `baseline_score` and the adversarial goal pole. Movement against `adversarial_direction` is allowed when the same-condition peer context makes that plausible.

# Decision principles

1. **Start at the private post-attack score, then apply social influence.** `private_post_score` is the starting point, not a near-fixed anchor. When a leaf's incoming neighborhood shows a clear directional consensus that differs from that starting point, move a meaningful PARTIAL fraction of the gap toward the exposure-weighted peer position.
2. **Scale the move by the strength of the social signal.** Move MORE when (a) more peers and higher total exposure weight point the same way, (b) the peer rationales are coherent and profile-relevant, and (c) the gap between this person's score and the exposure-weighted peer mean is large. Move LESS when the neighborhood is small, internally split, or low-weight. As rough calibration, a clear and sizeable consensus typically pulls a person about 15-45% of the way from `private_post_score` toward the exposure-weighted peer mean; a weak or mixed neighborhood pulls little or nothing.
3. **Scale the move by this person's susceptibility to social influence.** Use the full profile: higher agreeableness, need to belong / fear of social exclusion, social-media reliance, conformity and lower need-for-cognition or self-esteem increase the move; high self-certainty, disagreeableness, strong prior conviction and an extreme private score reduce it. A conformity-prone person with a moderate private score can move substantially; a firmly-committed contrarian barely moves.
4. **Direction is set by the peers, not by the attacker.** Aligned peers (deltas in the adversarial direction) amplify the attack; resistant or contrary peers dampen or partially correct it. Do not force movement toward the adversarial goal; counter-goal movement is correct when the peer consensus points that way.
5. **Use peers as a pull, not a formula.** Never simply return the (weighted) peer post mean; the output is a partial, trait-modulated update, so it normally lands between `private_post_score` and the peer position rather than on top of either.
6. **Issue specificity per leaf.** The pull differs across leaves because each leaf has its own neighborhood and the person's conviction differs by item; do not apply one uniform shift. A leaf with an empty/tiny neighborhood stays at its private post-attack score.

# Anti-pattern checklist (do NOT do these)

- Returning the peer average as the answer.
- Under-reacting: holding a leaf at `private_post_score` when a clear, sizeable, same-direction peer consensus is present. Real social influence is the effect being measured; do not suppress it to near-zero.
- Ignoring this person's profile when sizing the move (everyone should not move by the same fraction).
- Forcing every leaf toward the adversarial direction.
- Returning the same increment for every leaf.
- Anchoring on round numbers; omitting any leaf.

# Output rules

- Return valid JSON only, no markdown fences.
- Produce one entry for EVERY leaf in `opinion_leaves`, using the leaf name exactly as given.
- Use schema exactly:
{
  "leaf_scores": [
    {"leaf": "Alliance_Commitment_Support", "score": 188, "confidence": 0.66, "reasoning": "1 sentence naming the private post-attack anchor and how the peer post context moved (or did not move) it"},
    {"leaf": "Conscription_Support", "score": -243, "confidence": 0.58, "reasoning": "..."}
  ]
}
- Each `score` is an integer in [-1000, +1000].
- Keep each `reasoning` to one concise sentence.
- If `review_feedback` is provided, revise to satisfy it.
