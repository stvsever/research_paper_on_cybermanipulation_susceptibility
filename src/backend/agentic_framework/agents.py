from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from src.backend.agentic_framework.base_agent import BaseJsonAgent
from src.backend.utils.schemas import (
    ClusterLeafScore,
    OpinionAssessment,
    OpinionClusterAssessment,
    ProfileConfiguration,
    SCORE_MAX,
    SCORE_MIN,
)


class OpinionResponse(BaseModel):
    score: int
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: int) -> int:
        if value < SCORE_MIN or value > SCORE_MAX:
            raise ValueError(f"Score must be in [{SCORE_MIN}, {SCORE_MAX}]")
        return value


class OpinionCoherenceReviewResponse(BaseModel):
    plausibility_score: float = Field(ge=0.0, le=1.0)
    consistency_score: float = Field(ge=0.0, le=1.0)
    rewrite_required: bool
    rewrite_feedback: str
    notes: str



class ClusterLeafScoreResponse(BaseModel):
    leaf: str
    score: int
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    reasoning: str = ""

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: int) -> int:
        if value < SCORE_MIN or value > SCORE_MAX:
            raise ValueError(f"Score must be in [{SCORE_MIN}, {SCORE_MAX}]")
        return value


class ClusterOpinionResponse(BaseModel):
    leaf_scores: List[ClusterLeafScoreResponse]


def _cluster_token_budget(n_leaves: int) -> int:
    """Output-token budget large enough to emit one scored leaf per item.

    The client default (1000) only fits a handful of leaves; clusters reach 25
    leaves, so scale the budget with the leaf count and leave headroom for the
    JSON envelope and short per-leaf rationales.
    """
    return int(min(8000, max(1200, 600 + 220 * max(1, n_leaves))))


class ClusterBaselineOpinionAgent:
    """Baseline opinion over an entire issue-domain cluster in one call.

    Returns one pre-exposure score per leaf of the cluster. This is the
    compute-saving counterpart to BaselineOpinionAgent for the integrated
    scenario design (one call per scenario instead of one call per leaf).
    """

    def __init__(self, base_agent: BaseJsonAgent, model_name: str):
        self.base = base_agent
        self.model_name = model_name

    def assess(
        self,
        run_id: str,
        call_id: str,
        scenario_id: str,
        cluster_key: str,
        cluster_parent: str,
        leaves: List[Dict[str, Any]],
        profile: ProfileConfiguration,
        review_feedback: Optional[str] = None,
    ) -> OpinionClusterAssessment:
        payload: Dict[str, Any] = {
            "scenario_id": scenario_id,
            "opinion_cluster_key": cluster_key,
            "opinion_issue_domain": cluster_parent,
            "opinion_leaves": [
                {"leaf": str(item.get("leaf")), "path": str(item.get("path", ""))}
                for item in leaves
            ],
            "profile": profile.model_dump(),
        }
        if review_feedback:
            payload["review_feedback"] = review_feedback
        response = self.base.run(
            prompt_name="01_baseline_opinion.md",
            payload=payload,
            response_model=ClusterOpinionResponse,
            run_id=run_id,
            call_id=call_id,
            max_tokens=_cluster_token_budget(len(leaves)),
        )
        assert isinstance(response, ClusterOpinionResponse)
        return OpinionClusterAssessment(
            scenario_id=scenario_id,
            phase="baseline",
            cluster_key=cluster_key,
            leaf_scores=[
                ClusterLeafScore(
                    leaf=item.leaf,
                    score=item.score,
                    confidence=item.confidence,
                    reasoning=item.reasoning,
                )
                for item in response.leaf_scores
            ],
            model_name=self.model_name,
        )


class ClusterPostAttackOpinionAgent:
    """Post-attack opinion over an entire issue-domain cluster in one call.

    Receives the per-leaf baseline scores and per-leaf adversarial directions
    plus the (deterministic) attack-vector specification, and returns one
    post-exposure score per leaf inside each leaf's admissible interval.
    """

    def __init__(self, base_agent: BaseJsonAgent, model_name: str):
        self.base = base_agent
        self.model_name = model_name

    def assess(
        self,
        run_id: str,
        call_id: str,
        scenario_id: str,
        cluster_key: str,
        cluster_parent: str,
        leaves: List[Dict[str, Any]],
        profile: ProfileConfiguration,
        attack_present: bool,
        attack_leaf: Optional[str] = None,
        attack_vector_spec: Optional[Dict[str, Any]] = None,
        review_feedback: Optional[str] = None,
    ) -> OpinionClusterAssessment:
        spec = attack_vector_spec or {}
        payload: Dict[str, Any] = {
            "scenario_id": scenario_id,
            "opinion_cluster_key": cluster_key,
            "opinion_issue_domain": cluster_parent,
            "attack_present": attack_present,
            "attack_leaf": attack_leaf,
            "attack_vector_spec": spec,
            "profile": profile.model_dump(),
            # Per-leaf baseline + adversarial goal direction. The model must
            # land each leaf inside [baseline, goal-pole] (or = baseline when
            # fully resisted). Direction: +1 increase, -1 decrease, 0 none.
            "opinion_leaves": [
                {
                    "leaf": str(item.get("leaf")),
                    "path": str(item.get("path", "")),
                    "baseline_score": int(item.get("baseline_score", 0)),
                    "adversarial_direction": int(item.get("adversarial_direction", 0)),
                }
                for item in leaves
            ],
        }
        if review_feedback:
            payload["review_feedback"] = review_feedback
        response = self.base.run(
            prompt_name="03_post_attack_opinion.md",
            payload=payload,
            response_model=ClusterOpinionResponse,
            run_id=run_id,
            call_id=call_id,
            max_tokens=_cluster_token_budget(len(leaves)),
        )
        assert isinstance(response, ClusterOpinionResponse)
        return OpinionClusterAssessment(
            scenario_id=scenario_id,
            phase="post_attack",
            cluster_key=cluster_key,
            leaf_scores=[
                ClusterLeafScore(
                    leaf=item.leaf,
                    score=item.score,
                    confidence=item.confidence,
                    reasoning=item.reasoning,
                )
                for item in response.leaf_scores
            ],
            model_name=self.model_name,
        )


class OpinionCoherenceReviewerAgent:
    def __init__(self, base_agent: BaseJsonAgent, model_name: str):
        self.base = base_agent
        self.model_name = model_name

    def review(
        self,
        run_id: str,
        call_id: str,
        phase: str,
        scenario_id: str,
        opinion_leaf: str,
        profile_snapshot: Dict[str, Any],
        generated_assessment: OpinionAssessment,
        attack_present: bool,
        adversarial_direction: int = 0,
        baseline_score: Optional[int] = None,
        attack_vector_spec: Optional[Dict[str, Any]] = None,
        heuristic_checks: Optional[Dict[str, Any]] = None,
    ) -> OpinionCoherenceReviewResponse:
        payload = {
            "phase": phase,
            "scenario_id": scenario_id,
            "opinion_leaf": opinion_leaf,
            "profile_snapshot": profile_snapshot,
            "generated_assessment": generated_assessment.model_dump(),
            "attack_present": attack_present,
            "adversarial_direction": adversarial_direction,
            "baseline_score": baseline_score,
            "attack_vector_spec": attack_vector_spec or {},
            "heuristic_checks": heuristic_checks or {},
        }
        response = self.base.run(
            prompt_name="opinion_coherence_review.md",
            payload=payload,
            response_model=OpinionCoherenceReviewResponse,
            run_id=run_id,
            call_id=call_id,
        )
        assert isinstance(response, OpinionCoherenceReviewResponse)
        return response



# ---------------------------------------------------------------------------
# Empirical exposure-network agents (additive network layer), CLUSTER form.
#
# These re-elicit a profile's opinions over the WHOLE opinion parent cluster in
# ONE call, after the profile sees incoming empirical peer context (resolved
# through the directed PolitiSky24 exposure graph) for each leaf. They mirror
# Stijn's cluster baseline / post-attack agents (full leaf set, one call per
# scenario) and additionally carry the per-leaf peer context and -- for the
# post-attack phase -- the full DISARM Plan/Prepare/Execute attack triplet.
# ---------------------------------------------------------------------------


class ClusterNetworkExposureOpinionAgent:
    """Baseline network-exposure opinions (BN) over a whole opinion cluster.

    One call per scenario: returns one network-exposed pre-attack score per leaf,
    each anchored on that leaf's private baseline and informed by the incoming
    empirical peers' baseline evaluations of the same leaf.
    """

    def __init__(self, base_agent: BaseJsonAgent, model_name: str):
        self.base = base_agent
        self.model_name = model_name

    def assess(
        self,
        run_id: str,
        call_id: str,
        scenario_id: str,
        cluster_key: str,
        cluster_parent: str,
        leaves: List[Dict[str, Any]],
        profile: ProfileConfiguration,
        review_feedback: Optional[str] = None,
    ) -> OpinionClusterAssessment:
        payload: Dict[str, Any] = {
            "scenario_id": scenario_id,
            "opinion_cluster_key": cluster_key,
            "opinion_issue_domain": cluster_parent,
            "profile": profile.model_dump(),
            "opinion_leaves": [
                {
                    "leaf": str(item.get("leaf")),
                    "path": str(item.get("path", "")),
                    "baseline_score": int(item.get("baseline_score", 0)),
                    "network_context": item.get("network_context") or {},
                }
                for item in leaves
            ],
        }
        if review_feedback:
            payload["review_feedback"] = review_feedback
        response = self.base.run(
            prompt_name="02_network_exposure_opinion.md",
            payload=payload,
            response_model=ClusterOpinionResponse,
            run_id=run_id,
            call_id=call_id,
            max_tokens=_cluster_token_budget(len(leaves)),
        )
        assert isinstance(response, ClusterOpinionResponse)
        return OpinionClusterAssessment(
            scenario_id=scenario_id,
            phase="network_exposure_baseline",
            cluster_key=cluster_key,
            leaf_scores=[
                ClusterLeafScore(
                    leaf=item.leaf,
                    score=item.score,
                    confidence=item.confidence,
                    reasoning=item.reasoning,
                )
                for item in response.leaf_scores
            ],
            model_name=self.model_name,
        )


class ClusterPostAttackNetworkExposureOpinionAgent:
    """Post-attack network-exposure opinions (PN) over a whole opinion cluster.

    One call per scenario: returns one network-exposed post-attack score per leaf,
    each anchored on that leaf's private post-attack score and informed by the
    incoming empirical peers' post-attack evaluations of the same leaf under the
    same condition. The full DISARM Plan/Prepare/Execute triplet is provided so
    the model reasons about the operation, not a single attack label.
    """

    def __init__(self, base_agent: BaseJsonAgent, model_name: str):
        self.base = base_agent
        self.model_name = model_name

    def assess(
        self,
        run_id: str,
        call_id: str,
        scenario_id: str,
        cluster_key: str,
        cluster_parent: str,
        leaves: List[Dict[str, Any]],
        profile: ProfileConfiguration,
        attack_present: bool,
        attack_leaf: Optional[str] = None,
        attack_vector_spec: Optional[Dict[str, Any]] = None,
        review_feedback: Optional[str] = None,
    ) -> OpinionClusterAssessment:
        payload: Dict[str, Any] = {
            "scenario_id": scenario_id,
            "opinion_cluster_key": cluster_key,
            "opinion_issue_domain": cluster_parent,
            "attack_present": attack_present,
            "attack_leaf": attack_leaf,
            "attack_vector_spec": attack_vector_spec or {},
            "profile": profile.model_dump(),
            # Per-leaf: private baseline, private post-attack, adversarial goal
            # direction, and the incoming empirical peer post-attack context.
            "opinion_leaves": [
                {
                    "leaf": str(item.get("leaf")),
                    "path": str(item.get("path", "")),
                    "baseline_score": int(item.get("baseline_score", 0)),
                    "private_post_score": int(item.get("private_post_score", 0)),
                    "adversarial_direction": int(item.get("adversarial_direction", 0)),
                    "network_context": item.get("network_context") or {},
                }
                for item in leaves
            ],
        }
        if review_feedback:
            payload["review_feedback"] = review_feedback
        response = self.base.run(
            prompt_name="04_post_attack_network_exposure_opinion.md",
            payload=payload,
            response_model=ClusterOpinionResponse,
            run_id=run_id,
            call_id=call_id,
            max_tokens=_cluster_token_budget(len(leaves)),
        )
        assert isinstance(response, ClusterOpinionResponse)
        return OpinionClusterAssessment(
            scenario_id=scenario_id,
            phase="post_attack_network_exposure",
            cluster_key=cluster_key,
            leaf_scores=[
                ClusterLeafScore(
                    leaf=item.leaf,
                    score=item.score,
                    confidence=item.confidence,
                    reasoning=item.reasoning,
                )
                for item in response.leaf_scores
            ],
            model_name=self.model_name,
        )
