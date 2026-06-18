from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from src.backend.agentic_framework.base_agent import BaseJsonAgent
from src.backend.utils.schemas import (
    OpinionAssessment,
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


class BaselineOpinionAgent:
    def __init__(self, base_agent: BaseJsonAgent, model_name: str):
        self.base = base_agent
        self.model_name = model_name

    def assess(
        self,
        run_id: str,
        call_id: str,
        scenario_id: str,
        opinion_leaf: str,
        profile: ProfileConfiguration,
        review_feedback: Optional[str] = None,
    ) -> OpinionAssessment:
        payload = {
            "scenario_id": scenario_id,
            "opinion_leaf": opinion_leaf,
            "profile": profile.model_dump(),
        }
        if review_feedback:
            payload["review_feedback"] = review_feedback
        response = self.base.run(
            prompt_name="baseline_opinion.md",
            payload=payload,
            response_model=OpinionResponse,
            run_id=run_id,
            call_id=call_id,
        )
        assert isinstance(response, OpinionResponse)
        return OpinionAssessment(
            scenario_id=scenario_id,
            phase="baseline",
            opinion_leaf=opinion_leaf,
            score=response.score,
            confidence=response.confidence,
            reasoning=response.reasoning,
            model_name=self.model_name,
        )


class NetworkExposureOpinionAgent:
    def __init__(self, base_agent: BaseJsonAgent, model_name: str):
        self.base = base_agent
        self.model_name = model_name

    def assess(
        self,
        run_id: str,
        call_id: str,
        scenario_id: str,
        opinion_leaf: str,
        profile: ProfileConfiguration,
        baseline_score: int,
        network_context: Dict[str, Any],
        review_feedback: Optional[str] = None,
    ) -> OpinionAssessment:
        payload = {
            "scenario_id": scenario_id,
            "opinion_leaf": opinion_leaf,
            "profile": profile.model_dump(),
            "baseline_score": baseline_score,
            "network_context": network_context,
        }
        if review_feedback:
            payload["review_feedback"] = review_feedback
        response = self.base.run(
            prompt_name="network_exposure_opinion.md",
            payload=payload,
            response_model=OpinionResponse,
            run_id=run_id,
            call_id=call_id,
        )
        assert isinstance(response, OpinionResponse)
        return OpinionAssessment(
            scenario_id=scenario_id,
            phase="network_exposure_baseline",
            opinion_leaf=opinion_leaf,
            score=response.score,
            confidence=response.confidence,
            reasoning=response.reasoning,
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


class PostAttackOpinionAgent:
    def __init__(self, base_agent: BaseJsonAgent, model_name: str):
        self.base = base_agent
        self.model_name = model_name

    def assess(
        self,
        run_id: str,
        call_id: str,
        scenario_id: str,
        opinion_leaf: str,
        profile: ProfileConfiguration,
        baseline_score: int,
        attack_present: bool,
        adversarial_direction: int = 0,
        attack_leaf: Optional[str] = None,
        attack_vector_spec: Optional[Dict[str, Any]] = None,
        review_feedback: Optional[str] = None,
    ) -> OpinionAssessment:
        spec = attack_vector_spec or {}
        payload = {
            "scenario_id": scenario_id,
            "opinion_leaf": opinion_leaf,
            "profile": profile.model_dump(),
            "baseline_score": baseline_score,
            "attack_present": attack_present,
            "adversarial_direction": adversarial_direction,
            "attack_leaf": attack_leaf,
            "attack_vector_spec": spec,
        }
        if review_feedback:
            payload["review_feedback"] = review_feedback
        response = self.base.run(
            prompt_name="post_attack_opinion.md",
            payload=payload,
            response_model=OpinionResponse,
            run_id=run_id,
            call_id=call_id,
        )
        assert isinstance(response, OpinionResponse)
        return OpinionAssessment(
            scenario_id=scenario_id,
            phase="post_attack",
            opinion_leaf=opinion_leaf,
            score=response.score,
            confidence=response.confidence,
            reasoning=response.reasoning,
            model_name=self.model_name,
        )


class PostAttackNetworkExposureOpinionAgent:
    def __init__(self, base_agent: BaseJsonAgent, model_name: str):
        self.base = base_agent
        self.model_name = model_name

    def assess(
        self,
        run_id: str,
        call_id: str,
        scenario_id: str,
        opinion_leaf: str,
        profile: ProfileConfiguration,
        baseline_score: int,
        private_post_score: int,
        attack_present: bool,
        adversarial_direction: int = 0,
        attack_leaf: Optional[str] = None,
        attack_vector_spec: Optional[Dict[str, Any]] = None,
        post_attack_network_context: Optional[Dict[str, Any]] = None,
        review_feedback: Optional[str] = None,
    ) -> OpinionAssessment:
        payload = {
            "scenario_id": scenario_id,
            "opinion_leaf": opinion_leaf,
            "profile": profile.model_dump(),
            "baseline_score": baseline_score,
            "private_post_score": private_post_score,
            "attack_present": attack_present,
            "adversarial_direction": adversarial_direction,
            "attack_leaf": attack_leaf,
            "attack_vector_spec": attack_vector_spec or {},
            "post_attack_network_context": post_attack_network_context or {},
        }
        if review_feedback:
            payload["review_feedback"] = review_feedback
        response = self.base.run(
            prompt_name="post_attack_network_exposure_opinion.md",
            payload=payload,
            response_model=OpinionResponse,
            run_id=run_id,
            call_id=call_id,
        )
        assert isinstance(response, OpinionResponse)
        return OpinionAssessment(
            scenario_id=scenario_id,
            phase="post_attack_network_exposure",
            opinion_leaf=opinion_leaf,
            score=response.score,
            confidence=response.confidence,
            reasoning=response.reasoning,
            model_name=self.model_name,
        )
