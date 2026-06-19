from __future__ import annotations

from pathlib import Path

from src.backend.agentic_framework.agents import (
    ClusterBaselineOpinionAgent,
    ClusterNetworkExposureOpinionAgent,
    ClusterPostAttackNetworkExposureOpinionAgent,
    ClusterPostAttackOpinionAgent,
    OpinionCoherenceReviewerAgent,
)
from src.backend.agentic_framework.base_agent import BaseJsonAgent
from src.backend.agentic_framework.openrouter_client import OpenRouterClient
from src.backend.agentic_framework.prompt_loader import PromptLoader


class AgentFactory:
    def __init__(
        self,
        prompts_dir: str | Path,
        openrouter_api_key: str,
        openrouter_model: str,
        max_repair_iter: int,
        temperature: float,
        timeout_sec: int,
        save_raw_dir: str | None,
    ) -> None:
        self.prompt_loader = PromptLoader(prompts_dir)
        self.client = OpenRouterClient(
            api_key=openrouter_api_key,
            model=openrouter_model,
            timeout_sec=timeout_sec,
        )
        self.max_repair_iter = max_repair_iter
        self.temperature = temperature
        self.save_raw_dir = save_raw_dir
        self.model_name = openrouter_model

    def _base(self, name: str) -> BaseJsonAgent:
        return BaseJsonAgent(
            name=name,
            client=self.client,
            prompt_loader=self.prompt_loader,
            max_repair_iter=self.max_repair_iter,
            temperature=self.temperature,
            save_raw_dir=self.save_raw_dir,
        )

    def opinion_coherence_reviewer_agent(self) -> OpinionCoherenceReviewerAgent:
        return OpinionCoherenceReviewerAgent(
            self._base("opinion_coherence_reviewer"),
            model_name=self.model_name,
        )

    def cluster_baseline_opinion_agent(self) -> ClusterBaselineOpinionAgent:
        return ClusterBaselineOpinionAgent(
            self._base("cluster_baseline_opinion"), model_name=self.model_name
        )

    def cluster_post_attack_opinion_agent(self) -> ClusterPostAttackOpinionAgent:
        return ClusterPostAttackOpinionAgent(
            self._base("cluster_post_attack_opinion"), model_name=self.model_name
        )

    # --- Additive empirical exposure-network agents (CLUSTER network layer) ---
    def cluster_network_exposure_opinion_agent(self) -> ClusterNetworkExposureOpinionAgent:
        return ClusterNetworkExposureOpinionAgent(
            self._base("cluster_network_exposure_opinion"), model_name=self.model_name
        )

    def cluster_post_attack_network_exposure_opinion_agent(self) -> ClusterPostAttackNetworkExposureOpinionAgent:
        return ClusterPostAttackNetworkExposureOpinionAgent(
            self._base("cluster_post_attack_network_exposure_opinion"), model_name=self.model_name
        )
