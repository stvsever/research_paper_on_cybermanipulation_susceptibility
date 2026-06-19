from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

LAB_ROOT = Path(__file__).resolve().parents[3]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from src.backend.agentic_framework.base_agent import BaseJsonAgent
from src.backend.agentic_framework.factory import AgentFactory


class TinyResponse(BaseModel):
    ok: bool


class FakePromptLoader:
    def load(self, prompt_name: str) -> str:
        return f"Prompt: {prompt_name}"


class CapturingClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1000,
        response_format: dict[str, str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": response_format,
            }
        )
        return '{"ok": true}', {"id": "fake-call"}


def test_base_json_agent_passes_and_persists_max_tokens(tmp_path: Path) -> None:
    client = CapturingClient()
    agent = BaseJsonAgent(
        name="post_attack_network_exposure_opinion",
        client=client,  # type: ignore[arg-type]
        prompt_loader=FakePromptLoader(),  # type: ignore[arg-type]
        max_repair_iter=0,
        max_tokens=2000,
        save_raw_dir=str(tmp_path),
    )

    result = agent.run(
        prompt_name="post_attack_network_exposure_opinion.md",
        payload={"scenario_id": "s1"},
        response_model=TinyResponse,
        run_id="run_1",
        call_id="profile_0001",
    )

    assert result.ok is True
    assert client.calls[0]["max_tokens"] == 2000

    raw_files = list(tmp_path.glob("*.json"))
    assert len(raw_files) == 1
    raw_payload = json.loads(raw_files[0].read_text(encoding="utf-8"))
    assert raw_payload["max_tokens"] == 2000


def test_agent_factory_uses_2000_tokens_only_for_post_network() -> None:
    factory = AgentFactory(
        prompts_dir=LAB_ROOT / "src/backend/agentic_framework/prompts",
        openrouter_api_key="test-key",
        openrouter_model="mock/model",
        max_repair_iter=2,
        temperature=0.2,
        timeout_sec=30,
        save_raw_dir=None,
    )

    assert factory.baseline_opinion_agent().base.max_tokens == 1000
    assert factory.network_exposure_opinion_agent().base.max_tokens == 1000
    assert factory.post_attack_opinion_agent().base.max_tokens == 1000
    assert factory.opinion_coherence_reviewer_agent().base.max_tokens == 1000
    assert factory.post_attack_network_exposure_opinion_agent().base.max_tokens == 2000
