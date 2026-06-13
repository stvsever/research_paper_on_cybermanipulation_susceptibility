from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import httpx


LOGGER = logging.getLogger(__name__)


class OpenRouterClient:
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_sec: int = 90,
        referer: str = "https://ghent-research.local",
        app_title: str = "GhentUniversity_AttackOpinionSimulation",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_sec = timeout_sec
        self.referer = referer
        self.app_title = app_title

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1000,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.referer,
            "X-Title": self.app_title,
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format is not None:
            payload["response_format"] = response_format

        with httpx.Client(timeout=self.timeout_sec) as client:
            response = client.post(self.BASE_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        content = self._extract_content(data)
        return content, data

    @staticmethod
    def _extract_content(payload: Dict[str, Any]) -> str:
        choices = payload.get("choices", [])
        if not choices:
            raise RuntimeError("OpenRouter response has no choices")

        message = choices[0].get("message", {})
        content = message.get("content", "")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    maybe_text = item.get("text")
                    if isinstance(maybe_text, str):
                        text_parts.append(maybe_text)
            return "\n".join(text_parts)

        # Reasoning-capable models occasionally return a null content field
        # (reasoning-only turn). Returning an empty string routes the attempt
        # into the JSON repair/retry loop instead of aborting the call.
        if content is None:
            return ""

        raise RuntimeError("Unsupported content format from OpenRouter")
