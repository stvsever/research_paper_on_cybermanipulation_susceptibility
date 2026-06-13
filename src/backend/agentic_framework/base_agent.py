from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel, ValidationError

from src.backend.agentic_framework.openrouter_client import OpenRouterClient
from src.backend.agentic_framework.prompt_loader import PromptLoader
from src.backend.utils.io import clean_filename, ensure_dir, write_json


LOGGER = logging.getLogger(__name__)


class BaseJsonAgent:
    def __init__(
        self,
        name: str,
        client: OpenRouterClient,
        prompt_loader: PromptLoader,
        max_repair_iter: int = 2,
        temperature: float = 0.2,
        save_raw_dir: Optional[str] = None,
    ) -> None:
        self.name = name
        self.client = client
        self.prompt_loader = prompt_loader
        self.max_repair_iter = max_repair_iter
        self.temperature = temperature
        self.save_raw_dir = Path(save_raw_dir) if save_raw_dir else None
        if self.save_raw_dir is not None:
            ensure_dir(self.save_raw_dir)

    def run(
        self,
        prompt_name: str,
        payload: Dict[str, Any],
        response_model: Type[BaseModel],
        run_id: str,
        call_id: str,
    ) -> BaseModel:
        system_prompt = self.prompt_loader.load(prompt_name)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, indent=2),
            },
        ]

        max_attempts = self.max_repair_iter + 1
        for attempt in range(1, max_attempts + 1):
            try:
                content, raw = self.client.chat(
                    messages=messages,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
            except Exception as exc:
                # Transient transport or payload-shape failures retry within
                # the same attempt budget instead of aborting the call.
                LOGGER.warning(
                    "Agent %s chat attempt %s/%s failed for %s: %s",
                    self.name, attempt, max_attempts, call_id, exc,
                )
                if attempt >= max_attempts:
                    raise
                continue

            self._persist_raw(
                run_id=run_id,
                call_id=call_id,
                attempt=attempt,
                prompt_name=prompt_name,
                request_payload=payload,
                response_text=content,
                raw_response=raw,
            )

            try:
                parsed = self._extract_json(content)
                return response_model.model_validate(parsed)
            except (json.JSONDecodeError, ValidationError, RuntimeError) as exc:
                LOGGER.warning(
                    "Agent %s failed attempt %s/%s for %s: %s",
                    self.name,
                    attempt,
                    max_attempts,
                    call_id,
                    exc,
                )
                if attempt >= max_attempts:
                    raise RuntimeError(
                        f"{self.name} failed after {max_attempts} attempts for {call_id}"
                    ) from exc

                repair_prompt = (
                    "The previous output was invalid. Return ONLY strict JSON that matches the schema. "
                    f"Validation error: {str(exc)}"
                )
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": repair_prompt})

        raise RuntimeError("Unreachable agent failure state")

    @staticmethod
    def _extract_json(content: str) -> Dict[str, Any]:
        stripped = content.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 3:
                stripped = "\n".join(lines[1:-1]).strip()
        if not stripped:
            raise RuntimeError("Empty model output")
        data = json.loads(stripped)
        if not isinstance(data, dict):
            raise RuntimeError("Model output JSON must be an object")
        return data

    def _persist_raw(
        self,
        run_id: str,
        call_id: str,
        attempt: int,
        prompt_name: str,
        request_payload: Dict[str, Any],
        response_text: str,
        raw_response: Dict[str, Any],
    ) -> None:
        if self.save_raw_dir is None:
            return

        filename = clean_filename(f"{run_id}_{self.name}_{call_id}_attempt_{attempt}.json")
        write_json(
            self.save_raw_dir / filename,
            {
                "agent_name": self.name,
                "prompt_name": prompt_name,
                "call_id": call_id,
                "attempt": attempt,
                "request_payload": request_payload,
                "response_text": response_text,
                "raw_response": raw_response,
            },
        )
