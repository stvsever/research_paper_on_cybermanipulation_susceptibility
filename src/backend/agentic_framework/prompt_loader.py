from __future__ import annotations

from pathlib import Path


class PromptLoader:
    def __init__(self, prompts_dir: str | Path):
        self.prompts_dir = Path(prompts_dir)

    def load(self, prompt_name: str) -> str:
        prompt_path = self.prompts_dir / prompt_name
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt not found: {prompt_path}")
        return prompt_path.read_text(encoding="utf-8")
