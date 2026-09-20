"""Jinja2 prompt template rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, StrictUndefined, TemplateNotFound

from eval_harness.config import resolve_path


class PromptRenderer:
    """Render versioned Jinja2 prompt templates."""

    def __init__(self, template_path: Path | str, base_dir: Path | None = None) -> None:
        self.template_path = resolve_path(base_dir or Path.cwd(), str(template_path))
        self.base_dir = self.template_path.parent
        self._env = Environment(undefined=StrictUndefined, autoescape=False)

    def render(self, **variables: Any) -> str:
        """Render the template with the given variables."""
        try:
            template = self._env.from_string(self.template_path.read_text(encoding="utf-8"))
        except TemplateNotFound as exc:
            msg = f"prompt template not found: {self.template_path}"
            raise FileNotFoundError(msg) from exc
        return template.render(**variables).strip()
