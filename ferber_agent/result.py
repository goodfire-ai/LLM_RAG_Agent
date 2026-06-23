"""Result type for the Ferber agent."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FerberResult:
    answer_text: str
    citations: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
