from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class StarExtraction:
    situation: float
    task: float
    action: float
    result: float
    evidence_spans: Tuple[str, ...] = ()
    evidence_quality: float = 1.0


class StarExtractor:
    """Example-pattern scaffold based on the Instructor/Outlines repository style."""

    def __init__(self, backend: str = "heuristic", model_name: Optional[str] = None, base_url: Optional[str] = None):
        self.backend = backend
        self.model_name = model_name or "llama3.2"
        self.base_url = base_url or "http://localhost:11434/v1"

    def extract(self, text: str) -> Dict[str, float]:
        """Return a STAR-style structure using the same schema pattern from the example repos."""
        if self.backend == "instructor":
            return self._extract_with_instructor_style(text)
        if self.backend == "outlines":
            return self._extract_with_outlines_style(text)
        return self._heuristic_extract(text)

    def _extract_with_instructor_style(self, text: str) -> Dict[str, float]:
        # Mirrors the Instructor example pattern: define a response model and validate structured output.
        # In a real run this would call instructor.from_provider(...).create(..., response_model=...) .
        return self._heuristic_extract(text)

    def _extract_with_outlines_style(self, text: str) -> Dict[str, float]:
        # Mirrors the Outlines example pattern: create a generator from a local model and constrain output.
        return self._heuristic_extract(text)

    def _heuristic_extract(self, text: str) -> Dict[str, float]:
        lowered = text.lower()
        return {
            "situation": 1.0 if any(keyword in lowered for keyword in ["situation", "context", "when"]) else 0.4,
            "task": 1.0 if any(keyword in lowered for keyword in ["task", "responsibility", "role"]) else 0.4,
            "action": 1.0 if any(keyword in lowered for keyword in ["action", "i", "led", "implemented"]) else 0.3,
            "result": 1.0 if any(keyword in lowered for keyword in ["result", "outcome", "improved", "delivered"]) else 0.3,
        }
