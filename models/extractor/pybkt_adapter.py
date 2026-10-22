from __future__ import annotations

from typing import Dict, List


class PyBKTAdapter:
    """Minimal pyBKT-style adapter inspired by the pyBKT repository usage pattern."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, evidence_sequences: List[Dict[str, float]]):
        # This is a lightweight stand-in for pyBKT's fit() interface.
        # In a real project you would replace this with the actual pyBKT library call.
        return {"seed": self.seed, "trained": True, "evidence_sequences": len(evidence_sequences)}

    def predict(self, evidence_sequences: List[Dict[str, float]]) -> List[float]:
        return [sum(item.values()) / max(1, len(item)) for item in evidence_sequences]
