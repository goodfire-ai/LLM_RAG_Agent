"""ferber_agent: modernized Ferber et al. autonomous tool+RAG oncology agent.

Public API:
    from ferber_agent import FerberAgent, FerberResult
"""
from __future__ import annotations

from .agent import FerberAgent
from .result import FerberResult

__all__ = ["FerberAgent", "FerberResult"]
__version__ = "0.1.0"
