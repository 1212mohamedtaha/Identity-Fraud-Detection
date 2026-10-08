"""Verity: verify claims by asking adaptive questions.

The core (``verity.core``) knows nothing about any domain. A *domain pack*
(``verity.packs.*``) plugs in the parts that differ: where claims come from,
what knowledge backs them, which questions to ask and how to grade answers.
"""
from .core.engine import Session
from .core.pack import DomainPack

__all__ = ["DomainPack", "Session"]
