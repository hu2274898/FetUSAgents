"""Shared base type for tool-backed agents.

A :class:`ToolAgent` is a self-contained unit that wraps one or more
underlying CV / ML tools, applies a fixed arbitration policy, and emits
a structured payload. Its ``runner`` is a plain callable — the signature
and return type are deliberately left untyped at this layer so the two
codepaths in this repo can share metadata (``name``, ``description``,
``tools``) without forcing their inputs and outputs to converge:

* the general workflow's experts take ``case_dir`` (a directory of
  frames) and return a dict ``{"task", "algo_results", "expert_text"}``
* the specific VQA tool runners take ``sample: VQASample`` (one image +
  options) and return ``ToolDecision`` (a single-letter answer + detail)

What the two share is the *agent metaphor* — each ToolAgent has a name,
a one-line description, and an enumerated list of underlying tools it
draws on. Discovery and registry consumers can rely on those fields
without knowing how the runner is invoked.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


@dataclass
class ToolAgent:
    """A modality-specific tool agent.

    Attributes:
        name: stable identifier used by the orchestrator / dispatcher.
        description: one-line human-readable summary.
        tools: list of underlying CV / ML tool names this agent uses.
        runner: the callable that performs the work. Signature is
            codepath-specific (see module docstring).
    """

    name: str
    description: str
    tools: List[str] = field(default_factory=list)
    runner: Optional[Callable[..., Any]] = None
