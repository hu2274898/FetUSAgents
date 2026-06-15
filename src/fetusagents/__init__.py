"""Top-level package for FetUSAgents.

A tool-augmented multi-agent system for fetal ultrasound interpretation.
Provides automatic routing between two workflows:

* specific VQA  - answer multiple-choice / yes-no questions about a scan
* general       - free-form caption / report / video summary

The unified entry points are :func:`fetusagents.cli.main` and
:func:`fetusagents.coordinator.Coordinator.route`.
"""
from .schemas import (
    QueryType,
    TaskType,
    CoordinatorDecision,
    WorkflowResult,
)
from .config import FetUSConfig, load_config

__version__ = "0.1.0"

__all__ = [
    "QueryType",
    "TaskType",
    "CoordinatorDecision",
    "WorkflowResult",
    "FetUSConfig",
    "load_config",
    "__version__",
]
