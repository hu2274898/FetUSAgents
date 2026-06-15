"""Typed result schemas shared by the Coordinator and both workflows.

Plain :mod:`dataclasses` are used to keep the package lightweight and
dependency-free. Every dataclass exposes :meth:`to_dict` for trivial JSON
serialisation, since results are written to disk in several places.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class QueryType(str, Enum):
    """Top-level routing decision produced by the Coordinator."""

    SPECIFIC = "specific"
    GENERAL = "general"


class TaskType(str, Enum):
    """Concrete task identifier resolved by the Coordinator.

    The ten ``specific`` values mirror the keys of ``TASK_SPECS`` inside
    :mod:`fetusagents.specific`; the two ``general`` values describe the
    open-ended workflows wrapped from :mod:`fetusagents.core`.
    """

    PLANE_CLASSIFICATION = "plane_classification"
    PLANE_BINARY = "plane_binary"
    BRAIN_SUBPLANE = "brain_subplane"
    BRAIN_SUBPLANE_BINARY = "brain_subplane_binary"
    GA_TRIMESTER_BINARY = "ga_trimester_binary"
    GA_TRIMESTER_MULTI = "ga_trimester_multi"
    HC_ESTIMATION_PIXEL = "hc_estimation_pixel"
    AC_ESTIMATION_PIXEL = "ac_estimation_pixel"
    AOP_BINARY = "aop_binary"
    STOMACH_VOLUME_ESTIMATION = "stomach_volume_estimation"

    IMAGE_CAPTION = "image_caption"
    VIDEO_SUMMARY = "video_summary"

    @classmethod
    def specific_values(cls) -> List["TaskType"]:
        return [
            cls.PLANE_CLASSIFICATION,
            cls.PLANE_BINARY,
            cls.BRAIN_SUBPLANE,
            cls.BRAIN_SUBPLANE_BINARY,
            cls.GA_TRIMESTER_BINARY,
            cls.GA_TRIMESTER_MULTI,
            cls.HC_ESTIMATION_PIXEL,
            cls.AC_ESTIMATION_PIXEL,
            cls.AOP_BINARY,
            cls.STOMACH_VOLUME_ESTIMATION,
        ]

    @classmethod
    def general_values(cls) -> List["TaskType"]:
        return [cls.IMAGE_CAPTION, cls.VIDEO_SUMMARY]


@dataclass
class CoordinatorDecision:
    """Output of :class:`fetusagents.coordinator.Coordinator.route`.

    ``confidence`` is heuristic (a value in ``[0, 1]``) and ``route_reason``
    is a human-readable trace so the routing decision can be explained.
    """

    query_type: QueryType
    task_type: TaskType
    confidence: float
    route_reason: str
    parsed_options: Dict[str, str] = field(default_factory=dict)
    overrides: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["query_type"] = self.query_type.value
        d["task_type"] = self.task_type.value
        return d


@dataclass
class Report:
    findings: str = ""
    impression: str = ""
    note: str = ""

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class WorkflowResult:
    """Unified result type returned by both workflows.

    The same shape is produced regardless of which workflow ran, which
    keeps downstream consumers (CLI output, batch script, evaluation
    notebooks) provider-agnostic.
    """

    query_type: QueryType
    task_type: TaskType
    input_path: str
    question: str
    coordinator: Dict[str, Any] = field(default_factory=dict)
    options: Dict[str, str] = field(default_factory=dict)

    final_answer: Optional[str] = None
    final_option_text: Optional[str] = None
    route: str = ""

    voters: List[Dict[str, Any]] = field(default_factory=list)
    tool_outputs: Dict[str, Any] = field(default_factory=dict)
    rag_snippets: List[str] = field(default_factory=list)
    evidence_bank: Dict[str, Any] = field(default_factory=dict)

    report: Report = field(default_factory=Report)
    summary: str = ""

    dry_run: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "query_type": self.query_type.value,
            "task_type": self.task_type.value,
            "input_path": self.input_path,
            "question": self.question,
            "options": self.options,
            "final_answer": self.final_answer,
            "final_option_text": self.final_option_text,
            "route": self.route,
            "coordinator": self.coordinator,
            "voters": self.voters,
            "tool_outputs": self.tool_outputs,
            "rag_snippets": self.rag_snippets,
            "evidence_bank": self.evidence_bank,
            "report": self.report.to_dict(),
            "summary": self.summary,
            "dry_run": self.dry_run,
        }
        if self.error is not None:
            d["error"] = self.error
        return d
