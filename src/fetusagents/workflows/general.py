"""General workflow.

This module is the public adapter for the open-ended clinical pipeline
that lives under :mod:`fetusagents.core`. Its entry point
:func:`fetusagents.general.orchestrate` dispatches internally between

* the standard image-caption / clinical workflow, and
* the video-summary workflow,

depending on the user query. We treat that function as a black box and
adapt its return type to :class:`WorkflowResult`.
"""
from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Optional, Tuple

from ..config import FetUSConfig, load_config
from ..coordinator import Coordinator
from ..schemas import CoordinatorDecision, QueryType, Report, TaskType, WorkflowResult


_SECTION_RE = re.compile(
    r"^(Findings|Impression|Impressions|Note|Notes)\s*:\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _split_report_blob(text: str) -> Tuple[str, str, str]:
    """Split a ``Findings: ... Impression: ... Note: ...`` blob into parts.

    ``general.orchestrate`` returns a single string with section headers
    already in place. Without splitting, the CLI template would wrap it
    in another ``Findings:`` / ``Impression:`` shell, producing nested
    headers in ``report.txt``.

    Unknown content before the first header is appended to findings so
    nothing gets dropped.
    """
    if not text:
        return "", "", ""
    sections = {"findings": [], "impression": [], "note": []}
    current = "findings"
    for line in text.splitlines():
        m = _SECTION_RE.match(line.strip())
        if m:
            head = m.group(1).lower().rstrip("s")
            current = head if head in sections else current
            continue
        sections[current].append(line)
    return (
        "\n".join(sections["findings"]).strip("\n"),
        "\n".join(sections["impression"]).strip("\n"),
        "\n".join(sections["note"]).strip("\n"),
    )


def run_single(
    *,
    query: str,
    input_path: str,
    case_dir: Optional[str] = None,
    cfg: Optional[FetUSConfig] = None,
    dry_run: bool = False,
    decision: Optional[CoordinatorDecision] = None,
) -> WorkflowResult:
    """Run the general workflow on a single image / case dir / video folder."""
    cfg = cfg or load_config()
    if decision is None:
        decision = Coordinator().route(query=query, input_path=input_path, mode="general")

    resolved_case_dir = _resolve_case_dir(input_path, case_dir)

    if dry_run:
        return _mock_result(query, input_path, decision)

    return asyncio.run(
        _run_single_async(
            query=query,
            case_dir=resolved_case_dir,
            input_path=input_path,
            decision=decision,
            cfg=cfg,
        )
    )


async def _run_single_async(
    *,
    query: str,
    case_dir: str,
    input_path: str,
    decision: CoordinatorDecision,
    cfg: FetUSConfig,
) -> WorkflowResult:
    # IMPORTANT: export env vars BEFORE the lazy import. TOOL_CONFIG
    # freezes ``FETALAGENT_*_PYTHON`` defaults at class-definition time,
    # which happens during ``from ..general import orchestrate``.
    cfg.export_environment()

    # Lazy import so simply loading this module never pulls in autogen /
    # torch / openai.
    from ..general import orchestrate

    final_text = await orchestrate(query, case_dir)
    findings, impression, note = _split_report_blob(final_text)
    report = Report(findings=findings, impression=impression, note=note)
    return WorkflowResult(
        query_type=QueryType.GENERAL,
        task_type=decision.task_type,
        input_path=input_path,
        question=query,
        coordinator={
            "confidence": decision.confidence,
            "route_reason": decision.route_reason,
        },
        report=report,
        summary=final_text,
        evidence_bank={"case_dir": case_dir},
    )


def _resolve_case_dir(input_path: str, case_dir: Optional[str]) -> str:
    """Compute the directory ``orchestrate`` expects.

    ``orchestrate`` always reads its image list with ``os.listdir(case_dir)``
    so a single image must be packed into a directory by the caller.

    The resolved path is converted to an **absolute** path: the CV tools
    are launched as subprocesses with their own ``cwd=`` set to the tool
    directory, so any relative path the caller passed in would break
    ``os.path.exists`` checks inside those subprocesses.
    """
    if case_dir and os.path.isdir(case_dir):
        return os.path.abspath(case_dir)
    if os.path.isdir(input_path):
        return os.path.abspath(input_path)
    if os.path.isfile(input_path):
        parent = os.path.dirname(input_path) or "."
        return os.path.abspath(parent)
    raise ValueError(
        f"Could not resolve a directory for general workflow from "
        f"input_path={input_path!r} case_dir={case_dir!r}"
    )


def _mock_result(query: str, input_path: str, decision: CoordinatorDecision) -> WorkflowResult:
    summary = (
        f"[dry_run] task_type={decision.task_type.value}\n"
        f"Reason: {decision.route_reason}\n"
        f"Input: {input_path}\n"
        f"Query: {query[:200]}"
    )
    return WorkflowResult(
        query_type=QueryType.GENERAL,
        task_type=decision.task_type,
        input_path=input_path,
        question=query,
        coordinator={
            "confidence": decision.confidence,
            "route_reason": decision.route_reason,
        },
        tool_outputs={"mock": True},
        evidence_bank={"mock": True, "case_dir": input_path},
        report=Report(
            findings="[dry_run] Findings stub.",
            impression="[dry_run] Impression stub.",
            note="Set --dry_run=false and configure paths to obtain real results.",
        ),
        summary=summary,
        dry_run=True,
    )
