"""Specific VQA workflow.

This module is the public adapter for the 5-Voter / Expert / Checker /
RAG / Report pipeline that lives under :mod:`fetusagents.specific`. It
adds two things on top of the underlying pipeline:

1. **Automatic ``task_type`` inference** (via :class:`Coordinator`) so the
   caller does not have to pass ``--task_name``.
2. **A normalised :class:`WorkflowResult`** so both workflows have the
   same JSON shape on disk.

Heavy work (loading CV tool weights, calling the LLM) is gated behind
``dry_run=False``; ``dry_run=True`` returns a deterministic mock so the
control flow is exercisable in tests.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import FetUSConfig, load_config
from ..coordinator import Coordinator
from ..schemas import CoordinatorDecision, QueryType, Report, TaskType, WorkflowResult


_DEFAULT_OPTION_TO_TEXT: Dict[str, Dict[str, str]] = {
    "plane_classification": {
        "A": "Fetal abdomen",
        "B": "Fetal femur",
        "C": "Fetal brain",
        "D": "Fetal thorax",
    },
    "plane_binary": {"A": "Yes", "B": "No"},
    "brain_subplane": {
        "A": "Trans-thalamic",
        "B": "Trans-cerebellar",
        "C": "Trans-ventricular",
    },
    "brain_subplane_binary": {"A": "Yes", "B": "No"},
    "ga_trimester_binary": {"A": "Yes", "B": "No"},
    "ga_trimester_multi": {
        "A": "First trimester",
        "B": "Second trimester",
        "C": "Third trimester",
    },
    "hc_estimation_pixel": {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"},
    "ac_estimation_pixel": {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"},
    "aop_binary": {
        "A": "AoP >= 120°, spontaneous vaginal delivery is indicated",
        "B": "AoP < 120°, instrumental delivery or cesarean may be necessary",
    },
    "stomach_volume_estimation": {
        "A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D",
    },
}


def _options_for(task_type: TaskType, parsed: Dict[str, str]) -> Dict[str, str]:
    if parsed:
        return parsed
    return dict(_DEFAULT_OPTION_TO_TEXT.get(task_type.value, {}))


def _options_to_list(options: Dict[str, str]) -> List[str]:
    return [f"{letter}. {options[letter]}" for letter in sorted(options.keys())]


def run_single(
    *,
    image_path: str,
    question: str,
    options: Optional[Dict[str, str]] = None,
    task_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    cfg: Optional[FetUSConfig] = None,
    dry_run: bool = False,
    decision: Optional[CoordinatorDecision] = None,
) -> WorkflowResult:
    """Run the specific VQA workflow on a single sample.

    Parameters mirror the high-level CLI; if ``decision`` is omitted, the
    Coordinator is invoked to resolve ``task_type`` from ``question`` and
    ``options``.
    """
    cfg = cfg or load_config()
    if decision is None:
        decision = Coordinator().route(
            query=question,
            input_path=image_path,
            options=options,
            mode="specific",
            task_name=task_name,
        )
    options = _options_for(decision.task_type, decision.parsed_options or (options or {}))

    if dry_run:
        return _mock_result(image_path, question, options, decision)

    return asyncio.run(
        _run_single_async(
            image_path=image_path,
            question=question,
            options=options,
            decision=decision,
            metadata=metadata or {},
            cfg=cfg,
        )
    )


async def _run_single_async(
    *,
    image_path: str,
    question: str,
    options: Dict[str, str],
    decision: CoordinatorDecision,
    metadata: Dict[str, Any],
    cfg: FetUSConfig,
) -> WorkflowResult:
    # IMPORTANT: export env vars BEFORE the lazy import. The CV tool
    # wrappers freeze their default Python paths from
    # ``FETALAGENT_*_PYTHON`` at TOOL_CONFIG class-definition time, which
    # is triggered by ``from ..specific import ...``. Setting them
    # afterwards is too late.
    cfg.export_environment()

    # Import the heavy specific-VQA pipeline lazily so that simply
    # loading this module (e.g. for routing tests) does not require
    # autogen / torch / openai to be installed.
    from ..specific import (
        TASK_SPECS,
        VQASample,
        build_agents,
        build_model_client,
        solve_one_sample,
    )

    task_name = decision.task_type.value
    if task_name not in TASK_SPECS:
        raise ValueError(
            f"task_type={task_name} is not a registered specific task. "
            f"Valid options: {list(TASK_SPECS.keys())}"
        )

    task_spec = TASK_SPECS[task_name]
    # Subprocess-based CV tools run with their own cwd, so a relative
    # ``image_dir`` would break ``os.path.exists`` inside them.
    image_dir = os.path.abspath(os.path.dirname(image_path) or ".")
    image_id = os.path.basename(image_path)
    pixel_size = metadata.get("pixel_size") if metadata else None

    sample = VQASample(
        image_id=image_id,
        question=question,
        options=_options_to_list(options),
        answer="",
        image_dir=image_dir,
        task_name=task_name,
        pixel_size=pixel_size,
    )

    model_client = build_model_client()
    agents = build_agents(model_client, task_spec)
    sample_result = await solve_one_sample(sample, agents)

    return _convert_sample_result(sample_result, decision, image_path, question, options)


def _convert_sample_result(
    sr: Any,
    decision: CoordinatorDecision,
    image_path: str,
    question: str,
    options: Dict[str, str],
) -> WorkflowResult:
    votes = [
        {
            "agent_name": v.agent_name,
            "answer_letter": v.answer_letter,
            "answer_text": v.answer_text,
            "raw_output": v.raw_output,
        }
        for v in (sr.votes or [])
    ]
    tool_decision = sr.tool_decision
    tool_outputs: Dict[str, Any] = {
        "used_tool": getattr(tool_decision, "used_tool", False),
        "tool_name": getattr(tool_decision, "tool_name", None),
        "tool_answer_letter": getattr(tool_decision, "tool_answer_letter", None),
        "tool_answer_text": getattr(tool_decision, "tool_answer_text", None),
        "tool_detail": getattr(tool_decision, "tool_detail", {}),
    }

    ks = sr.knowledge_store
    rag_snippets: List[str] = []
    evidence_bank: Dict[str, Any] = {}
    if ks is not None:
        rag_snippets = list(getattr(ks, "rag_knowledge", []) or [])
        evidence_bank = {
            "voter_reasons": getattr(ks, "voter_reasons", []),
            "tool_summary": getattr(ks, "tool_summary", {}),
            "analysis_text": getattr(ks, "analysis_text", ""),
        }

    final_letter = sr.final_answer
    final_text = options.get(final_letter or "", "")

    report_text = sr.report or ""
    report = _parse_report_sections(report_text)

    return WorkflowResult(
        query_type=QueryType.SPECIFIC,
        task_type=decision.task_type,
        input_path=image_path,
        question=question,
        coordinator={
            "confidence": decision.confidence,
            "route_reason": decision.route_reason,
        },
        options=options,
        final_answer=final_letter,
        final_option_text=final_text,
        route=sr.route or "",
        voters=votes,
        tool_outputs=tool_outputs,
        rag_snippets=rag_snippets,
        evidence_bank=evidence_bank,
        report=report,
        summary=report_text,
    )


_REPORT_SECTION_RE = re.compile(
    r"^\s*(findings?|impressions?|notes?)\s*[:：]\s*",
    re.IGNORECASE,
)
_REPORT_SECTION_KEYS = {
    "finding": "findings", "findings": "findings",
    "impression": "impression", "impressions": "impression",
    "note": "note", "notes": "note",
}


def _parse_report_sections(text: str) -> Report:
    """Split the ``Findings: ... Impressions: ... Note: ...`` report blob.

    The specific-VQA report generator is prompted to emit exactly those
    three section headers. When the model complies we extract each one;
    otherwise the whole string falls into ``findings`` so nothing is
    lost.
    """
    if not text:
        return Report()
    sections: Dict[str, List[str]] = {"findings": [], "impression": [], "note": []}
    current = "findings"
    seen_header = False
    for line in text.splitlines():
        m = _REPORT_SECTION_RE.match(line)
        if m:
            current = _REPORT_SECTION_KEYS[m.group(1).lower()]
            remainder = line[m.end():].strip()
            if remainder:
                sections[current].append(remainder)
            seen_header = True
            continue
        sections[current].append(line)
    if not seen_header:
        return Report(findings=text.strip())
    return Report(
        findings="\n".join(sections["findings"]).strip(),
        impression="\n".join(sections["impression"]).strip(),
        note="\n".join(sections["note"]).strip(),
    )


def _mock_result(
    image_path: str,
    question: str,
    options: Dict[str, str],
    decision: CoordinatorDecision,
) -> WorkflowResult:
    first_letter = sorted(options.keys())[0] if options else "A"
    return WorkflowResult(
        query_type=QueryType.SPECIFIC,
        task_type=decision.task_type,
        input_path=image_path,
        question=question,
        coordinator={
            "confidence": decision.confidence,
            "route_reason": decision.route_reason,
        },
        options=options,
        final_answer=first_letter,
        final_option_text=options.get(first_letter, ""),
        route="checker_decision",
        voters=[
            {"agent_name": f"voter_{i}", "answer_letter": first_letter, "answer_text": "MOCK", "raw_output": "MOCK"}
            for i in range(1, 6)
        ],
        tool_outputs={
            "used_tool": True,
            "tool_name": f"mock::{decision.task_type.value}",
            "tool_answer_letter": first_letter,
            "tool_answer_text": options.get(first_letter, ""),
            "tool_detail": {"mock": True},
        },
        rag_snippets=["[dry_run] no RAG retrieval performed"],
        evidence_bank={
            "voter_reasons": [],
            "tool_summary": {"mock": True},
            "analysis_text": "[dry_run] analyst stub",
        },
        report=Report(
            findings="[dry_run] No real findings; this is a control-flow smoke test.",
            impression=f"[dry_run] task_type={decision.task_type.value}",
            note="Set --dry_run=false and configure paths to obtain real results.",
        ),
        summary=f"[dry_run] {decision.task_type.value} mock answer: {first_letter}",
        dry_run=True,
    )


def _options_input_to_dict(options: Any) -> Dict[str, str]:
    if isinstance(options, dict):
        return {str(k).upper(): str(v) for k, v in options.items()}
    if isinstance(options, list):
        out: Dict[str, str] = {}
        for item in options:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if len(text) >= 2 and text[0].isalpha() and text[1] in ".:":
                out[text[0].upper()] = text[2:].strip()
            else:
                letter = chr(ord("A") + len(out))
                out[letter] = text
        return out
    return {}
