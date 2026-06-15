"""Knowledge collection: option maps + assembly of :class:`KnowledgeStore`
from voter outputs, tool decisions, and RAG snippets.

This module sits between ``rag`` (RAG retrieval) and the upper-level
``pipeline`` (which actually orchestrates the agents).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .rag import _try_get_rag, build_rag_query
from .types import AgentVote, KnowledgeStore, ToolDecision, VQASample


# Per-task option ↔ text mappings (used by parsers and the LLM prompts)
PLANE_OPTION_TO_TEXT = {
    "A": "Fetal abdomen",
    "B": "Fetal femur",
    "C": "Fetal brain",
    "D": "Fetal thorax",
}

BRAIN_OPTION_TO_TEXT = {
    "A": "Trans-cerebellum",
    "B": "Trans-thalamic",
    "C": "Trans-ventricular",
}

PLANE_CANONICAL_TO_OPTION = {
    "abdomen": "A",
    "femur": "B",
    "brain": "C",
    "thorax": "D",
}

BRAIN_CANONICAL_TO_OPTION = {
    "trans-cerebellum": "A",
    "trans-thalamic": "B",
    "trans-ventricular": "C",
}

YESNO_OPTION_TO_TEXT = {
    "A": "Yes",
    "B": "No",
}

TRIMESTER_OPTION_TO_TEXT = {
    "A": "First Trimester",
    "B": "Second Trimester",
    "C": "Third Trimester",
}


# Voter reasoning extraction
def extract_reason_from_voter(raw_output: str) -> str:
    """Pull the 'Reason: ...' clause out of a voter's raw LLM output."""
    if not raw_output:
        return "(no reasoning provided)"
    m = re.search(
        r"Reason\s*[:：]\s*(.+?)(?=\n\s*Final\s+Answer|$)",
        raw_output,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    # fallback: first 500 chars
    return raw_output.strip()[:500]


# Knowledge Store assembly
def collect_knowledge(
    sample: VQASample,
    votes: List[AgentVote],
    tool_decision: ToolDecision,
    analysis_text: str,
    final_answer: Optional[str],
    option_to_text: Dict[str, str],
) -> KnowledgeStore:
    """Combine voter outputs, tool decision and RAG snippets into a
    :class:`KnowledgeStore` that the report-generation agent consumes."""

    # ── voter reasons ──
    voter_reasons = []
    for v in votes:
        voter_reasons.append({
            "agent_name": v.agent_name,
            "reason": extract_reason_from_voter(v.raw_output),
            "answer_letter": v.answer_letter or "N/A",
            "answer_text": v.answer_text or "N/A",
        })

    # ── tool summary ──
    tool_summary: Dict[str, Any] = {
        "used_tool": tool_decision.used_tool,
        "tool_name": tool_decision.tool_name,
        "tool_answer_letter": tool_decision.tool_answer_letter,
        "tool_answer_text": tool_decision.tool_answer_text,
    }
    detail = tool_decision.tool_detail or {}
    for key in (
        "pred_pixel", "recommended_total_weeks", "recommended_aop_deg",
        "predicted_trimester", "tool1_plane", "tool2_plane",
        "tool1_label", "tool2_label", "tool3_label",
        "final_label", "predicted_label", "target_label",
        "option_map", "note", "threshold_deg",
    ):
        if key in detail:
            tool_summary[key] = detail[key]

    # ── RAG ──
    rag_knowledge: List[str] = []
    rag = _try_get_rag()
    if rag is not None:
        query = build_rag_query(sample, analysis_text)
        rag_knowledge = rag.retrieve(query, k=5)
        print(f"  [RAG] Retrieved {len(rag_knowledge)} knowledge chunks")

    return KnowledgeStore(
        voter_reasons=voter_reasons,
        tool_summary=tool_summary,
        rag_knowledge=rag_knowledge,
        analysis_text=analysis_text,
        question=sample.question,
        options=sample.options,
        final_answer=final_answer,
        final_answer_text=option_to_text.get(final_answer) if final_answer else None,
        task_name=sample.task_name,
        image_id=sample.image_id,
    )
