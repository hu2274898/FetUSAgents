"""Data classes and lightweight checkpoint helpers used by the specific
VQA pipeline.

Annotations use PEP 563 (``from __future__ import annotations``) so the
dataclasses can refer to each other in either order without circular
issues at class-body evaluation time.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.agent_base import ToolAgent


# Checkpoint helpers
SAVE_EVERY = 1  # save every N samples (recommended 1, since each is slow)


def load_checkpoint(checkpoint_path: str) -> Tuple[List[dict], set]:
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        done_ids = {r["image_id"] for r in saved}
        print(f"[RESUME] Loaded checkpoint: {len(done_ids)} samples already done")
        return saved, done_ids
    return [], set()


def save_checkpoint(results_serializable: List[dict], checkpoint_path: str) -> None:
    tmp = checkpoint_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results_serializable, f, ensure_ascii=False, indent=2)
    os.replace(tmp, checkpoint_path)  # atomic replace, safe against partial writes


# Defaults
DEFAULT_PIXEL_SIZE_MM = 0.15


# Dataclasses
@dataclass
class AgentVote:
    agent_name: str
    answer_letter: Optional[str]
    answer_text: Optional[str]
    raw_output: str


@dataclass
class ToolDecision:
    used_tool: bool
    tool_name: Optional[str]
    tool_answer_letter: Optional[str]
    tool_answer_text: Optional[str]
    tool_detail: Dict[str, Any]


@dataclass
class TaskSpec:
    name: str
    vqa_json: str
    default_image_dir: str
    option_to_text: Dict[str, str]
    allowed_letters: Tuple[str, ...]
    answer_parser: Callable[[str], Optional[str]]
    expert: Optional["ToolAgent"] = None


@dataclass
class VQASample:
    image_id: str
    question: str
    options: List[str]
    answer: str
    image_dir: str
    task_name: str
    pixel_size: Optional[float] = None

    @property
    def image_filename(self) -> str:
        # If image_id already ends with an image extension, use as-is.
        if str(self.image_id).lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")):
            return self.image_id
        return f"{self.image_id}.png"

    @property
    def image_path(self) -> str:
        return os.path.join(self.image_dir, self.image_filename)


@dataclass
class SampleResult:
    image_id: str
    gt_answer: str
    task_type: str
    allocator_text: str
    analysis_text: str
    votes: List[AgentVote]
    vote_count: Dict[str, int]
    consensus_answer: Optional[str]
    final_answer: Optional[str]
    route: str
    checker_text: str
    tool_decision: ToolDecision
    correct: bool
    report: str = ""
    knowledge_store: Optional[KnowledgeStore] = None


@dataclass
class KnowledgeStore:
    """Aggregates voter reasoning, tool outputs and RAG snippets for the
    report-generation agent."""
    voter_reasons: List[Dict[str, str]]
    tool_summary: Dict[str, Any]
    rag_knowledge: List[str]
    analysis_text: str
    question: str
    options: List[str]
    final_answer: Optional[str]
    final_answer_text: Optional[str]
    task_name: str
    image_id: str
