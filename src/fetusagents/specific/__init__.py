"""``fetus_specific`` — the specific VQA workflow library.

Originally a single ~2500-line ``fetalagent.py``. Split here into focused
modules; the legacy ``fetalagent.py`` shim re-exports everything below.

Layout::

    schemas.py            Dataclasses (VQASample, SampleResult, ...) + checkpoint helpers
    rag.py                RAGRetriever + build_rag_query
    knowledge.py          Option maps + collect_knowledge + extract_reason_from_voter
    parsers.py            Answer parsers, normalisers, dataset loader, helpers
    tools_for_sample.py   load_tools_from_demo + 10 run_*_tool_for_sample
    experts.py            SpecificExpert + build_specific_experts (10 expert agents)
    pipeline.py           build_agents + solve_one_sample + TASK_SPECS
"""
from __future__ import annotations

# Schemas / data classes
from .types import (
    AgentVote,
    DEFAULT_PIXEL_SIZE_MM,
    KnowledgeStore,
    SAVE_EVERY,
    SampleResult,
    TaskSpec,
    ToolDecision,
    VQASample,
    load_checkpoint,
    save_checkpoint,
)

# RAG
from .rag import (
    HAS_RAG,
    RAGRetriever,
    _try_get_rag,
    build_rag_query,
)

# Knowledge / option maps
from .knowledge import (
    BRAIN_CANONICAL_TO_OPTION,
    BRAIN_OPTION_TO_TEXT,
    PLANE_CANONICAL_TO_OPTION,
    PLANE_OPTION_TO_TEXT,
    TRIMESTER_OPTION_TO_TEXT,
    YESNO_OPTION_TO_TEXT,
    collect_knowledge,
    extract_reason_from_voter,
)

# Parsers / helpers
from .parsers import (
    _majority_vote_masks,
    _stomach_area_pixel_from_mask_array,
    count_votes,
    ensure_single_image_case_dir,
    extract_target_brain_subplane_from_question,
    extract_target_plane_from_question,
    extract_target_trimester_from_question,
    get_consensus_answer,
    load_vqa_dataset,
    make_mm_message,
    normalize_brain_subplane_label,
    normalize_plane_label,
    normalize_space,
    parse_abcd_answer,
    parse_brain_subplane_answer,
    parse_numeric_options,
    parse_plane_answer,
    parse_trimester_multi_answer,
    parse_yesno_answer,
    pick_closest_option_letter,
    trimester_from_total_weeks,
)

# Tool-for-sample adapters
from .tools_for_sample import (
    load_tools_from_demo,
    run_ac_pixel_tool_for_sample,
    run_aop_binary_tool_for_sample,
    run_brain_subplane_binary_tool_for_sample,
    run_brain_subplane_tool_for_sample,
    run_ga_trimester_binary_tool_for_sample,
    run_ga_trimester_multi_tool_for_sample,
    run_hc_pixel_tool_for_sample,
    run_plane_binary_tool_for_sample,
    run_plane_tool_for_sample,
    run_stomach_volume_pixel_tool_for_sample,
)

# Pipeline (TASK_SPECS, agents, solve_one_sample, ...)
from .pipeline import (
    TASK_SPECS,
    build_agents,
    build_model_client,
    run_text_agent,
    solve_one_sample,
)

# Expert agents — uniform registry shared with the general workflow's
# GeneralExpert instances. ``ToolAgent`` is the common base type.
from ..core.agent_base import ToolAgent
from .experts import SPECIFIC_EXPERTS, SpecificExpert, build_specific_experts


__all__ = [
    # schemas
    "AgentVote", "DEFAULT_PIXEL_SIZE_MM", "KnowledgeStore", "SAVE_EVERY",
    "SampleResult", "TaskSpec", "ToolDecision", "VQASample",
    "load_checkpoint", "save_checkpoint",
    # rag
    "HAS_RAG", "RAGRetriever", "_try_get_rag", "build_rag_query",
    # knowledge
    "BRAIN_CANONICAL_TO_OPTION", "BRAIN_OPTION_TO_TEXT",
    "PLANE_CANONICAL_TO_OPTION", "PLANE_OPTION_TO_TEXT",
    "TRIMESTER_OPTION_TO_TEXT", "YESNO_OPTION_TO_TEXT",
    "collect_knowledge", "extract_reason_from_voter",
    # parsers
    "_majority_vote_masks", "_stomach_area_pixel_from_mask_array",
    "count_votes", "ensure_single_image_case_dir",
    "extract_target_brain_subplane_from_question",
    "extract_target_plane_from_question",
    "extract_target_trimester_from_question",
    "get_consensus_answer", "load_vqa_dataset", "make_mm_message",
    "normalize_brain_subplane_label", "normalize_plane_label", "normalize_space",
    "parse_abcd_answer", "parse_brain_subplane_answer", "parse_numeric_options",
    "parse_plane_answer", "parse_trimester_multi_answer", "parse_yesno_answer",
    "pick_closest_option_letter", "trimester_from_total_weeks",
    # tool runners
    "load_tools_from_demo",
    "run_ac_pixel_tool_for_sample", "run_aop_binary_tool_for_sample",
    "run_brain_subplane_binary_tool_for_sample", "run_brain_subplane_tool_for_sample",
    "run_ga_trimester_binary_tool_for_sample", "run_ga_trimester_multi_tool_for_sample",
    "run_hc_pixel_tool_for_sample", "run_plane_binary_tool_for_sample",
    "run_plane_tool_for_sample", "run_stomach_volume_pixel_tool_for_sample",
    # expert agents
    "SPECIFIC_EXPERTS", "SpecificExpert", "ToolAgent", "build_specific_experts",
    # pipeline
    "TASK_SPECS", "build_agents", "build_model_client",
    "run_text_agent", "solve_one_sample",
]
