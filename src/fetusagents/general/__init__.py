"""``fetusagents.general`` — the general (open-ended) workflow.

This package mirrors :mod:`fetusagents.specific`: it contains the
pipeline and agents used when the coordinator routes a query to the
open-ended caption / video-summary path. Shared infrastructure (LLM
client, biometry helpers, tool runners, the :class:`ToolAgent` base
type) lives in :mod:`fetusagents.core`.

Layout::

    pipeline.py        build_agents + orchestrate + video summary + report template
    experts.py         GeneralExpert + build_general_experts (seven expert agents)
    expert_runners.py  run_*_expert policy implementations
"""
from __future__ import annotations

# Public surface — keeps the same names that used to live under
# ``fetusagents.core`` so existing call paths continue to work.
from .pipeline import (
    _build_structured_text_summary,
    _enforce_per_image_json,
    build_agents,
    extract_agent_text,
    orchestrate,
    parse_forwarding_and_rephrased,
    run_video_summary_workflow,
)
from .expert_runners import (
    run_abdomen_seg_expert,
    run_aop_expert,
    run_brain_subplane_expert,
    run_ga_expert,
    run_hc_expert,
    run_plane_expert,
    run_stomach_seg_expert,
)
from .experts import GeneralExpert, build_general_experts


__all__ = [
    # pipeline
    "_build_structured_text_summary", "_enforce_per_image_json", "build_agents",
    "extract_agent_text", "orchestrate", "parse_forwarding_and_rephrased",
    "run_video_summary_workflow",
    # expert_runners
    "run_abdomen_seg_expert", "run_aop_expert", "run_brain_subplane_expert",
    "run_ga_expert", "run_hc_expert", "run_plane_expert", "run_stomach_seg_expert",
    # experts
    "GeneralExpert", "build_general_experts",
]
