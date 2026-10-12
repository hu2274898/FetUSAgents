"""Coordinator agent: query routing + task allocation in one pass.

By default the Coordinator uses a vision-capable LLM allocator when an API
key is configured, with strict local enum validation and deterministic rules
as a fallback.  ``routing_strategy="rules"`` preserves fully offline,
reproducible routing for tests and ablations.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from .llm_routing import LLMRouteSelection, infer_route_with_llm
from .routing import (
    has_explicit_option_block,
    infer_specific_task,
    is_video_input,
    looks_like_open_ended,
    parse_options,
)
from .schemas import CoordinatorDecision, QueryType, TaskType


logger = logging.getLogger(__name__)


class Coordinator:
    """Two-stage router used as the top of every workflow.

    Stage A decides ``query_type`` (specific vs general); stage B picks
    the concrete ``task_type``. Either stage may be short-circuited by
    explicit user overrides (``mode``, ``task_name``).
    """

    def __init__(self, llm_allocator: Any = None) -> None:
        # Dependency injection keeps closed-set validation and fallback logic
        # unit-testable without an API call.
        self._llm_allocator = llm_allocator or infer_route_with_llm
        self._custom_llm_allocator = llm_allocator is not None

    def route(
        self,
        query: str,
        input_path: str = "",
        options: Optional[Dict[str, str]] = None,
        mode: str = "auto",
        task_name: Optional[str] = None,
        hints: Optional[Dict[str, Any]] = None,
        routing_strategy: str = "auto",
    ) -> CoordinatorDecision:
        hints = dict(hints or {})
        if options is None or not options:
            options = parse_options(query)
        else:
            options = {str(k).upper(): str(v) for k, v in options.items()}

        if mode not in ("auto", "specific", "general"):
            raise ValueError(f"Unknown --mode value: {mode!r}; expected auto/specific/general")
        if routing_strategy not in ("auto", "llm", "rules"):
            raise ValueError(
                f"Unknown routing_strategy={routing_strategy!r}; expected auto/llm/rules"
            )

        if task_name:
            task_id = task_name.strip()
            task_type = self._task_from_string(task_id)
            query_type = (
                QueryType.SPECIFIC
                if task_type in TaskType.specific_values()
                else QueryType.GENERAL
            )
            if mode != "auto" and mode != query_type.value:
                raise ValueError(
                    f"task_name={task_id!r} belongs to mode={query_type.value!r}, "
                    f"but mode={mode!r} was requested"
                )
            return CoordinatorDecision(
                query_type=query_type,
                task_type=task_type,
                confidence=1.0,
                route_reason=f"task_name override → {task_id}; workflow inferred from task registry",
                parsed_options=options,
                overrides={
                    "task_name": task_id,
                    "mode": mode,
                    "routing_strategy": "override",
                },
            )

        allowed_query_types = self._allowed_query_types(mode)
        llm_failure_reason = ""
        should_try_llm = routing_strategy in ("auto", "llm") and (
            self._custom_llm_allocator or bool(os.environ.get("OPENAI_API_KEY"))
        )
        if should_try_llm:
            try:
                selection: LLMRouteSelection = self._llm_allocator(
                    query=query,
                    input_path=input_path,
                    options=options,
                    allowed_query_types=allowed_query_types,
                )
                return CoordinatorDecision(
                    query_type=selection.query_type,
                    task_type=selection.task_type,
                    confidence=round(selection.confidence, 3),
                    route_reason=f"LLM allocator: {selection.reason}",
                    parsed_options=options,
                    overrides={"mode": mode, "routing_strategy": "llm"},
                )
            except Exception as exc:
                if routing_strategy == "llm":
                    raise RuntimeError(f"LLM allocator failed: {exc}") from exc
                llm_failure_reason = f"LLM allocator failed ({type(exc).__name__}); "
                logger.warning("LLM allocator failed; using rule fallback: %s", exc)
        elif routing_strategy == "llm":
            raise RuntimeError("OPENAI_API_KEY is required when routing_strategy='llm'")
        elif routing_strategy == "auto":
            llm_failure_reason = "LLM allocator unavailable (no API key); "

        decision = self._route_with_rules(query, input_path, options, mode)
        if llm_failure_reason:
            decision.route_reason = llm_failure_reason + "rule fallback: " + decision.route_reason
        decision.overrides["routing_strategy"] = "rules"
        return decision

    def _route_with_rules(
        self,
        query: str,
        input_path: str,
        options: Dict[str, str],
        mode: str,
    ) -> CoordinatorDecision:
        query_type, qreason = self._decide_query_type(query, input_path, options, mode)

        if query_type is QueryType.SPECIFIC:
            task_id, task_conf, task_reason = infer_specific_task(query, options)
            try:
                task_enum = TaskType(task_id)
            except ValueError as exc:
                raise ValueError(
                    f"Coordinator inferred unknown specific task_id={task_id!r}. "
                    f"Valid: {[t.value for t in TaskType.specific_values()]}"
                ) from exc
            confidence = round(0.5 * 1.0 + 0.5 * task_conf, 3)
            return CoordinatorDecision(
                query_type=query_type,
                task_type=task_enum,
                confidence=confidence,
                route_reason=f"{qreason}; {task_reason}",
                parsed_options=options,
                overrides={"mode": mode},
            )

        general_task, greason = self._decide_general_task(query, input_path)
        return CoordinatorDecision(
            query_type=query_type,
            task_type=general_task,
            confidence=0.8,
            route_reason=f"{qreason}; {greason}",
            parsed_options=options,
            overrides={"mode": mode},
        )

    @staticmethod
    def _allowed_query_types(mode: str) -> tuple[QueryType, ...]:
        if mode == "specific":
            return (QueryType.SPECIFIC,)
        if mode == "general":
            return (QueryType.GENERAL,)
        return (QueryType.SPECIFIC, QueryType.GENERAL)

    @staticmethod
    def _decide_query_type(
        query: str,
        input_path: str,
        options: Dict[str, str],
        mode: str,
    ) -> tuple[QueryType, str]:
        if mode and mode != "auto":
            if mode == "specific":
                return QueryType.SPECIFIC, "user override mode=specific"
            if mode == "general":
                return QueryType.GENERAL, "user override mode=general"
            raise ValueError(f"Unknown --mode value: {mode!r}; expected auto/specific/general")

        if has_explicit_option_block(query) or (options and len(options) >= 2):
            return (
                QueryType.SPECIFIC,
                f"detected {len(options)} multiple-choice options",
            )

        if looks_like_open_ended(query):
            return QueryType.GENERAL, "open-ended trigger phrase in query"

        if is_video_input(input_path) and not options:
            return QueryType.GENERAL, "input is a video / frame folder and no MC options"

        return QueryType.GENERAL, "no MC options detected; treating as open-ended"

    @staticmethod
    def _decide_general_task(query: str, input_path: str) -> tuple[TaskType, str]:
        q = (query or "").lower()
        if is_video_input(input_path) or "video" in q or "continuous screenshots" in q:
            return TaskType.VIDEO_SUMMARY, "video / frame folder → video_summary"
        return TaskType.IMAGE_CAPTION, "single image / open-ended request → image_caption"

    @staticmethod
    def _task_from_string(task_id: str) -> TaskType:
        try:
            return TaskType(task_id)
        except ValueError as exc:
            valid = [t.value for t in TaskType]
            raise ValueError(
                f"Unknown task_name={task_id!r}. Valid options: {valid}"
            ) from exc


def route(
    query: str,
    input_path: str = "",
    **kwargs: Any,
) -> CoordinatorDecision:
    """Module-level convenience wrapper around :class:`Coordinator`."""
    return Coordinator().route(query=query, input_path=input_path, **kwargs)
