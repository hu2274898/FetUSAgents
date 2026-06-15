"""Coordinator agent: query routing + task allocation in one pass.

The Coordinator never touches checkpoints, GPUs, or LLM APIs. It is a pure
function from ``(query, input_path, options, hints)`` → :class:`CoordinatorDecision`,
which keeps it trivially testable. The downstream workflows are responsible
for actually executing tools.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .routing import (
    has_explicit_option_block,
    infer_specific_task,
    is_video_input,
    looks_like_open_ended,
    parse_options,
)
from .schemas import CoordinatorDecision, QueryType, TaskType


class Coordinator:
    """Two-stage router used as the top of every workflow.

    Stage A decides ``query_type`` (specific vs general); stage B picks
    the concrete ``task_type``. Either stage may be short-circuited by
    explicit user overrides (``mode``, ``task_name``).
    """

    def route(
        self,
        query: str,
        input_path: str = "",
        options: Optional[Dict[str, str]] = None,
        mode: str = "auto",
        task_name: Optional[str] = None,
        hints: Optional[Dict[str, Any]] = None,
    ) -> CoordinatorDecision:
        hints = dict(hints or {})
        if options is None or not options:
            options = parse_options(query)
        else:
            options = {str(k).upper(): str(v) for k, v in options.items()}

        query_type, qreason = self._decide_query_type(query, input_path, options, mode)

        if task_name:
            task_id = task_name.strip()
            return CoordinatorDecision(
                query_type=query_type,
                task_type=self._task_from_string(task_id),
                confidence=1.0,
                route_reason=f"task_name override → {task_id}; {qreason}",
                parsed_options=options,
                overrides={"task_name": task_id, "mode": mode},
            )

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
