from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fetusagents.coordinator import Coordinator
from fetusagents.llm_routing import LLMRouteSelection, parse_allocator_response
from fetusagents.schemas import QueryType, TaskType


class LLMRoutingTests(unittest.TestCase):
    def test_strict_json_response_is_accepted(self) -> None:
        decision = parse_allocator_response(
            '{"query_type":"specific","task_type":"brain_subplane_binary",'
            '"confidence":0.91,"reason":"The question asks for a yes/no brain-plane check."}',
            (QueryType.SPECIFIC, QueryType.GENERAL),
        )
        self.assertEqual(decision.query_type, QueryType.SPECIFIC)
        self.assertEqual(decision.task_type, TaskType.BRAIN_SUBPLANE_BINARY)
        self.assertEqual(decision.confidence, 0.91)

    def test_invented_task_label_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the registered enums"):
            parse_allocator_response(
                '{"query_type":"specific","task_type":"brain_magic",'
                '"confidence":0.9,"reason":"test"}',
                (QueryType.SPECIFIC, QueryType.GENERAL),
            )

    def test_task_must_match_selected_workflow(self) -> None:
        with self.assertRaisesRegex(ValueError, "incompatible"):
            parse_allocator_response(
                '{"query_type":"general","task_type":"aop_binary",'
                '"confidence":0.9,"reason":"test"}',
                (QueryType.SPECIFIC, QueryType.GENERAL),
            )

    def test_coordinator_uses_injected_llm_allocator(self) -> None:
        seen = {}

        def fake_allocator(**kwargs):
            seen.update(kwargs)
            return LLMRouteSelection(
                query_type=QueryType.SPECIFIC,
                task_type=TaskType.AOP_BINARY,
                confidence=0.94,
                reason="Candidate options ask for the AoP threshold outcome.",
            )

        decision = Coordinator(llm_allocator=fake_allocator).route(
            query="Which delivery outcome is likely? (A) vaginal (B) cesarean",
            input_path="scan.png",
            routing_strategy="llm",
        )
        self.assertEqual(decision.query_type, QueryType.SPECIFIC)
        self.assertEqual(decision.task_type, TaskType.AOP_BINARY)
        self.assertEqual(decision.overrides["routing_strategy"], "llm")
        self.assertIn(QueryType.SPECIFIC, seen["allowed_query_types"])

    def test_forced_mode_limits_llm_choices(self) -> None:
        def fake_allocator(**kwargs):
            self.assertEqual(kwargs["allowed_query_types"], (QueryType.GENERAL,))
            return LLMRouteSelection(
                query_type=QueryType.GENERAL,
                task_type=TaskType.IMAGE_CAPTION,
                confidence=0.88,
                reason="The caller forced the general workflow.",
            )

        decision = Coordinator(llm_allocator=fake_allocator).route(
            query="Describe the scan",
            mode="general",
            routing_strategy="llm",
        )
        self.assertEqual(decision.query_type, QueryType.GENERAL)

    def test_auto_falls_back_to_rules_without_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            decision = Coordinator().route(
                query="Is this trans-thalamic? Yes / No",
                routing_strategy="auto",
            )
        self.assertEqual(decision.task_type, TaskType.BRAIN_SUBPLANE_BINARY)
        self.assertEqual(decision.overrides["routing_strategy"], "rules")
        self.assertIn("no API key", decision.route_reason)

    def test_llm_strategy_requires_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                Coordinator().route(query="Describe this image", routing_strategy="llm")

    def test_task_override_also_sets_compatible_workflow(self) -> None:
        decision = Coordinator().route(
            query="Estimate AoP",
            task_name="aop_binary",
            routing_strategy="rules",
        )
        self.assertEqual(decision.query_type, QueryType.SPECIFIC)
        self.assertEqual(decision.task_type, TaskType.AOP_BINARY)


if __name__ == "__main__":
    unittest.main()
