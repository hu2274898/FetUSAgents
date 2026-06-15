"""Workflow adapters.

Each submodule glues the Coordinator decision to one of the two
internal pipelines (:mod:`fetusagents.specific` for VQA,
:mod:`fetusagents.general` for open-ended captioning / video
summaries) and produces a :class:`fetusagents.schemas.WorkflowResult`
regardless of which path was taken. ``dry_run`` short-circuits the
heavy work and returns a deterministic mock so CLI smoke tests can run
without GPUs.
"""
