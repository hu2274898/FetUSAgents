"""Specific-VQA expert agents.

Each VQA task is backed by a :class:`SpecificExpert` whose runner takes
a :class:`VQASample` and returns a :class:`ToolDecision`. Experts are
registered here once and looked up by name from :data:`TASK_SPECS`.

The runner signature differs from :class:`fetusagents.general.GeneralExpert`
(which takes a ``case_dir``), but both kinds of expert share the
:class:`fetusagents.core.agent_base.ToolAgent` base type, so registry
consumers can treat them uniformly when reporting ``name`` /
``description`` / ``tools``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from ..core.agent_base import ToolAgent
from .tools_for_sample import (
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
from .types import ToolDecision, VQASample


@dataclass
class SpecificExpert(ToolAgent):
    """Specific-VQA expert. Runner is synchronous and takes a ``VQASample``.

    ``runner`` is typed only loosely on the parent (``Callable``); on
    this subclass we expect ``(VQASample) -> ToolDecision``. The
    :meth:`run` method is the canonical entry point — :mod:`pipeline`
    calls it from a thread pool so it can race with the LLM voters.
    """

    runner: Optional[Callable[[VQASample], ToolDecision]] = None

    def run(self, sample: VQASample) -> ToolDecision:
        if self.runner is None:
            raise RuntimeError(f"SpecificExpert({self.name}) has no runner bound")
        return self.runner(sample)


def build_specific_experts() -> Dict[str, SpecificExpert]:
    """Build the ten specific-VQA expert agents."""
    return {
        "plane_classification": SpecificExpert(
            name="plane_classification",
            description="Pick the standard scan plane (abdomen / femur / brain / thorax) from a single image.",
            tools=["Plane-FetalCLIP", "Plane-FU-LoRA"],
            runner=run_plane_tool_for_sample,
        ),
        "plane_binary": SpecificExpert(
            name="plane_binary",
            description="Yes/No check that the image matches the plane named in the question.",
            tools=["Plane-FetalCLIP", "Plane-FU-LoRA"],
            runner=run_plane_binary_tool_for_sample,
        ),
        "brain_subplane": SpecificExpert(
            name="brain_subplane",
            description="Pick the brain subplane (trans-cerebellum / trans-thalamic / trans-ventricular).",
            tools=["BrainSubplane-FetalCLIP", "BrainSubplane-ResNet", "BrainSubplane-ViT"],
            runner=run_brain_subplane_tool_for_sample,
        ),
        "brain_subplane_binary": SpecificExpert(
            name="brain_subplane_binary",
            description="Yes/No check that the image matches the brain subplane named in the question.",
            tools=["BrainSubplane-FetalCLIP", "BrainSubplane-ResNet", "BrainSubplane-ViT"],
            runner=run_brain_subplane_binary_tool_for_sample,
        ),
        "ga_trimester_binary": SpecificExpert(
            name="ga_trimester_binary",
            description="Yes/No on whether the image's gestational age is within the trimester asked about.",
            tools=["GA-RadImageNet", "GA-FetalCLIP", "GA-ConvNeXt"],
            runner=run_ga_trimester_binary_tool_for_sample,
        ),
        "ga_trimester_multi": SpecificExpert(
            name="ga_trimester_multi",
            description="Choose the most likely trimester (1 / 2 / 3) from a single image.",
            tools=["GA-RadImageNet", "GA-FetalCLIP", "GA-ConvNeXt"],
            runner=run_ga_trimester_multi_tool_for_sample,
        ),
        "hc_estimation_pixel": SpecificExpert(
            name="hc_estimation_pixel",
            description="Pick the option whose pixel-space HC range matches the head shown.",
            tools=["CSM-HC", "HC-nnUNet"],
            runner=run_hc_pixel_tool_for_sample,
        ),
        "aop_binary": SpecificExpert(
            name="aop_binary",
            description="Yes/No on whether the AoP is >= 120 degrees.",
            tools=["AoP-SAM", "USFM-AoP", "UperNet-AoP"],
            runner=run_aop_binary_tool_for_sample,
        ),
        "ac_estimation_pixel": SpecificExpert(
            name="ac_estimation_pixel",
            description="Pick the option whose pixel-space AC range matches the abdomen shown.",
            tools=["AbdomenSeg-FetalCLIP+SAMUS", "AbdomenSeg-FetalCLIP"],
            runner=run_ac_pixel_tool_for_sample,
        ),
        "stomach_volume_estimation": SpecificExpert(
            name="stomach_volume_estimation",
            description="Pick the option whose pixel-space stomach area matches the stomach shown.",
            tools=["StomachSeg-FetalCLIP", "StomachSeg-FetalCLIP+SAMUS", "StomachSeg-nnUNet"],
            runner=run_stomach_volume_pixel_tool_for_sample,
        ),
    }


# Module-level registry — built once at import time so TASK_SPECS can
# reference instances by key without rebuilding on every lookup.
SPECIFIC_EXPERTS: Dict[str, SpecificExpert] = build_specific_experts()
