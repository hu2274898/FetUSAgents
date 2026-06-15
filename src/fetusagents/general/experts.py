"""Expert agents — the general-workflow side.

Each modality has a :class:`GeneralExpert` (a :class:`ToolAgent` subclass)
whose runner is async and takes ``case_dir`` + ``vignette``. The agent
owns its list of underlying CV tools and an arbitration rule, and emits
the standard expert payload ``{"task", "algo_results", "expert_text"}``.

The allocator (an LLM ``AssistantAgent``) decides which experts to
invoke; once selected, an expert is a deterministic
``await expert.run(case_dir, vignette)`` away. No LLM call happens
inside an expert.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict

from ..core.agent_base import ToolAgent


@dataclass
class GeneralExpert(ToolAgent):
    """General-workflow expert. Runner is async over a case directory."""

    output_schema: str = ""

    async def run(self, case_dir: str, vignette: str, **kwargs: Any) -> Dict[str, Any]:
        if self.runner is None:
            raise RuntimeError(f"GeneralExpert({self.name}) has no runner bound")
        return await self.runner(case_dir=case_dir, vignette=vignette, **kwargs)


def build_general_experts() -> Dict[str, GeneralExpert]:
    """Build the seven expert agents.

    Runners live in :mod:`fetusagents.general.expert_runners` and are
    imported lazily here to keep ``experts.py`` cheap to load at module
    initialisation time.
    """
    from . import expert_runners as _runners

    def _bind(fn: Callable[..., Awaitable[Dict[str, Any]]]) -> Callable[..., Awaitable[Dict[str, Any]]]:
        async def _runner(*, case_dir: str, vignette: str, **kwargs: Any) -> Dict[str, Any]:
            return await fn(case_dir=case_dir, vignette=vignette, **kwargs)
        return _runner

    return {
        "plane_classification": GeneralExpert(
            name="plane_classification",
            description="Identify the standard fetal scan plane.",
            tools=["Plane-FetalCLIP", "Plane-FU-LoRA"],
            output_schema='{"recommended": <string>, "decision_note": <string>}',
            runner=_bind(_runners.run_plane_expert),
        ),
        "aop": GeneralExpert(
            name="aop",
            description="Predict angle of progression (AoP) from intra-partum images.",
            tools=["AoP-SAM", "USFM-AoP", "UperNet-AoP"],
            output_schema='{"recommended": <number|null>, "recommended_mask_path": <string|null>, "decision_note": <string>}',
            runner=_bind(_runners.run_aop_expert),
        ),
        "head_circumference": GeneralExpert(
            name="head_circumference",
            description="Measure head circumference with CSM + nnUNet voting.",
            tools=["CSM-HC", "HC-nnUNet"],
            output_schema='{"recommended": <number|null>, "recommended_mask_path": <string|null>, "decision_note": <string>}',
            runner=_bind(_runners.run_hc_expert),
        ),
        "gestational_age": GeneralExpert(
            name="gestational_age",
            description="Estimate gestational age with 3-way weighted vote, optional HC consistency check.",
            tools=["GA-RadImageNet", "GA-FetalCLIP", "GA-ConvNeXt"],
            output_schema='{"recommended": {"weeks": <int|null>, "days": <int|null>}, "decision_note": <string>}',
            runner=_bind(_runners.run_ga_expert),
        ),
        "brain_subplanes": GeneralExpert(
            name="brain_subplanes",
            description="Classify trans-thalamic / trans-cerebellum / trans-ventricular subplane.",
            tools=["BrainSubplane-FetalCLIP", "BrainSubplane-ResNet", "BrainSubplane-ViT"],
            output_schema='{"recommended": <string>, "decision_note": <string>}',
            runner=_bind(_runners.run_brain_subplane_expert),
        ),
        "stomach_segmentation": GeneralExpert(
            name="stomach_segmentation",
            description="Segment fetal stomach with FetalCLIP / FetalCLIP+SAMUS / nnUNet, shape-prior arbitrated.",
            tools=["StomachSeg-FetalCLIP", "StomachSeg-FetalCLIP+SAMUS", "StomachSeg-nnUNet"],
            output_schema='{"recommended": <string|null>, "decision_note": <string>}',
            runner=_bind(_runners.run_stomach_seg_expert),
        ),
        "abdomen_segmentation": GeneralExpert(
            name="abdomen_segmentation",
            description="Segment fetal abdomen and derive AC + Hadlock GA.",
            tools=["AbdomenSeg-FetalCLIP+SAMUS"],
            output_schema=(
                '{"recommended_mask_path": <string|null>, "recommended_ac_mm": <number|null>, '
                '"recommended_ga_weeks_from_ac": <number|null>, "decision_note": <string>}'
            ),
            runner=_bind(_runners.run_abdomen_seg_expert),
        ),
    }
