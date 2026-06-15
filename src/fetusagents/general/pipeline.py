"""Top-level multi-agent orchestrator.

This is the brains of the general workflow. It sets up the LLM agents
(``build_agents``), routes the allocator's decision to the per-modality
experts in :mod:`~fetusagents.general.expert_runners`, synthesises the
structured text summary, and exposes the public :func:`orchestrate`
entry point that the FetUSAgents general workflow adapter wraps.

Most of the file is prompt engineering and async LLM glue; the heavy
work is delegated to ``tool_runners`` and the biometry helpers.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

try:
    import numpy as np
except Exception:
    np = None

try:
    import cv2
except Exception:
    cv2 = None

# Module-level state and infrastructure
from ..core._state import _SCRIPT_DIR
from ..core.llm import build_model_client

# Biometry / GA helpers (used by report assembly + video summary)
from ..core.biometry import (
    _extract_lmp_ga_weeks,
    _format_ga_weeks_days,
    _hc_percentile_sanity_check,
    _load_ga_reference_table,
    _parse_expert_per_image,
    _percentile_assessment,
    _plane_display_name,
)

# Image / mask helpers used by report assembly + video summary
from ..core.image_utils import _make_single_image_case_dir

# Stdout / filename parsers
from ..core.parsing import (
    _is_video_summary_request,
    _resolve_image_key,
)

# Tool runners still used directly by the video-summary workflow.
from ..core.tool_runners import run_video_keyframe_tool


def _build_structured_text_summary(
    user_inquiry: str,
    images: List[str],
    expert_outputs: List[Dict[str, Any]],
) -> str:
    parsed = _parse_expert_per_image(expert_outputs)
    plane_map = parsed.get("plane_classification", {})
    brain_map = parsed.get("brain_subplanes", {})
    hc_map = parsed.get("head_circumference", {})
    ga_map = parsed.get("gestational_age", {})
    abd_map = parsed.get("abdomen_segmentation", {})
    sto_map = parsed.get("stomach_segmentation", {})

    # Extract HC algo_results for the percentile sanity check.
    hc_algo_final: Dict[str, Dict[str, Any]] = {}
    for item in expert_outputs:
        if item.get("task") == "head_circumference":
            hc_algo_final = (item.get("algo_results") or {}).get("final_hc") or {}
            break

    hc_ref_path = str(_SCRIPT_DIR / "reference" / "HC_GA_reference.csv")
    ac_ref_path = str(_SCRIPT_DIR / "reference" / "AC_GA_reference.csv")
    hc_table = _load_ga_reference_table(hc_ref_path)
    ac_table = _load_ga_reference_table(ac_ref_path)
    lmp_ga_weeks = _extract_lmp_ga_weeks(user_inquiry)

    image_names = set(images)
    for task_map in (plane_map, brain_map, hc_map, ga_map, abd_map, sto_map):
        image_names.update(task_map.keys())

    reports: List[str] = []
    for fname in sorted(image_names):
        plane = (plane_map.get(fname) or {}).get("recommended")
        brain_plane = (brain_map.get(fname) or {}).get("recommended")
        hc_mm = (hc_map.get(fname) or {}).get("recommended")
        hc_mask = (hc_map.get(fname) or {}).get("recommended_mask_path")
        ga_rec = (ga_map.get(fname) or {}).get("recommended") or {}
        ac_mm = (abd_map.get(fname) or {}).get("recommended_ac_mm")
        ac_ga = (abd_map.get(fname) or {}).get("recommended_ga_weeks_from_ac")
        abd_mask = (abd_map.get(fname) or {}).get("recommended_mask_path")
        sto_mask = (sto_map.get(fname) or {}).get("recommended")

        ga_us_weeks: Optional[float] = None
        if isinstance(ga_rec, dict):
            wk = ga_rec.get("weeks")
            dy = ga_rec.get("days")
            if wk is not None and dy is not None:
                try:
                    ga_us_weeks = float(wk) + float(dy) / 7.0
                except Exception:
                    ga_us_weeks = None

        # Post-hoc HC sanity check against percentile reference.
        if hc_mm is not None and ga_us_weeks is not None and hc_algo_final:
            hc_detail = hc_algo_final.get(fname, {})
            rec_src = hc_detail.get("source", "")
            csm_val = hc_detail.get("csm_hc_mm")
            nn_val = hc_detail.get("nnunet_hc_mm")
            if "csm" in rec_src:
                alt_val, alt_src = nn_val, "nnunet"
            else:
                alt_val, alt_src = csm_val, "csm"
            checked_hc, checked_src, check_note = _hc_percentile_sanity_check(
                float(hc_mm), alt_val, ga_us_weeks,
                hc_table, rec_src, alt_src,
            )
            if checked_hc is not None and abs(float(checked_hc) - float(hc_mm)) > 0.05:
                print(f"    [HC sanity check] {check_note}")
                hc_mm = checked_hc
                if "csm" in checked_src:
                    hc_mask = hc_detail.get("csm_mask_path") or hc_mask
                else:
                    hc_mask = hc_detail.get("nnunet_mask_path") or hc_mask

        findings: List[str] = []
        findings.append("Findings:")
        findings.append(f"Plane Identification: {_plane_display_name(plane)}")
        if brain_plane and str(brain_plane).upper() != "N/A":
            findings.append(f"Brain Plane Classification: {brain_plane}")
        if hc_mask:
            findings.append(f"Fetal Brain segmentation mask created in {hc_mask}")
        if abd_mask:
            findings.append(f"Abdomen segmentation mask created in {abd_mask}")
        if sto_mask:
            findings.append(f"Stomach segmentation mask created in {sto_mask}")
        if hc_mm is not None:
            findings.append(f"Head Circumference (HC) {float(hc_mm):.1f} mm.")
        if ac_mm is not None:
            findings.append(f"Estimated Abdomen Circumference (AC) {float(ac_mm):.1f} mm.")
        ga_us_text = _format_ga_weeks_days(ga_us_weeks)
        ga_ac_text = _format_ga_weeks_days(float(ac_ga) if ac_ga is not None else None)
        lmp_text = _format_ga_weeks_days(lmp_ga_weeks)
        if ga_us_text is not None:
            findings.append(f"Estimated Gestational Age (GA) {ga_us_text}.")
        if ga_ac_text is not None:
            findings.append(f"Estimated Gestational Age (GA) {ga_ac_text} (from Hadlock formula).")

        impression: List[str] = []
        impression.append("")
        impression.append("Impression:")
        ga_for_impression = ga_us_weeks if ga_us_weeks is not None else ac_ga
        ga_imp_text = _format_ga_weeks_days(float(ga_for_impression) if ga_for_impression is not None else None)
        if ga_imp_text is not None:
            impression.append(f"Estimated fetal age {ga_imp_text} by ultrasound measurement.")
        if lmp_text is not None:
            impression.append(f"Estimated fetal age {lmp_text} from last menstrual period (LMP).")

        # Growth statement priority:
        # 1) If LMP GA is present and AC is available -> AC vs LMP reference.
        # 2) Else if HC and GA(US) are available -> HC vs GA(US) reference.
        growth_line = None
        if lmp_ga_weeks is not None and ac_mm is not None:
            ac_assess = _percentile_assessment(ac_mm, lmp_ga_weeks, ac_table)
            if ac_assess:
                if ac_assess["status"] == "within":
                    growth_line = (
                        f"Compared with GA estimated from last menstrual period (LMP), AC falls within normal fetal growth range "
                        f"({ac_assess['band_text']})."
                    )
                elif ac_assess["status"] == "larger":
                    growth_line = (
                        f"Compared with GA estimated from last menstrual period (LMP), AC is larger than normal fetal growth range "
                        f"(>{ac_assess['normal_text'].split('-')[-1]})."
                    )
                else:
                    growth_line = (
                        f"Compared with GA estimated from last menstrual period (LMP), AC is smaller than normal fetal growth range "
                        f"(<{ac_assess['normal_text'].split('-')[0]})."
                    )
        elif hc_mm is not None and ga_us_weeks is not None:
            hc_assess = _percentile_assessment(hc_mm, ga_us_weeks, hc_table)
            if hc_assess:
                if hc_assess["status"] == "within":
                    growth_line = f"HC falls within normal fetal growth range ({hc_assess['band_text']})."
                elif hc_assess["status"] == "larger":
                    growth_line = f"HC is larger than normal fetal growth range (>{hc_assess['normal_text'].split('-')[-1]})."
                else:
                    growth_line = f"HC is smaller than normal fetal growth range (<{hc_assess['normal_text'].split('-')[0]})."
        if growth_line:
            impression.append(growth_line)

        report_text = "\n".join([f"Image: {fname}", ""] + findings + impression)
        reports.append(report_text)

    return "\n\n" + ("\n\n" + ("-" * 60) + "\n\n").join(reports)



def extract_agent_text(task_result: Any, agent_name: str) -> str:
    if not task_result or not getattr(task_result, "messages", None):
        return ""
    msgs = [m for m in task_result.messages if getattr(m, "source", None) == agent_name]
    if not msgs:
        for m in reversed(task_result.messages):
            if getattr(m, "type", "").endswith("TextMessage") or getattr(m, "content", None):
                return getattr(m, "content", "")
        return ""
    return getattr(msgs[-1], "content", "")


# Build Agents
def build_agents(model_client: OpenAIChatCompletionClient) -> Dict[str, Any]:
    """Wire one LLM agent (the allocator) and seven deterministic expert agents.

    The allocator is the only entity that talks to an LLM — it parses the
    user query, decides which experts to consult, and emits a
    ``Forwarding to: ...`` line. Each expert returned here is an
    :class:`GeneralExpert` whose policy is fixed in code (tool calls +
    arbitration) and exposes a uniform ``await agent.run(case_dir,
    vignette, **kwargs)`` surface.
    """
    from .experts import build_general_experts

    allocator = AssistantAgent(
        name="task_allocator",
        model_client=model_client,
        system_message=(
            "You are the Task Allocation Agent for fetal ultrasound analysis.\n"
            "Input: a user inquiry, image list, and optional preliminary plane-classification results.\n"
            "First decide Inquiry Type as either:\n"
            "  - specific: a targeted request (e.g., estimate GA, measure HC, predict AoP)\n"
            "  - general: broad/comprehensive request (e.g., comprehensive caption, all operations for this plane)\n"
            "Decide which experts to consult from: plane_classification, aop, head_circumference, gestational_age, brain_subplanes, stomach_segmentation, abdomen_segmentation.\n"
            "Rules:\n"
            "  - For specific inquiries, route directly to requested expert(s); do not force plane_classification unless explicitly needed.\n"
            "  - For general inquiries, include plane_classification first, then route plane-dependent experts.\n"
            "  - Respect plane-dependent capabilities summarized in the prompt.\n"
            "  - Output a line: Forwarding to: <comma-separated list>\n"
            "  - Output a line: Inquiry Type: <specific|general>\n"
            "  - After that, include 'Rephrased case:' followed by a concise case description.\n"
            "Example:\n"
            "Forwarding to: gestational_age, head_circumference\n"
            "Inquiry Type: specific\n\n"
            "Rephrased case: Estimate gestational age from the provided fetal brain ultrasound images."
        ),
    )

    return {"allocator": allocator, **build_general_experts()}


# Parse Allocator Output
def parse_forwarding_and_rephrased(text: str) -> Tuple[List[str], str, str]:
    forwarding = []
    rephrased = ""
    inquiry_type = "specific"
    t = text.strip()
    
    m = re.search(r"[Ff]orwarding to\s*:\s*([^\n\r]+)", t)
    if m:
        raw = m.group(1)
        parts = re.split(r"[,;]+|\band\b", raw)
        forwarding = [p.strip().lower() for p in parts if p.strip()]
    
    mrep = re.search(r"[Rr]ephrased(?: case)?\s*:\s*(.+)", t, flags=re.DOTALL)
    if mrep:
        rephrased = mrep.group(1).strip()
    else:
        if m:
            rephrased = re.sub(r"[Ff]orwarding to\s*:\s*[^\n\r]+\n?", "", t).strip()
        else:
            rephrased = t

    mt = re.search(r"[Ii]nquiry\s*[Tt]ype\s*:\s*(specific|general)", t)
    if mt:
        inquiry_type = mt.group(1).strip().lower()
    if inquiry_type not in ("specific", "general"):
        inquiry_type = "specific"

    return forwarding, rephrased, inquiry_type


def _enforce_per_image_json(final_text: str) -> str:
    """
    Ensure final output is per-image JSON objects under `per_image_reports`.
    If summarizer merged images into one object, split by image keys when possible.
    """
    try:
        data = json.loads(final_text)
    except Exception:
        return final_text

    if isinstance(data, dict) and isinstance(data.get("per_image_reports"), list):
        return json.dumps(data, ensure_ascii=False, indent=2)

    if not isinstance(data, dict):
        return final_text

    findings = data.get("findings", {})
    if not isinstance(findings, dict):
        return final_text

    # Collect image names from any findings sub-dict keyed by image.
    image_names: set[str] = set()
    for key in ("standard_plane", "brain_plane", "biometry", "segmentation"):
        v = findings.get(key)
        if isinstance(v, dict):
            for k, vv in v.items():
                if isinstance(vv, dict):
                    # nested form: metric -> {image -> value}
                    for kk in vv.keys():
                        if isinstance(kk, str) and kk.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")):
                            image_names.add(kk)
                if isinstance(k, str) and k.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")):
                    image_names.add(k)

    if not image_names:
        return json.dumps({"per_image_reports": []}, ensure_ascii=False, indent=2)

    reports: List[Dict[str, Any]] = []
    for image_name in sorted(image_names):
        std = None
        bp = None
        bio: Dict[str, Any] = {}
        seg: Dict[str, Any] = {}

        sv = findings.get("standard_plane")
        if isinstance(sv, dict) and image_name in sv:
            std = sv.get(image_name)

        bv = findings.get("brain_plane")
        if isinstance(bv, dict) and image_name in bv:
            bp = bv.get(image_name)

        biov = findings.get("biometry")
        if isinstance(biov, dict):
            if image_name in biov and isinstance(biov.get(image_name), dict):
                bio = biov.get(image_name, {})
            else:
                for metric, val in biov.items():
                    if isinstance(val, dict) and image_name in val:
                        bio[metric] = val.get(image_name)

        segv = findings.get("segmentation")
        if isinstance(segv, dict):
            if image_name in segv and isinstance(segv.get(image_name), dict):
                seg = segv.get(image_name, {})
            else:
                for metric, val in segv.items():
                    if isinstance(val, dict) and image_name in val:
                        seg[metric] = val.get(image_name)

        report = {
            "image_name": image_name,
            "patient_information": data.get("patient_information", {"patient_name": "", "date_of_exam": "", "indication": "", "technique": ""}),
            "findings": {
                "standard_plane": std,
                "brain_plane": bp,
                "biometry": bio,
                "segmentation": seg,
            },
            "impression": data.get("impression", {"estimated_fetal_age": "", "consistency": []}),
            "comments": data.get("comments", []),
        }
        reports.append(report)

    return json.dumps({"per_image_reports": reports}, ensure_ascii=False, indent=2)

# Orchestration
async def orchestrate(user_inquiry: str, case_dir: str) -> str:
    """
    Main orchestration function.
    
    Args:
        user_inquiry: User's question/request
        case_dir: Directory containing images and pixel_size.csv
    """
    model_client = build_model_client()
    agents = build_agents(model_client)

    try:
        # Validate case_dir
        if not os.path.isdir(case_dir):
            raise ValueError(f"Case directory does not exist: {case_dir}")
        
        images = [f for f in os.listdir(case_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f">>> Found {len(images)} images in {case_dir}")
        
        if len(images) == 0:
            raise ValueError(f"No images found in {case_dir}")

        if _is_video_summary_request(user_inquiry):
            print(">>> Video-summary request detected. Running key-frame driven workflow...")
            final_text = await run_video_summary_workflow(user_inquiry, case_dir, images, agents)
            print("=" * 60)
            print(">>> FINAL VIDEO REPORT:\n")
            print(final_text)
            print("=" * 60)
            return final_text

        expert_outputs: List[Dict[str, Any]] = []
        plane_summary: Dict[str, Any] = {}

        # Step I: First-pass allocation decides inquiry type (specific/general).
        alloc_prompt_1 = f"""User inquiry: {user_inquiry}

Available images in case directory:
{chr(10).join(['- ' + img for img in images[:20]])}
{"... and more" if len(images) > 20 else ""}

Capabilities:
- Brain plane tasks that are available: brain_subplanes, head_circumference, gestational_age
- Abdomen plane tasks that are available: abdomen_segmentation, stomach_segmentation
- AoP is independent and can be selected directly for specific AoP requests.

Decide inquiry type and initial experts."""

        allocator_res_1 = await agents["allocator"].run(task=alloc_prompt_1)
        allocator_text_1 = extract_agent_text(allocator_res_1, agents["allocator"].name)
        print(f">>> Allocator pass-1 output:\n{allocator_text_1}\n")

        selected, rephrased, inquiry_type = parse_forwarding_and_rephrased(allocator_text_1)
        candidates = {
            "plane_classification",
            "aop",
            "head_circumference",
            "gestational_age",
            "brain_subplanes",
            "stomach_segmentation",
            "abdomen_segmentation",
        }
        selected = [s for s in selected if s in candidates]
        vignette = rephrased if rephrased else user_inquiry

        # General inquiries keep plane-first logic.
        if inquiry_type == "general":
            plane_out = await agents["plane_classification"].run(case_dir, user_inquiry)
            expert_outputs.append(plane_out)
            try:
                plane_json = json.loads(plane_out.get("expert_text", "{}"))
                plane_summary = plane_json.get("per_image", {}) if isinstance(plane_json, dict) else {}
            except Exception:
                plane_summary = {}
            alloc_prompt_2 = f"""User inquiry: {user_inquiry}

Available images in case directory:
{chr(10).join(['- ' + img for img in images[:20]])}
{"... and more" if len(images) > 20 else ""}

Preliminary plane classification result (source of truth):
{json.dumps(plane_summary, ensure_ascii=False)}

Capabilities by plane context:
- Brain plane tasks that are available: brain_subplanes, head_circumference, gestational_age
- Abdomen plane tasks that are available: abdomen_segmentation, stomach_segmentation
- AoP is independent and can be selected when requested by the inquiry.

This is a general inquiry. Decide which experts to consult next and rephrase the case."""
            allocator_res_2 = await agents["allocator"].run(task=alloc_prompt_2)
            allocator_text_2 = extract_agent_text(allocator_res_2, agents["allocator"].name)
            print(f">>> Allocator pass-2 output:\n{allocator_text_2}\n")
            selected2, rephrased2, _ = parse_forwarding_and_rephrased(allocator_text_2)
            selected = [s for s in selected2 if s in candidates and s != "plane_classification"]
            if rephrased2:
                vignette = rephrased2
        else:
            # Specific inquiry: keep direct routing; do not force plane classification.
            selected = [s for s in selected if s != "plane_classification"]

        # GA should consume HC cross-check when GA is requested.
        if "gestational_age" in selected and "head_circumference" not in selected:
            selected.append("head_circumference")

        # Deterministic execution order after plane is done.
        desired_order = [
            "plane_classification",
            "brain_subplanes",
            "aop",
            "stomach_segmentation",
            "abdomen_segmentation",
            "head_circumference",
            "gestational_age",
        ]
        order_index = {name: i for i, name in enumerate(desired_order)}
        seen = set()
        selected = [x for x in selected if not (x in seen or seen.add(x))]
        selected.sort(key=lambda x: order_index.get(x, 999))

        print(f">>> Experts selected (after plane-first allocation): {selected}\n")

        # Step II: Expert execution (remaining experts)
        hc_algo_results_for_ga: Optional[Dict[str, Any]] = None
        
        for name in selected:
            if name == "aop":
                expert_outputs.append(await agents["aop"].run(case_dir, vignette))
            elif name == "plane_classification":
                expert_outputs.append(await agents["plane_classification"].run(case_dir, vignette))
            elif name == "brain_subplanes":
                expert_outputs.append(await agents["brain_subplanes"].run(case_dir, vignette))
            elif name == "stomach_segmentation":
                expert_outputs.append(await agents["stomach_segmentation"].run(case_dir, vignette))
            elif name == "abdomen_segmentation":
                expert_outputs.append(await agents["abdomen_segmentation"].run(case_dir, vignette))
            elif name == "head_circumference":
                hc_out = await agents["head_circumference"].run(case_dir, vignette)
                hc_algo_results_for_ga = hc_out.get("algo_results")
                expert_outputs.append(hc_out)
            elif name == "gestational_age":
                expert_outputs.append(await agents["gestational_age"].run(case_dir, vignette, hc_algo_results=hc_algo_results_for_ga))

        # Print expert messages
        print(">>> Expert messages to summarizer:")
        for item in expert_outputs:
            print(f"[Expert={item['task']}]\n{item['expert_text']}\n----\n")

        # Step III: Deterministic final report formatting (less JSON-like, template-style).
        final_text = _build_structured_text_summary(
            user_inquiry=user_inquiry,
            images=images,
            expert_outputs=expert_outputs,
        )
        
        print("=" * 60)
        print(">>> FINAL ANSWER:\n")
        print(final_text)
        print("=" * 60)
        
        return final_text
        
    finally:
        try:
            await model_client.close()
        except Exception as e:
            print(f"[ModelClient] close failed: {e}")


async def run_video_summary_workflow(
    user_inquiry: str,
    case_dir: str,
    images: List[str],
    agents: Dict[str, AssistantAgent],
) -> str:
    def _safe_median(vals: List[float]) -> Optional[float]:
        if not vals:
            return None
        if np is not None:
            return float(np.median(vals))
        s = sorted(vals)
        n = len(s)
        m = n // 2
        if n % 2 == 1:
            return float(s[m])
        return float((s[m - 1] + s[m]) / 2.0)

    def _frame_caption(case: Dict[str, Any]) -> str:
        fname = case["image_name"]
        plane_raw = case["plane_raw"]
        plane_norm = case["plane_norm"]
        lines: List[str] = []
        if plane_norm == "other":
            lines.append(f"{fname} is classified as No Plane (non-key frame).")
            return " ".join(lines)
        lines.append(f"{fname} shows a clear view of fetal {str(plane_raw).lower()}.")
        if case.get("hc_mm") is not None:
            lines.append(f"Estimated HC is {float(case['hc_mm']):.1f} mm (from {fname}).")
        if case.get("ga_us_weeks") is not None:
            lines.append(f"Estimated GA is {_format_ga_weeks_days(float(case['ga_us_weeks']))} (from {fname}).")
        if case.get("ac_mm") is not None:
            lines.append(f"Estimated AC is {float(case['ac_mm']):.1f} mm (from {fname}).")
        if case.get("ac_ga_weeks") is not None:
            lines.append(
                f"Estimated GA from AC is {_format_ga_weeks_days(float(case['ac_ga_weeks']))} (from {fname})."
            )
        if case.get("stomach_mask_path"):
            lines.append(f"Stomach segmentation mask available at {case['stomach_mask_path']}.")
        if case.get("growth_note"):
            lines.append(case["growth_note"])
        if plane_norm in ("femur", "thorax", "spine"):
            lines.append("No downstream biometry expert is configured for this plane in current system.")
        return " ".join(lines)

    def _video_report_generator(
        user_inquiry_text: str,
        case_summaries: List[Dict[str, Any]],
        hc_table: List[Dict[str, Any]],
        ac_table: List[Dict[str, Any]],
    ) -> str:
        planes: Dict[str, List[str]] = {}
        hc_vals: List[float] = []
        ga_vals: List[float] = []
        ac_vals: List[float] = []
        ac_ga_vals: List[float] = []
        hc_frames: List[str] = []
        ga_frames: List[str] = []
        ac_frames: List[str] = []
        ac_ga_frames: List[str] = []

        for c in case_summaries:
            if c["plane_norm"] != "other":
                planes.setdefault(c["plane_raw"], []).append(c["image_name"])
            if c.get("hc_mm") is not None:
                hc_vals.append(float(c["hc_mm"]))
                hc_frames.append(c["image_name"])
            if c.get("ga_us_weeks") is not None:
                ga_vals.append(float(c["ga_us_weeks"]))
                ga_frames.append(c["image_name"])
            if c.get("ac_mm") is not None:
                ac_vals.append(float(c["ac_mm"]))
                ac_frames.append(c["image_name"])
            if c.get("ac_ga_weeks") is not None:
                ac_ga_vals.append(float(c["ac_ga_weeks"]))
                ac_ga_frames.append(c["image_name"])

        med_hc = _safe_median(hc_vals)
        med_ga = _safe_median(ga_vals)
        med_ac = _safe_median(ac_vals)
        med_ac_ga = _safe_median(ac_ga_vals)
        lmp_ga_weeks = _extract_lmp_ga_weeks(user_inquiry_text)

        # ---- Build structured report ----
        lines: List[str] = []
        lines.append("Findings:")
        lines.append("")

        # 1) Frame-level plane detection
        lines.append("Frame-level Plane Detection:")
        for c in case_summaries:
            if c["plane_norm"] == "other":
                continue
            lines.append(f"  {c['image_name']} shows a clear view of fetal {str(c['plane_raw']).lower()}.")
        non_key = [c["image_name"] for c in case_summaries if c["plane_norm"] == "other"]
        if non_key:
            lines.append(f"  {len(non_key)} frame(s) classified as non-key (no standard plane detected).")
        lines.append("")

        # Summary of plane counts
        lines.append("Detected Planes Summary:")
        for plane_name, files in sorted(planes.items(), key=lambda kv: kv[0].lower()):
            lines.append(f"  {plane_name}: {len(files)} frame(s)")
        lines.append("")

        # 2) Biometry results
        lines.append("Biometry Results:")
        if med_hc is not None:
            src_txt = ", ".join(hc_frames)
            lines.append(f"  Estimated Head Circumference (HC): {med_hc:.1f} mm"
                         f"{' (median from ' + src_txt + ')' if len(hc_frames) > 1 else ' (from ' + src_txt + ')'}.")
        if med_ga is not None:
            src_txt = ", ".join(ga_frames)
            lines.append(f"  Estimated Gestational Age (GA): {_format_ga_weeks_days(med_ga)}"
                         f"{' (median from ' + src_txt + ')' if len(ga_frames) > 1 else ' (from ' + src_txt + ')'}.")
        if med_ac is not None:
            src_txt = ", ".join(ac_frames)
            lines.append(f"  Estimated Abdomen Circumference (AC): {med_ac:.1f} mm"
                         f"{' (median from ' + src_txt + ')' if len(ac_frames) > 1 else ' (from ' + src_txt + ')'}.")
        if med_ac_ga is not None:
            src_txt = ", ".join(ac_ga_frames)
            lines.append(f"  Estimated GA from AC (Hadlock): {_format_ga_weeks_days(med_ac_ga)}"
                         f"{' (median from ' + src_txt + ')' if len(ac_ga_frames) > 1 else ' (from ' + src_txt + ')'}.")
        if lmp_ga_weeks is not None:
            lines.append(f"  Estimated fetal age {_format_ga_weeks_days(lmp_ga_weeks)} from last menstrual period (LMP).")
        if med_hc is None and med_ga is None and med_ac is None:
            lines.append("  No biometry results available from key frames.")
        lines.append("")

        # 3) Impression & Growth evaluation
        lines.append("Impression:")
        if med_ga is not None:
            lines.append(f"  Estimated fetal age {_format_ga_weeks_days(med_ga)} by ultrasound.")
        elif med_ac_ga is not None:
            lines.append(f"  Estimated fetal age {_format_ga_weeks_days(med_ac_ga)} by ultrasound (from AC/Hadlock).")

        growth_line = None
        if med_hc is not None and med_ga is not None:
            hc_assess = _percentile_assessment(med_hc, med_ga, hc_table)
            if hc_assess:
                if hc_assess["status"] == "within":
                    growth_line = f"HC falls within normal fetal growth range ({hc_assess['band_text']})."
                elif hc_assess["status"] == "larger":
                    growth_line = f"HC is larger than normal fetal growth range (>{hc_assess['normal_text'].split('-')[-1]})."
                else:
                    growth_line = f"HC is smaller than normal fetal growth range (<{hc_assess['normal_text'].split('-')[0]})."
        if growth_line is None and med_ac is not None and lmp_ga_weeks is not None:
            ac_assess = _percentile_assessment(med_ac, lmp_ga_weeks, ac_table)
            if ac_assess:
                if ac_assess["status"] == "within":
                    growth_line = (
                        f"Compared with GA from LMP, AC falls within normal fetal growth range "
                        f"({ac_assess['band_text']})."
                    )
                elif ac_assess["status"] == "larger":
                    growth_line = (
                        f"Compared with GA from LMP, AC is larger than normal fetal growth range "
                        f"(>{ac_assess['normal_text'].split('-')[-1]})."
                    )
                else:
                    growth_line = (
                        f"Compared with GA from LMP, AC is smaller than normal fetal growth range "
                        f"(<{ac_assess['normal_text'].split('-')[0]})."
                    )

        if growth_line:
            lines.append(f"  {growth_line}")
        else:
            lines.append("  Growth evaluation unavailable (insufficient paired biometry/GA evidence).")

        return "\n".join(lines)

    keyframe_res = run_video_keyframe_tool(case_dir)
    if not keyframe_res.ok:
        return "Video summary failed: key-frame detector did not return valid outputs."

    image_to_plane: Dict[str, Dict[str, Any]] = {}
    for pred_name, pdata in keyframe_res.per_image.items():
        matched = _resolve_image_key(pred_name, images)
        if not matched:
            continue
        image_to_plane[matched] = pdata

    for img in images:
        if img not in image_to_plane:
            image_to_plane[img] = {
                "pred_plane_raw": "No Plane",
                "pred_plane_norm": "other",
                "is_key_frame": False,
            }

    hc_ref_path = str(_SCRIPT_DIR / "reference" / "HC_GA_reference.csv")
    ac_ref_path = str(_SCRIPT_DIR / "reference" / "AC_GA_reference.csv")
    hc_table = _load_ga_reference_table(hc_ref_path)
    ac_table = _load_ga_reference_table(ac_ref_path)
    case_summaries: List[Dict[str, Any]] = []
    total_frames = len(images)

    print(f"\n>>> Processing {total_frames} frames individually...")
    print("-" * 60)

    for frame_idx, fname in enumerate(sorted(images), start=1):
        pred = image_to_plane.get(fname, {})
        plane_raw = str(pred.get("pred_plane_raw") or "No Plane")
        plane_norm = str(pred.get("pred_plane_norm") or "other")

        print(f"\n>>> [{frame_idx}/{total_frames}] Processing: {fname}")
        print(f"    Detected plane: {plane_raw}")

        single_dir = _make_single_image_case_dir(case_dir, fname)
        case_item: Dict[str, Any] = {
            "image_name": fname,
            "plane_raw": plane_raw,
            "plane_norm": plane_norm,
            "hc_mm": None,
            "ga_us_weeks": None,
            "ac_mm": None,
            "ac_ga_weeks": None,
            "stomach_mask_path": None,
            "growth_note": None,
            "caption": "",
        }
        try:
            if plane_norm == "brain":
                print("    Experts assigned: head_circumference, gestational_age")
                hc_out = await agents["head_circumference"].run(single_dir, user_inquiry)
                ga_out = await agents["gestational_age"].run(
                    single_dir,
                    user_inquiry,
                    hc_algo_results=hc_out.get("algo_results"),
                )
                parsed = _parse_expert_per_image([hc_out, ga_out])
                hc_map = parsed.get("head_circumference", {})
                ga_map = parsed.get("gestational_age", {})
                hc_entry = next(iter(hc_map.values()), {}) if hc_map else {}
                ga_entry = next(iter(ga_map.values()), {}) if ga_map else {}
                hc_mm = hc_entry.get("recommended")
                ga_rec = ga_entry.get("recommended") or {}
                ga_w = ga_rec.get("weeks")
                ga_d = ga_rec.get("days")
                ga_total = None
                if ga_w is not None and ga_d is not None:
                    ga_total = float(ga_w) + float(ga_d) / 7.0
                if hc_mm is not None:
                    case_item["hc_mm"] = float(hc_mm)
                if ga_total is not None:
                    case_item["ga_us_weeks"] = float(ga_total)

                # Post-hoc HC sanity check: if recommended HC is out of
                # percentile range but the other tool is in range, switch.
                if case_item["hc_mm"] is not None and case_item["ga_us_weeks"] is not None:
                    hc_final_map = (hc_out.get("algo_results") or {}).get("final_hc") or {}
                    hc_detail = next(iter(hc_final_map.values()), {}) if hc_final_map else {}
                    rec_src = hc_detail.get("source", "")
                    csm_val = hc_detail.get("csm_hc_mm")
                    nn_val = hc_detail.get("nnunet_hc_mm")
                    if "csm" in rec_src:
                        alt_val, alt_src = nn_val, "nnunet"
                    else:
                        alt_val, alt_src = csm_val, "csm"
                    checked_hc, checked_src, check_note = _hc_percentile_sanity_check(
                        case_item["hc_mm"], alt_val, case_item["ga_us_weeks"],
                        hc_table, rec_src, alt_src,
                    )
                    if checked_hc is not None and checked_hc != case_item["hc_mm"]:
                        print(f"    [HC sanity check] {check_note}")
                        case_item["hc_mm"] = float(checked_hc)

                if case_item["hc_mm"] is not None and case_item["ga_us_weeks"] is not None:
                    hc_assess = _percentile_assessment(case_item["hc_mm"], case_item["ga_us_weeks"], hc_table)
                    if hc_assess:
                        if hc_assess["status"] == "within":
                            case_item["growth_note"] = f"HC is within normal range ({hc_assess['band_text']})."
                        elif hc_assess["status"] == "larger":
                            case_item["growth_note"] = (
                                f"HC is larger than normal range (>{hc_assess['normal_text'].split('-')[-1]})."
                            )
                        else:
                            case_item["growth_note"] = (
                                f"HC is smaller than normal range (<{hc_assess['normal_text'].split('-')[0]})."
                            )
            elif plane_norm == "abdomen":
                print("    Experts assigned: abdomen_segmentation, stomach_segmentation")
                abd_out = await agents["abdomen_segmentation"].run(single_dir, user_inquiry)
                sto_out = await agents["stomach_segmentation"].run(single_dir, user_inquiry)
                parsed = _parse_expert_per_image([abd_out, sto_out])
                abd_map = parsed.get("abdomen_segmentation", {})
                sto_map = parsed.get("stomach_segmentation", {})
                abd_entry = next(iter(abd_map.values()), {}) if abd_map else {}
                sto_entry = next(iter(sto_map.values()), {}) if sto_map else {}
                ac_mm = abd_entry.get("recommended_ac_mm")
                ac_ga = abd_entry.get("recommended_ga_weeks_from_ac")
                # Stomach mask path: stomach_segmentation's recommended slot
                # holds the mask path string (see expert_runners.run_stomach_seg_expert).
                sto_mask = sto_entry.get("recommended")
                if sto_mask:
                    case_item["stomach_mask_path"] = sto_mask
                if ac_mm is not None:
                    case_item["ac_mm"] = float(ac_mm)
                if ac_ga is not None:
                    case_item["ac_ga_weeks"] = float(ac_ga)
                lmp_ga_weeks = _extract_lmp_ga_weeks(user_inquiry)
                if case_item["ac_mm"] is not None and lmp_ga_weeks is not None:
                    ac_assess = _percentile_assessment(case_item["ac_mm"], lmp_ga_weeks, ac_table)
                    if ac_assess:
                        if ac_assess["status"] == "within":
                            case_item["growth_note"] = (
                                f"Compared with GA from LMP, AC is within normal range ({ac_assess['band_text']})."
                            )
                        elif ac_assess["status"] == "larger":
                            case_item["growth_note"] = (
                                f"Compared with GA from LMP, AC is larger than normal range "
                                f"(>{ac_assess['normal_text'].split('-')[-1]})."
                            )
                        else:
                            case_item["growth_note"] = (
                                f"Compared with GA from LMP, AC is smaller than normal range "
                                f"(<{ac_assess['normal_text'].split('-')[0]})."
                            )
            elif plane_norm in ("femur", "thorax", "spine"):
                print(f"    Experts assigned: none (no biometry/segmentation expert for {plane_raw})")
            else:
                print("    Non-key frame; skipping downstream experts.")
        except Exception as e:
            print(f"    [WARNING] Expert execution failed for {fname}: {e}")
        finally:
            shutil.rmtree(single_dir, ignore_errors=True)

        case_item["caption"] = _frame_caption(case_item)
        case_summaries.append(case_item)
        print(f"    Caption: {case_item['caption']}")

    print("\n" + "-" * 60)
    print(f">>> All {total_frames} frames processed. Generating video report...\n")

    report = _video_report_generator(user_inquiry, case_summaries, hc_table, ac_table)
    return report


