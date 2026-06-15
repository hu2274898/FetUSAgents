"""``run_*_tool_for_sample`` adapters.

Each function takes a :class:`VQASample`, calls one or more general-
workflow tool runners (via :func:`load_tools_from_demo`), aggregates
the outputs into a :class:`ToolDecision`, and returns it.

The CV tool runners live in :mod:`fetusagents.core.tool_runners` and
are imported lazily inside :func:`load_tools_from_demo` so loading this
file is cheap even when the heavy ML environments are missing.
"""
from __future__ import annotations

import math
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:
    np = None

try:
    import cv2
except Exception:
    cv2 = None

from .types import ToolDecision, VQASample
from .knowledge import (
    BRAIN_CANONICAL_TO_OPTION,
    BRAIN_OPTION_TO_TEXT,
    PLANE_CANONICAL_TO_OPTION,
    PLANE_OPTION_TO_TEXT,
    TRIMESTER_OPTION_TO_TEXT,
    YESNO_OPTION_TO_TEXT,
)
from .parsers import (
    _majority_vote_masks,
    _stomach_area_pixel_from_mask_array,
    ensure_single_image_case_dir,
    extract_target_brain_subplane_from_question,
    extract_target_plane_from_question,
    extract_target_trimester_from_question,
    normalize_brain_subplane_label,
    normalize_plane_label,
    parse_numeric_options,
    pick_closest_option_letter,
    trimester_from_total_weeks,
)

# Shared mask / image helpers (one source of truth in ``core``).
from ..core.image_utils import _mask_to_raw_array, _safe_load_pil
from ..core.biometry import _ac_mm_from_mask_array


def load_tools_from_demo() -> Dict[str, Any]:
    try:
        from ..core.tool_runners import (
            run_plane_fetalclip_tool,
            run_plane_fulora_tool,
            run_brain_subplane_fetalclip_tool,
            run_brain_subplane_resnet_tool,
            run_brain_subplane_vit_tool,
            run_ga_algo1_tool,
            run_ga_algo2_tool,
            run_ga_algo3_tool,
            run_csm_hc_tool,
            run_nnunet_hc_tool,
            run_aop_sam_tool,
            run_usfm_aop_tool,
            run_upernet_aop_tool,
            run_abdomen_fetalclip_seg_tool,
            run_abdomen_fetalclip_samus_seg_tool,
            run_stomach_fetalclip_seg_tool,
            run_stomach_fetalclip_samus_seg_tool,
            run_stomach_nnunet_seg_tool,
        )
    except Exception as e:
        raise RuntimeError(f"Unable to import CV tool runners: {e}")

    return {
        "run_plane_fetalclip_tool": run_plane_fetalclip_tool,
        "run_plane_fulora_tool": run_plane_fulora_tool,
        "run_brain_subplane_fetalclip_tool": run_brain_subplane_fetalclip_tool,
        "run_brain_subplane_resnet_tool": run_brain_subplane_resnet_tool,
        "run_brain_subplane_vit_tool": run_brain_subplane_vit_tool,
        "run_ga_algo1_tool": run_ga_algo1_tool,
        "run_ga_algo2_tool": run_ga_algo2_tool,
        "run_ga_algo3_tool": run_ga_algo3_tool,
        "run_csm_hc_tool": run_csm_hc_tool,
        "run_nnunet_hc_tool": run_nnunet_hc_tool,
        "run_aop_sam_tool": run_aop_sam_tool,
        "run_usfm_aop_tool": run_usfm_aop_tool,
        "run_upernet_aop_tool": run_upernet_aop_tool,
        "run_abdomen_fetalclip_seg_tool": run_abdomen_fetalclip_seg_tool,
        "run_abdomen_fetalclip_samus_seg_tool": run_abdomen_fetalclip_samus_seg_tool,
        "run_stomach_fetalclip_seg_tool": run_stomach_fetalclip_seg_tool,
        "run_stomach_fetalclip_samus_seg_tool": run_stomach_fetalclip_samus_seg_tool,
        "run_stomach_nnunet_seg_tool": run_stomach_nnunet_seg_tool,
    }


# tool：plane
def run_plane_tool_for_sample(sample: VQASample) -> ToolDecision:
    image_path = sample.image_path
    tools = load_tools_from_demo()
    tmp_dir = ensure_single_image_case_dir(image_path)

    try:
        result1 = tools["run_plane_fetalclip_tool"](tmp_dir)
        result2 = tools["run_plane_fulora_tool"](tmp_dir)

        fname = os.path.basename(image_path)
        r1 = result1.per_image.get(fname, {}) if getattr(result1, "per_image", None) else {}
        r2 = result2.per_image.get(fname, {}) if getattr(result2, "per_image", None) else {}

        plane1 = normalize_plane_label(r1.get("plane"))
        plane2 = normalize_plane_label(r2.get("plane"))

        if plane1 == plane2 and plane1 != "other":
            final_plane = plane1
            note = "two plane tools agree"
        elif plane2 == "other" and plane1 != "other":
            final_plane = plane1
            note = "FU-LoRA returned other, use FetalCLIP"
        elif plane1 == "other" and plane2 != "other":
            final_plane = plane2
            note = "FetalCLIP returned other, use FU-LoRA"
        else:
            final_plane = plane1 if plane1 != "other" else plane2
            note = "tools disagree, fallback to preferred valid output"

        final_letter = PLANE_CANONICAL_TO_OPTION.get(final_plane)
        final_text = PLANE_OPTION_TO_TEXT.get(final_letter) if final_letter else None

        return ToolDecision(
            used_tool=True,
            tool_name="plane_fetalclip + plane_fulora",
            tool_answer_letter=final_letter,
            tool_answer_text=final_text,
            tool_detail={
                "tool1_raw": r1,
                "tool2_raw": r2,
                "tool1_plane": plane1,
                "tool2_plane": plane2,
                "note": note,
            },
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def run_plane_binary_tool_for_sample(sample: VQASample) -> ToolDecision:
    base_decision = run_plane_tool_for_sample(sample)

    target_label = extract_target_plane_from_question(sample.question)

    predicted_label = None
    if base_decision.tool_answer_letter is not None:
        inv_map = {v: k for k, v in PLANE_CANONICAL_TO_OPTION.items()}
        predicted_label = inv_map.get(base_decision.tool_answer_letter)

    if target_label is None or predicted_label is None:
        return ToolDecision(
            used_tool=True,
            tool_name="plane_binary_via_plane_tools",
            tool_answer_letter=None,
            tool_answer_text=None,
            tool_detail={
                "target_label": target_label,
                "predicted_label": predicted_label,
                "base_tool_detail": base_decision.tool_detail,
                "note": "could not determine target or predicted plane",
            },
        )

    final_letter = "A" if predicted_label == target_label else "B"
    final_text = YESNO_OPTION_TO_TEXT[final_letter]

    return ToolDecision(
        used_tool=True,
        tool_name="plane_binary_via_plane_tools",
        tool_answer_letter=final_letter,
        tool_answer_text=final_text,
        tool_detail={
            "target_label": target_label,
            "predicted_label": predicted_label,
            "base_tool_answer_letter": base_decision.tool_answer_letter,
            "base_tool_answer_text": base_decision.tool_answer_text,
            "base_tool_detail": base_decision.tool_detail,
            "note": "A=Yes if predicted plane matches target, else B=No",
        },
    )
# tool：brain_subplane
def run_brain_subplane_tool_for_sample(sample: VQASample) -> ToolDecision:
    image_path = sample.image_path
    tools = load_tools_from_demo()
    tmp_dir = ensure_single_image_case_dir(image_path)

    try:
        result1 = tools["run_brain_subplane_fetalclip_tool"](tmp_dir)
        result2 = tools["run_brain_subplane_resnet_tool"](tmp_dir)
        result3 = tools["run_brain_subplane_vit_tool"](tmp_dir)

        fname = os.path.basename(image_path)
        r1 = result1.per_image.get(fname, {}) if getattr(result1, "per_image", None) else {}
        r2 = result2.per_image.get(fname, {}) if getattr(result2, "per_image", None) else {}
        r3 = result3.per_image.get(fname, {}) if getattr(result3, "per_image", None) else {}

        l1 = normalize_brain_subplane_label(r1.get("subplane"))
        l2 = normalize_brain_subplane_label(r2.get("subplane"))
        l3 = normalize_brain_subplane_label(r3.get("subplane"))

        final_label = None
        note = "no valid tool output"

        if l1 == l2 == l3 and l1 != "other":
            final_label = l1
            note = "three tools agree"
        elif l1 != "other" and l1 == l2:
            final_label = l1
            note = "majority: fetalclip + resnet"
        elif l1 != "other" and l1 == l3:
            final_label = l1
            note = "majority: fetalclip + vit"
        elif l2 != "other" and l2 == l3:
            final_label = l2
            note = "majority: resnet + vit"
        elif l1 != "other":
            final_label = l1
            note = "all disagree, fallback to fetalclip"
        elif l2 != "other":
            final_label = l2
            note = "fallback to resnet"
        elif l3 != "other":
            final_label = l3
            note = "fallback to vit"

        final_letter = BRAIN_CANONICAL_TO_OPTION.get(final_label) if final_label else None
        final_text = BRAIN_OPTION_TO_TEXT.get(final_letter) if final_letter else None

        return ToolDecision(
            used_tool=True,
            tool_name="brain_subplane_fetalclip + resnet + vit",
            tool_answer_letter=final_letter,
            tool_answer_text=final_text,
            tool_detail={
                "tool1_raw": r1,
                "tool2_raw": r2,
                "tool3_raw": r3,
                "tool1_label": l1,
                "tool2_label": l2,
                "tool3_label": l3,
                "final_label": final_label,
                "note": note,
            },
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def run_brain_subplane_binary_tool_for_sample(sample: VQASample) -> ToolDecision:
    base_decision = run_brain_subplane_tool_for_sample(sample)

    target_label = extract_target_brain_subplane_from_question(sample.question)

    predicted_label = None
    if base_decision.tool_answer_letter is not None:
        inv_map = {v: k for k, v in BRAIN_CANONICAL_TO_OPTION.items()}
        predicted_label = inv_map.get(base_decision.tool_answer_letter)

    if target_label is None or predicted_label is None:
        return ToolDecision(
            used_tool=True,
            tool_name="brain_subplane_binary_via_brain_subplane_tools",
            tool_answer_letter=None,
            tool_answer_text=None,
            tool_detail={
                "target_label": target_label,
                "predicted_label": predicted_label,
                "base_tool_detail": base_decision.tool_detail,
                "note": "could not determine target or predicted subplane",
            },
        )

    final_letter = "A" if predicted_label == target_label else "B"
    final_text = YESNO_OPTION_TO_TEXT[final_letter]

    return ToolDecision(
        used_tool=True,
        tool_name="brain_subplane_binary_via_brain_subplane_tools",
        tool_answer_letter=final_letter,
        tool_answer_text=final_text,
        tool_detail={
            "target_label": target_label,
            "predicted_label": predicted_label,
            "base_tool_answer_letter": base_decision.tool_answer_letter,
            "base_tool_answer_text": base_decision.tool_answer_text,
            "base_tool_detail": base_decision.tool_detail,
            "note": "A=Yes if predicted subplane matches target, else B=No",
        },
    )

def run_ga_trimester_binary_tool_for_sample(sample: VQASample) -> ToolDecision:
    tools = load_tools_from_demo()
    tmp_dir = ensure_single_image_case_dir(sample.image_path, pixel_size_mm=sample.pixel_size)

    try:
        result1 = tools["run_ga_algo1_tool"](tmp_dir)
        result2 = tools["run_ga_algo2_tool"](tmp_dir)
        result3 = tools["run_ga_algo3_tool"](tmp_dir)

        fname = os.path.basename(sample.image_path)
        r1 = result1.per_image.get(fname, {}) if getattr(result1, "per_image", None) else {}
        r2 = result2.per_image.get(fname, {}) if getattr(result2, "per_image", None) else {}
        r3 = result3.per_image.get(fname, {}) if getattr(result3, "per_image", None) else {}

        vals = []
        if r1.get("total_weeks") is not None:
            vals.append(("ga_algo1", float(r1["total_weeks"]), 1.0))
        if r2.get("total_weeks") is not None:
            vals.append(("ga_algo2", float(r2["total_weeks"]), 2.0))
        if r3.get("total_weeks") is not None:
            vals.append(("ga_algo3", float(r3["total_weeks"]), 1.0))

        recommended_total_weeks = None
        source_note = "no valid ga output"

        if len(vals) >= 2:
            best_pair = None
            best_diff = None
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    diff = abs(vals[i][1] - vals[j][1])
                    if best_diff is None or diff < best_diff:
                        best_diff = diff
                        best_pair = (vals[i], vals[j])

            if best_pair is not None and best_diff is not None and best_diff <= 1.5:
                v1, v2 = best_pair
                recommended_total_weeks = (v1[1] * v1[2] + v2[1] * v2[2]) / (v1[2] + v2[2])
                source_note = f"pair-vote agreement: {v1[0]} + {v2[0]}"
            else:
                wsum = sum(v[2] for v in vals)
                recommended_total_weeks = sum(v[1] * v[2] for v in vals) / wsum
                source_note = "weighted-mean fallback"
        elif len(vals) == 1:
            recommended_total_weeks = vals[0][1]
            source_note = f"single-tool fallback: {vals[0][0]}"

        target_trimester = extract_target_trimester_from_question(sample.question)
        predicted_trimester = None if recommended_total_weeks is None else trimester_from_total_weeks(recommended_total_weeks)

        if target_trimester is None or predicted_trimester is None:
            return ToolDecision(
                used_tool=True,
                tool_name="ga_trimester_binary_via_ga_tools",
                tool_answer_letter=None,
                tool_answer_text=None,
                tool_detail={
                    "tool1_raw": r1,
                    "tool2_raw": r2,
                    "tool3_raw": r3,
                    "recommended_total_weeks": recommended_total_weeks,
                    "target_trimester": target_trimester,
                    "predicted_trimester": predicted_trimester,
                    "note": "could not determine target trimester or GA prediction",
                },
            )

        final_letter = "A" if predicted_trimester == target_trimester else "B"
        final_text = YESNO_OPTION_TO_TEXT[final_letter]

        return ToolDecision(
            used_tool=True,
            tool_name="ga_trimester_binary_via_ga_tools",
            tool_answer_letter=final_letter,
            tool_answer_text=final_text,
            tool_detail={
                "tool1_raw": r1,
                "tool2_raw": r2,
                "tool3_raw": r3,
                "recommended_total_weeks": recommended_total_weeks,
                "target_trimester": target_trimester,
                "predicted_trimester": predicted_trimester,
                "decision_source": source_note,
                "note": "A=Yes if predicted trimester matches target trimester, else B=No",
            },
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def run_hc_pixel_tool_for_sample(sample: VQASample) -> ToolDecision:
    tools = load_tools_from_demo()
    tmp_dir = ensure_single_image_case_dir(sample.image_path, pixel_size_mm=1.0)

    try:
        result1 = tools["run_csm_hc_tool"](tmp_dir)
        result2 = tools["run_nnunet_hc_tool"](tmp_dir)

        fname = os.path.basename(sample.image_path)
        r1 = result1.per_image.get(fname, {}) if getattr(result1, "per_image", None) else {}
        r2 = result2.per_image.get(fname, {}) if getattr(result2, "per_image", None) else {}

        csm_val = r1.get("hc_mm")
        nnunet_val = r2.get("hc_mm")

        pred_pixel = None
        note = "no valid HC output"

        if nnunet_val is not None:
            pred_pixel = float(nnunet_val)
            note = "use nnUNet HC output with pixel_size=1.0"
        elif csm_val is not None:
            pred_pixel = float(csm_val)
            note = "nnUNet unavailable, fallback to CSM with pixel_size=1.0"

        option_map = parse_numeric_options(sample.options)
        final_letter = pick_closest_option_letter(pred_pixel, option_map) if pred_pixel is not None else None
        final_text = None
        if final_letter is not None:
            final_text = next((x for x in sample.options if x.strip().upper().startswith(f"({final_letter})")), None)

        return ToolDecision(
            used_tool=True,
            tool_name="hc_pixel_via_csm_nnunet",
            tool_answer_letter=final_letter,
            tool_answer_text=final_text,
            tool_detail={
                "tool1_raw": r1,
                "tool2_raw": r2,
                "pred_pixel": pred_pixel,
                "option_map": option_map,
                "note": note,
            },
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def run_ga_trimester_multi_tool_for_sample(sample: VQASample) -> ToolDecision:
    tools = load_tools_from_demo()
    tmp_dir = ensure_single_image_case_dir(sample.image_path, pixel_size_mm=sample.pixel_size)

    try:
        result1 = tools["run_ga_algo1_tool"](tmp_dir)
        result2 = tools["run_ga_algo2_tool"](tmp_dir)
        result3 = tools["run_ga_algo3_tool"](tmp_dir)

        fname = os.path.basename(sample.image_path)
        r1 = result1.per_image.get(fname, {}) if getattr(result1, "per_image", None) else {}
        r2 = result2.per_image.get(fname, {}) if getattr(result2, "per_image", None) else {}
        r3 = result3.per_image.get(fname, {}) if getattr(result3, "per_image", None) else {}

        vals = []
        if r1.get("total_weeks") is not None:
            vals.append(("ga_algo1", float(r1["total_weeks"]), 1.0))
        if r2.get("total_weeks") is not None:
            vals.append(("ga_algo2", float(r2["total_weeks"]), 2.0))
        if r3.get("total_weeks") is not None:
            vals.append(("ga_algo3", float(r3["total_weeks"]), 1.0))

        recommended_total_weeks = None
        source_note = "no valid ga output"

        if len(vals) >= 2:
            best_pair = None
            best_diff = None
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    diff = abs(vals[i][1] - vals[j][1])
                    if best_diff is None or diff < best_diff:
                        best_diff = diff
                        best_pair = (vals[i], vals[j])

            if best_pair is not None and best_diff is not None and best_diff <= 1.5:
                v1, v2 = best_pair
                recommended_total_weeks = (v1[1] * v1[2] + v2[1] * v2[2]) / (v1[2] + v2[2])
                source_note = f"pair-vote agreement: {v1[0]} + {v2[0]}"
            else:
                wsum = sum(v[2] for v in vals)
                recommended_total_weeks = sum(v[1] * v[2] for v in vals) / wsum
                source_note = "weighted-mean fallback"
        elif len(vals) == 1:
            recommended_total_weeks = vals[0][1]
            source_note = f"single-tool fallback: {vals[0][0]}"

        if recommended_total_weeks is None:
            return ToolDecision(
                used_tool=True,
                tool_name="ga_trimester_multi_via_ga_tools",
                tool_answer_letter=None,
                tool_answer_text=None,
                tool_detail={
                    "tool1_raw": r1,
                    "tool2_raw": r2,
                    "tool3_raw": r3,
                    "recommended_total_weeks": None,
                    "predicted_trimester": None,
                    "note": "could not determine GA prediction",
                },
            )

        predicted_trimester = trimester_from_total_weeks(recommended_total_weeks)

        trimester_to_letter = {
            "first": "A",
            "second": "B",
            "third": "C",
        }

        final_letter = trimester_to_letter.get(predicted_trimester)
        final_text = TRIMESTER_OPTION_TO_TEXT.get(final_letter) if final_letter else None

        return ToolDecision(
            used_tool=True,
            tool_name="ga_trimester_multi_via_ga_tools",
            tool_answer_letter=final_letter,
            tool_answer_text=final_text,
            tool_detail={
                "tool1_raw": r1,
                "tool2_raw": r2,
                "tool3_raw": r3,
                "recommended_total_weeks": recommended_total_weeks,
                "predicted_trimester": predicted_trimester,
                "decision_source": source_note,
                "note": "A=First, B=Second, C=Third",
            },
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def run_aop_binary_tool_for_sample(sample: VQASample) -> ToolDecision:
    tools = load_tools_from_demo()
    tmp_dir = ensure_single_image_case_dir(sample.image_path, pixel_size_mm=sample.pixel_size)

    try:
        result1 = tools["run_aop_sam_tool"](tmp_dir)
        result2 = tools["run_usfm_aop_tool"](tmp_dir)
        result3 = tools["run_upernet_aop_tool"](tmp_dir)

        fname = os.path.basename(sample.image_path)
        r1 = result1.per_image.get(fname, {}) if getattr(result1, "per_image", None) else {}
        r2 = result2.per_image.get(fname, {}) if getattr(result2, "per_image", None) else {}
        r3 = result3.per_image.get(fname, {}) if getattr(result3, "per_image", None) else {}

        a1 = r1.get("aop_deg")
        a2 = r2.get("aop_deg")
        a3 = r3.get("aop_deg")

        p1 = r1.get("mask_path")
        p2 = r2.get("mask_path")
        p3 = r3.get("mask_path")

        recommended_aop = None
        recommended_mask_path = None
        source = "none"
        note = "no valid AoP output"

        if a1 is not None and a2 is not None and a3 is not None:
            vals = [
                ("tool1", float(a1), p1),
                ("tool2", float(a2), p2),
                ("tool3", float(a3), p3),
            ]
            sorted_vals = sorted([v for _, v, _ in vals])
            med = float(sorted_vals[1])

            if abs(float(a1) - med) >= 12.0:
                best = min(vals, key=lambda x: abs(x[1] - med))
                source = best[0]
                recommended_aop = best[1]
                recommended_mask_path = best[2]
                note = f"tool1 outlier, choose nearest-to-median tool ({source})"
            else:
                source = "tool1"
                recommended_aop = float(a1)
                recommended_mask_path = p1
                note = "tool1 not outlier, keep tool1"
        else:
            # fallback 顺序也尽量贴近 main.py
            if a1 is not None:
                source = "tool1"
                recommended_aop = float(a1)
                recommended_mask_path = p1
                note = "fallback to tool1"
            elif a2 is not None:
                source = "tool2"
                recommended_aop = float(a2)
                recommended_mask_path = p2
                note = "fallback to tool2"
            elif a3 is not None:
                source = "tool3"
                recommended_aop = float(a3)
                recommended_mask_path = p3
                note = "fallback to tool3"

        if recommended_aop is None:
            return ToolDecision(
                used_tool=True,
                tool_name="aop_binary_via_aop_tools",
                tool_answer_letter=None,
                tool_answer_text=None,
                tool_detail={
                    "tool1_raw": r1,
                    "tool2_raw": r2,
                    "tool3_raw": r3,
                    "recommended_aop_deg": None,
                    "source": source,
                    "note": note,
                },
            )

        final_letter = "A" if recommended_aop >= 120.0 else "B"
        final_text = (
            "AoP >= 120°, spontaneous vaginal delivery is indicated"
            if final_letter == "A"
            else "AoP < 120°, instrumental delivery or cesarean may be necessary"
        )

        return ToolDecision(
            used_tool=True,
            tool_name="aop_binary_via_aop_tools",
            tool_answer_letter=final_letter,
            tool_answer_text=final_text,
            tool_detail={
                "tool1_raw": r1,
                "tool2_raw": r2,
                "tool3_raw": r3,
                "recommended_aop_deg": recommended_aop,
                "recommended_mask_path": recommended_mask_path,
                "source": source,
                "threshold_deg": 120.0,
                "note": note,
            },
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def run_ac_pixel_tool_for_sample(sample: VQASample) -> ToolDecision:
    tools = load_tools_from_demo()
    tmp_dir = ensure_single_image_case_dir(sample.image_path, pixel_size_mm=1.0)

    try:
        result = tools["run_abdomen_fetalclip_samus_seg_tool"](tmp_dir)

        fname = os.path.basename(sample.image_path)
        r = result.per_image.get(fname, {}) if getattr(result, "per_image", None) else {}

        mask_path = r.get("mask_path")

        raw_img = _safe_load_pil(sample.image_path)
        mask_arr = _mask_to_raw_array(mask_path, raw_img, preprocess="resize_direct")

        pred_pixel = _ac_mm_from_mask_array(mask_arr, pixel_size_mm=1.0)
        note = "use abdomen_fetalclip_samus segmentation with pixel_size=1.0"

        if pred_pixel is None:
            result_fallback = tools["run_abdomen_fetalclip_seg_tool"](tmp_dir)
            r_fb = result_fallback.per_image.get(fname, {}) if getattr(result_fallback, "per_image", None) else {}
            mask_path_fb = r_fb.get("mask_path")
            mask_arr_fb = _mask_to_raw_array(mask_path_fb, raw_img, preprocess="resize_direct")
            pred_pixel = _ac_mm_from_mask_array(mask_arr_fb, pixel_size_mm=1.0)

            if pred_pixel is not None:
                mask_path = mask_path_fb
                note = "SAMUS unavailable, fallback to abdomen_fetalclip with pixel_size=1.0"

        option_map = parse_numeric_options(sample.options)
        final_letter = pick_closest_option_letter(pred_pixel, option_map) if pred_pixel is not None else None
        final_text = None
        if final_letter is not None:
            final_text = next(
                (x for x in sample.options if x.strip().upper().startswith(f"({final_letter})")),
                None
            )

        return ToolDecision(
            used_tool=True,
            tool_name="ac_pixel_via_abdomen_seg",
            tool_answer_letter=final_letter,
            tool_answer_text=final_text,
            tool_detail={
                "tool_raw": r,
                "mask_path": mask_path,
                "pred_pixel": pred_pixel,
                "option_map": option_map,
                "note": note,
            },
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def run_stomach_volume_pixel_tool_for_sample(sample: VQASample) -> ToolDecision:
    tools = load_tools_from_demo()
    tmp_dir = ensure_single_image_case_dir(sample.image_path, pixel_size_mm=1.0)

    try:
        result1 = tools["run_stomach_fetalclip_seg_tool"](tmp_dir)
        result2 = tools["run_stomach_fetalclip_samus_seg_tool"](tmp_dir)
        result3 = tools["run_stomach_nnunet_seg_tool"](tmp_dir)

        fname = os.path.basename(sample.image_path)
        r1 = result1.per_image.get(fname, {}) if getattr(result1, "per_image", None) else {}
        r2 = result2.per_image.get(fname, {}) if getattr(result2, "per_image", None) else {}
        r3 = result3.per_image.get(fname, {}) if getattr(result3, "per_image", None) else {}

        p1 = r1.get("mask_path")
        p2 = r2.get("mask_path")
        p3 = r3.get("mask_path")

        raw_img = _safe_load_pil(sample.image_path)

        # tool1 = pad_square
        # tool2 = resize_direct
        # tool3 = resize_direct
        m1 = _mask_to_raw_array(p1, raw_img, preprocess="pad_square")
        m2 = _mask_to_raw_array(p2, raw_img, preprocess="resize_direct")
        m3 = _mask_to_raw_array(p3, raw_img, preprocess="resize_direct")

        final_mask = None
        source = "none"
        note = "no valid stomach mask"

        total_pixels = float(raw_img.size[0] * raw_img.size[1]) if raw_img is not None else 1.0
        min_ratio = float(os.environ.get("AGENT_STOMACH_MIN_RATIO", "0.001"))

        base = _majority_vote_masks([m1, m2, m3])
        base_ratio = 0.0 if base is None else float(base.sum()) / total_pixels

        if base is not None and base_ratio >= min_ratio:
            final_mask = base
            source = "majority_vote"
            note = "majority-vote stomach mask passed min_ratio threshold"
        else:
            ratio3 = 0.0 if m3 is None else float(m3.sum()) / total_pixels
            ratio2 = 0.0 if m2 is None else float(m2.sum()) / total_pixels
            ratio1 = 0.0 if m1 is None else float(m1.sum()) / total_pixels

            if m3 is not None and ratio3 >= min_ratio:
                final_mask = m3
                source = "tool3"
                note = "fallback to stomach nnUNet mask"
            elif m2 is not None and ratio2 >= min_ratio:
                final_mask = m2
                source = "tool2"
                note = "fallback to stomach fetalclip+samus mask"
            elif m1 is not None and ratio1 >= min_ratio:
                final_mask = m1
                source = "tool1"
                note = "fallback to stomach fetalclip mask"
            else:
                if m3 is not None:
                    final_mask = m3
                    source = "tool3_default"
                    note = "all masks below threshold, default to tool3"
                elif m2 is not None:
                    final_mask = m2
                    source = "tool2_default"
                    note = "all masks below threshold, default to tool2"
                elif m1 is not None:
                    final_mask = m1
                    source = "tool1_default"
                    note = "all masks below threshold, default to tool1"

        pred_pixel = _stomach_area_pixel_from_mask_array(final_mask)

        option_map = parse_numeric_options(sample.options)
        final_letter = pick_closest_option_letter(pred_pixel, option_map) if pred_pixel is not None else None

        final_text = None
        if final_letter is not None:
            final_text = next(
                (x for x in sample.options if x.strip().upper().startswith(f"({final_letter})")),
                None
            )

        return ToolDecision(
            used_tool=True,
            tool_name="stomach_volume_pixel_via_seg_tools",
            tool_answer_letter=final_letter,
            tool_answer_text=final_text,
            tool_detail={
                "tool1_raw": r1,
                "tool2_raw": r2,
                "tool3_raw": r3,
                "tool1_mask_path": p1,
                "tool2_mask_path": p2,
                "tool3_mask_path": p3,
                "selected_source": source,
                "pred_pixel": pred_pixel,
                "option_map": option_map,
                "note": note,
            },
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
