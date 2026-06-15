"""Expert policy implementations.

Each ``run_*_expert`` here is the deterministic policy backing one
:class:`~fetusagents.general.experts.GeneralExpert` registered in
:mod:`~fetusagents.general.experts`. They take a ``case_dir`` (a
directory of image frames) plus the user query, invoke the relevant
CV tool runners, apply a fixed arbitration rule, and emit the standard
expert payload ``{"task", "algo_results", "expert_text"}``.

No LLM call happens inside an expert — the policy is fixed in code.
The orchestrator in :mod:`~fetusagents.general.pipeline` decides which
experts to invoke via the allocator's ``Forwarding to:`` line, and
then awaits each ``.run(case_dir, vignette)`` through the
``GeneralExpert.runner`` callable bound here.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Optional, Tuple

try:
    import cv2
except Exception:
    cv2 = None

from ..core.biometry import (
    _ac_mm_from_mask_array,
    _hadlock_ga_weeks_from_ac_mm,
    _hc_mm_from_mask_array,
    _round_1dp,
    float_weeks_to_weeks_days,
    hc_range_from_ga_weeks,
    parse_pixel_size_csv,
)
from ..core.image_utils import (
    _apply_postprocess,
    _compute_ellipse_residual,
    _load_mask_binary_cv2,
    _majority_voting,
    _mask_to_raw_array,
    _safe_load_pil,
)
from ..core.label_normalize import (
    normalize_brain_subplane,
    normalize_plane,
    title_brain_subplane,
)
from ..core.tool_runners import (
    _weighted_vote_ensemble_ga,
    run_abdomen_fetalclip_samus_seg_tool,
    run_aop_sam_tool,
    run_brain_subplane_fetalclip_tool,
    run_brain_subplane_resnet_tool,
    run_brain_subplane_vit_tool,
    run_csm_hc_tool,
    run_ga_algo1_tool,
    run_ga_algo2_tool,
    run_ga_algo3_tool,
    run_nnunet_hc_tool,
    run_plane_fetalclip_tool,
    run_plane_fulora_tool,
    run_stomach_fetalclip_samus_seg_tool,
    run_stomach_fetalclip_seg_tool,
    run_stomach_nnunet_seg_tool,
    run_upernet_aop_tool,
    run_usfm_aop_tool,
)


async def run_aop_expert(case_dir: str, vignette: str) -> Dict[str, Any]:
    """Run AoP expert with three tools and final outlier-median selection."""
    print(">>> [AoP] Running AoP-SAM tool...")
    result1 = run_aop_sam_tool(case_dir)
    
    print(">>> [AoP] Running USFM-AoP tool...")
    result2 = run_usfm_aop_tool(case_dir)

    print(">>> [AoP] Running UperNet-AoP tool...")
    result3 = run_upernet_aop_tool(case_dir)
    
    all_files = sorted(set(result1.per_image.keys()) | set(result2.per_image.keys()) | set(result3.per_image.keys()))
    final_struct: Dict[str, Any] = {}
    final_predictions: Dict[str, Dict[str, Any]] = {}
    for fname in all_files:
        r1 = result1.per_image.get(fname, {})
        r2 = result2.per_image.get(fname, {})
        r3 = result3.per_image.get(fname, {})
        a1 = r1.get("aop_deg")
        a2 = r2.get("aop_deg")
        a3 = r3.get("aop_deg")
        p1 = r1.get("mask_path")
        p2 = r2.get("mask_path")
        p3 = r3.get("mask_path")

        recommended = None
        recommended_mask_path = None
        source = "none"
        note = "No valid tool output"

        if a1 is not None and a2 is not None and a3 is not None:
            vals = [("tool1", float(a1), p1), ("tool2", float(a2), p2), ("tool3", float(a3), p3)]
            sorted_vals = sorted([v for _, v, _ in vals])
            med = float(sorted_vals[1])
            if abs(float(a1) - med) >= 12.0:
                best = min(vals, key=lambda x: abs(x[1] - med))
                source = best[0]
                recommended = best[1]
                recommended_mask_path = best[2]
                note = f"tool1 is outlier (|tool1-median|={abs(float(a1) - med):.2f} >= 12), choose median tool"
            else:
                source = "tool1"
                recommended = float(a1)
                recommended_mask_path = p1
                note = f"tool1 not outlier (|tool1-median|={abs(float(a1) - med):.2f} < 12), keep tool1"
        else:
            # Fallback order when 3-tool rule is not applicable.
            if a1 is not None:
                source = "tool1"
                recommended = float(a1)
                recommended_mask_path = p1
                note = "Fallback to tool1 (3-tool rule unavailable)"
            elif a2 is not None:
                source = "tool2"
                recommended = float(a2)
                recommended_mask_path = p2
                note = "Fallback to tool2 (tool1 unavailable)"
            elif a3 is not None:
                source = "tool3"
                recommended = float(a3)
                recommended_mask_path = p3
                note = "Fallback to tool3 (tool1/tool2 unavailable)"

        final_predictions[fname] = {
            "source": source,
            "recommended_aop_deg": recommended,
            "recommended_mask_path": recommended_mask_path,
            "note": note,
        }
        final_struct[fname] = {
            "recommended": recommended,
            "recommended_mask_path": recommended_mask_path,
            "decision_note": note,
        }

    text = json.dumps(
        {
            "task": "aop",
            "format_version": "1.0",
            "per_image": final_struct,
        },
        ensure_ascii=False,
    )
    
    return {
        "task": "aop",
        "algo_results": {
            "tool_1": {"name": result1.tool_name, "ok": result1.ok, "per_image": result1.per_image},
            "tool_2": {"name": result2.tool_name, "ok": result2.ok, "per_image": result2.per_image},
            "tool_3": {"name": result3.tool_name, "ok": result3.ok, "per_image": result3.per_image},
            "final_predictions": final_predictions,
        },
        "expert_text": text,
    }
    

async def run_hc_expert(case_dir: str, vignette: str) -> Dict[str, Any]:
    """Run HC expert with CSM + nnUNet and residual-based gating."""
    print(">>> [HC] Running CSM-HC tool...")
    result1 = run_csm_hc_tool(case_dir)
    
    print(">>> [HC] Running HC-nnUNet tool...")
    result2 = run_nnunet_hc_tool(case_dir)

    # Gating aligned with eval_hc_measurement:
    # default nnUNet, switch to CSM only when CSM agrees with nnUNet and has good ellipse residual.
    disagreement_threshold = float(os.environ.get("AGENT_HC_DISAGREEMENT_THRESHOLD", "0.03"))
    residual_threshold = float(os.environ.get("AGENT_HC_RESIDUAL_THRESHOLD", "8.0"))
    final_results: Dict[str, Dict[str, Any]] = {}
    all_files = sorted(set(result1.per_image.keys()) | set(result2.per_image.keys()))
    pixel_map = parse_pixel_size_csv(os.path.join(case_dir, "pixel_size.csv"))
    csm_recomputed_values: Dict[str, Optional[float]] = {}
    for fname in all_files:
        r1 = result1.per_image.get(fname, {})
        r2 = result2.per_image.get(fname, {})
        csm_mask_path = r1.get("mask_path")
        nn_mask_path = r2.get("mask_path")

        # Align with eval_hc_measurement.py:
        # - compute residual on native CSM mask
        # - compute HC from masks at original image resolution with Ramanujan-II
        orig_shape: Optional[Tuple[int, int]] = None
        if cv2 is not None:
            case_img = cv2.imread(os.path.join(case_dir, fname), cv2.IMREAD_GRAYSCALE)
            if case_img is not None:
                orig_shape = case_img.shape[:2]

        csm_mask_native = _load_mask_binary_cv2(csm_mask_path, target_shape=None)
        nn_mask_native = _load_mask_binary_cv2(nn_mask_path, target_shape=None)
        csm_mask_hc = _load_mask_binary_cv2(csm_mask_path, target_shape=orig_shape) if orig_shape else csm_mask_native
        nn_mask_hc = _load_mask_binary_cv2(nn_mask_path, target_shape=orig_shape) if orig_shape else nn_mask_native
        pixel_size = pixel_map.get(fname)

        csm_hc = _hc_mm_from_mask_array(csm_mask_hc, pixel_size)
        nn_hc = _hc_mm_from_mask_array(nn_mask_hc, pixel_size)
        csm_recomputed_values[fname] = csm_hc
        # Fallback to tool-emitted values if mask-based recomputation fails.
        if csm_hc is None:
            csm_hc = r1.get("hc_mm")
        if nn_hc is None:
            nn_hc = r2.get("hc_mm")

        csm_residual = _compute_ellipse_residual(csm_mask_native)

        hc_disagreement = float("nan")
        if csm_hc is not None and nn_hc is not None:
            denom = (float(csm_hc) + float(nn_hc)) / 2.0
            if denom != 0:
                hc_disagreement = abs(float(csm_hc) - float(nn_hc)) / denom

        if nn_hc is not None:
            use_csm_guard = (
                csm_hc is not None
                and not math.isnan(hc_disagreement)
                and hc_disagreement < disagreement_threshold
                and not math.isnan(csm_residual)
                and csm_residual < residual_threshold
            )
            if use_csm_guard:
                final_hc = csm_hc
                source = "csm_guard"
            else:
                final_hc = nn_hc
                source = "nnunet_default"
        elif csm_hc is not None:
            final_hc = csm_hc
            source = "csm_fallback"
        else:
            final_hc = None
            source = "none"

        final_results[fname] = {
            "csm_hc_mm": csm_hc,
            "nnunet_hc_mm": nn_hc,
            "csm_ellipse_residual": None if math.isnan(csm_residual) else csm_residual,
            "hc_disagreement": None if math.isnan(hc_disagreement) else hc_disagreement,
            "recommended_hc_mm": final_hc,
            "source": source,
            "csm_mask_path": csm_mask_path,
            "nnunet_mask_path": nn_mask_path,
        }

    # Print corrected CSM values from the aligned Ramanujan pipeline for transparency.
    print(">>> [HC] CSM recomputed (contour + Ramanujan) values:")
    for fname in all_files:
        val = csm_recomputed_values.get(fname)
        if val is None:
            print(f"[HC-CSM-Recomputed] {fname}: N/A")
        else:
            print(f"[HC-CSM-Recomputed] {fname}: {val:.2f} mm")

    structured: Dict[str, Any] = {}
    for fname in all_files:
        fr = final_results.get(fname, {})
        source = fr.get("source")
        csm = fr.get("csm_hc_mm")
        nnv = fr.get("nnunet_hc_mm")
        residual = fr.get("csm_ellipse_residual")
        disagreement = fr.get("hc_disagreement")
        note = "No valid tool output"
        if source == "csm_guard":
            if csm is not None and nnv is not None:
                diff = abs(float(csm) - float(nnv))
                note = f"CSM guard selected (disagreement={disagreement}, residual={residual}); abs diff={diff:.2f} mm"
            else:
                note = "CSM guard selected"
        elif source == "nnunet_default":
            note = "nnUNet default strategy"
        elif source == "csm_fallback":
            note = "nnUNet unavailable; CSM fallback"
        recommended_mask_path = None
        if source in ("nnunet_default",) and fr.get("nnunet_mask_path"):
            recommended_mask_path = fr.get("nnunet_mask_path")
        elif source in ("csm_guard", "csm_fallback") and fr.get("csm_mask_path"):
            recommended_mask_path = fr.get("csm_mask_path")
        structured[fname] = {
            "recommended": _round_1dp(fr.get("recommended_hc_mm")),
            "recommended_mask_path": recommended_mask_path,
            "decision_note": note,
        }

    text = json.dumps(
        {
            "task": "head_circumference",
            "format_version": "1.0",
            "per_image": structured,
            "decision_rule": {
                "type": "nn_with_csm_guard",
                "disagreement_threshold": disagreement_threshold,
                "csm_residual_threshold": residual_threshold,
            },
        },
        ensure_ascii=False,
    )
    
    return {
        "task": "head_circumference",
        "algo_results": {
            "tool_1": {"name": result1.tool_name, "ok": result1.ok, "per_image": result1.per_image},
            "tool_2": {"name": result2.tool_name, "ok": result2.ok, "per_image": result2.per_image},
            "final_hc": final_results,
            "gating_thresholds": {
                "disagreement": disagreement_threshold,
                "csm_residual": residual_threshold,
            },
        },
        "expert_text": text,
    }
    

async def run_ga_expert(
    case_dir: str,
    vignette: str,
    hc_algo_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run GA expert with 3 GA tools + weighted-vote ensemble and HC consistency check."""
    print(">>> [GA] Running GA-RadImageNet tool...")
    result1 = run_ga_algo1_tool(case_dir)
    
    print(">>> [GA] Running GA-FetalCLIP tool...")
    result2 = run_ga_algo2_tool(case_dir)

    print(">>> [GA] Running GA-ConvNeXt tool...")
    result3 = run_ga_algo3_tool(case_dir)

    # Optional HC reference from HC expert output
    hc_per_image_1: Dict[str, Dict[str, Any]] = {}
    hc_per_image_2: Dict[str, Dict[str, Any]] = {}
    if hc_algo_results:
        hc_per_image_1 = (hc_algo_results.get("tool_1") or {}).get("per_image", {}) or {}
        hc_per_image_2 = (hc_algo_results.get("tool_2") or {}).get("per_image", {}) or {}

    all_files = sorted(
        set(result1.per_image.keys())
        | set(result2.per_image.keys())
        | set(result3.per_image.keys())
        | set(hc_per_image_1.keys())
        | set(hc_per_image_2.keys())
    )

    crosscheck: Dict[str, Dict[str, Any]] = {}
    for fname in all_files:
        g1 = result1.per_image.get(fname, {})
        g2 = result2.per_image.get(fname, {})
        g3 = result3.per_image.get(fname, {})
        ga1 = float(g1.get("total_weeks")) if g1.get("total_weeks") is not None else None
        ga2 = float(g2.get("total_weeks")) if g2.get("total_weeks") is not None else None
        ga3 = float(g3.get("total_weeks")) if g3.get("total_weeks") is not None else None

        hc_vals: Dict[str, float] = {}
        if hc_per_image_1.get(fname, {}).get("hc_mm") is not None:
            try:
                hc_vals["hc_tool_1"] = float(hc_per_image_1[fname]["hc_mm"])
            except Exception:
                pass
        if hc_per_image_2.get(fname, {}).get("hc_mm") is not None:
            try:
                hc_vals["hc_tool_2"] = float(hc_per_image_2[fname]["hc_mm"])
            except Exception:
                pass
        hc_ref = (sum(hc_vals.values()) / len(hc_vals)) if hc_vals else None

        def _tool_check(ga_total: Optional[float]) -> Dict[str, Any]:
            if ga_total is None:
                return {"ga_weeks_total": None, "hc_range": None, "in_range": None}
            rng = hc_range_from_ga_weeks(ga_total)
            in_range = None if hc_ref is None else (rng["p2_5"] <= hc_ref <= rng["p97_5"])
            return {"ga_weeks_total": ga_total, "hc_range": rng, "in_range": in_range}

        check1 = _tool_check(ga1)
        check2 = _tool_check(ga2)
        check3 = _tool_check(ga3)

        ens_total, ens_source = _weighted_vote_ensemble_ga(ga1, ga2, ga3, tolerance=1.5)
        ens_weeks = ens_days = None
        ens_range = None
        ens_in_range = None
        if ens_total is not None:
            ens_weeks, ens_days = float_weeks_to_weeks_days(ens_total)
            ens_range = hc_range_from_ga_weeks(ens_total)
            ens_in_range = None if hc_ref is None else (ens_range["p2_5"] <= hc_ref <= ens_range["p97_5"])

        crosscheck[fname] = {
            "hc_values_mm": hc_vals,
            "hc_ref_mm": hc_ref,
            "algo1_check": check1,
            "algo2_check": check2,
            "algo3_check": check3,
            "recommended_ga": {
                "ga_weeks": ens_weeks,
                "ga_days": ens_days,
                "total_weeks": ens_total,
                "source": ens_source,
            },
            "hc_range_from_recommended_ga": ens_range,
            "hc_in_range_for_recommended_ga": ens_in_range,
        }

    structured: Dict[str, Any] = {}
    for fname in all_files:
        cx = crosscheck.get(fname, {})
        rec = cx.get("recommended_ga", {})
        source = str(rec.get("source") or "")
        if source.startswith("pair_vote_"):
            note_base = "Pair-vote agreement between tools"
        elif source == "weighted_mean_fallback":
            note_base = "No close pair; weighted-mean fallback"
        elif source.startswith("only_"):
            note_base = "Single-tool fallback"
        else:
            note_base = "No valid GA decision"
        in_range = cx.get("hc_in_range_for_recommended_ga")
        if in_range is False:
            note = f"{note_base}; HC cross-check out-of-range"
        elif in_range is True:
            note = f"{note_base}; HC cross-check in-range"
        else:
            note = f"{note_base}; HC cross-check unavailable"
        structured[fname] = {
            "recommended": {
                "weeks": rec.get("ga_weeks"),
                "days": rec.get("ga_days"),
            },
            "decision_note": note,
        }

    text = json.dumps(
        {
            "task": "gestational_age",
            "format_version": "1.0",
            "per_image": structured,
            "decision_rule": {
                "type": "weighted_vote_ensemble",
                "weights": {"tool1": 1.0, "tool2": 2.0, "tool3": 1.0},
                "tolerance_weeks": 1.5,
            },
        },
        ensure_ascii=False,
    )
    
    return {
        "task": "gestational_age",
        "algo_results": {
            "tool_1": {"name": result1.tool_name, "ok": result1.ok, "per_image": result1.per_image},
            "tool_2": {"name": result2.tool_name, "ok": result2.ok, "per_image": result2.per_image},
            "tool_3": {"name": result3.tool_name, "ok": result3.ok, "per_image": result3.per_image},
            "hc_crosscheck": crosscheck,
        },
        "expert_text": text,
    }


async def run_plane_expert(case_dir: str, vignette: str) -> Dict[str, Any]:
    """Run plane classification expert with two real tools."""
    print(">>> [Plane] Running Plane-FetalCLIP tool...")
    result1 = run_plane_fetalclip_tool(case_dir)
    
    print(">>> [Plane] Running Plane-FU-LoRA tool...")
    result2 = run_plane_fulora_tool(case_dir)

    all_files = sorted(set(result1.per_image.keys()) | set(result2.per_image.keys()))
    final_results: Dict[str, Dict[str, Any]] = {}
    structured: Dict[str, Any] = {}

    for fname in all_files:
        r1 = result1.per_image.get(fname, {})
        r2 = result2.per_image.get(fname, {})
        plane1 = normalize_plane(r1.get("plane", "other"), extended=True)
        plane2 = normalize_plane(r2.get("plane", "other"), extended=True)
        
        # Decision logic
        if plane1 == plane2:
            final = plane1
            note = "Both algorithms agree"
        elif plane2 == "other" and plane1 != "other":
            final = plane1
            note = "FU-LoRA returned 'other', using FetalCLIP"
        elif plane1 == "other" and plane2 != "other":
            final = plane2
            note = "FetalCLIP returned 'other', using FU-LoRA"
        else:
            final = plane1
            note = "Algorithms disagree, using FetalCLIP"
        
        final_results[fname] = {"plane1": plane1, "plane2": plane2, "final": final, "note": note}
        
        structured[fname] = {
            "recommended": final,
            "decision_note": note,
        }

    text = json.dumps(
        {
            "task": "plane_classification",
            "format_version": "1.0",
            "per_image": structured,
        },
        ensure_ascii=False,
    )
    
    return {
        "task": "plane_classification",
        "algo_results": {
            "tool_1": {"name": result1.tool_name, "ok": result1.ok, "per_image": result1.per_image},
            "tool_2": {"name": result2.tool_name, "ok": result2.ok, "per_image": result2.per_image},
            "final_classifications": final_results,
        },
        "expert_text": text,
    }
    

async def run_brain_subplane_expert(case_dir: str, vignette: str) -> Dict[str, Any]:
    """Run brain subplane expert with three tools (FetalCLIP, ResNet, ViT)."""
    print(">>> [BrainSubplanes] Running BrainSubplane-FetalCLIP tool...")
    result1 = run_brain_subplane_fetalclip_tool(case_dir)

    print(">>> [BrainSubplanes] Running BrainSubplane-ResNet tool...")
    result2 = run_brain_subplane_resnet_tool(case_dir)

    print(">>> [BrainSubplanes] Running BrainSubplane-ViT tool...")
    result3 = run_brain_subplane_vit_tool(case_dir)

    def norm_label(x: Any) -> str:
        # Preserve the original "empty stays empty, unknowns stay raw"
        # semantics that the partial-outputs vote logic below relies on,
        # but route the canonical branches through the shared helper so
        # the bucket vocabulary stays in lock-step with specific/.
        s = str(x).strip() if x is not None else ""
        if not s:
            return ""
        canonical = normalize_brain_subplane(s)
        if canonical == "other":
            return s
        return title_brain_subplane(canonical)

    final_results: Dict[str, Dict[str, Any]] = {}
    structured: Dict[str, Any] = {}
    all_files = sorted(set(result1.per_image.keys()) | set(result2.per_image.keys()) | set(result3.per_image.keys()))
    for fname in all_files:
        r1 = result1.per_image.get(fname, {})
        r2 = result2.per_image.get(fname, {})
        r3 = result3.per_image.get(fname, {})
        l1 = norm_label(r1.get("subplane"))
        l2 = norm_label(r2.get("subplane"))
        l3 = norm_label(r3.get("subplane"))

        labels = [x for x in [l1, l2, l3] if x]
        final = "N/A"
        note = "No result from any tool"
        if l1 and l2 and l3 and l1 == l2 == l3:
            final = l1
            note = "All three tools agree"
        elif l1 and l2 and l1 == l2:
            final = l1
            note = "Majority: FetalCLIP + ResNet"
        elif l1 and l3 and l1 == l3:
            final = l1
            note = "Majority: FetalCLIP + ViT"
        elif l2 and l3 and l2 == l3:
            final = l2
            note = "Majority: ResNet + ViT"
        elif l1 and l2 and l3 and (l1 != l2 and l1 != l3 and l2 != l3):
            final = l1
            note = "All disagree; default to FetalCLIP"
        elif labels:
            final = labels[0]
            note = "Partial outputs only; using first available prediction"

        final_results[fname] = {
            "tool1": l1 or None,
            "tool2": l2 or None,
            "tool3": l3 or None,
            "final": final,
            "note": note,
        }

        structured[fname] = {
            "recommended": final,
            "decision_note": note,
        }

    text = json.dumps(
        {
            "task": "brain_subplanes",
            "format_version": "1.0",
            "per_image": structured,
            "decision_rule": "majority_vote_3_tools_else_tool1",
        },
        ensure_ascii=False,
    )
    
    return {
        "task": "brain_subplanes",
        "algo_results": {
            "tool_1": {"name": result1.tool_name, "ok": result1.ok, "per_image": result1.per_image},
            "tool_2": {"name": result2.tool_name, "ok": result2.ok, "per_image": result2.per_image},
            "tool_3": {"name": result3.tool_name, "ok": result3.ok, "per_image": result3.per_image},
            "final_classifications": final_results,
        },
        "expert_text": text,
    }
    

async def run_stomach_seg_expert(case_dir: str, vignette: str) -> Dict[str, Any]:
    """Run stomach segmentation with 3 tools + tiered fallback shape-prior decision."""
    print(">>> [StomachSeg] Running StomachSeg-FetalCLIP tool...")
    result1 = run_stomach_fetalclip_seg_tool(case_dir)
    print(">>> [StomachSeg] Running StomachSeg-FetalCLIP+SAMUS tool...")
    result2 = run_stomach_fetalclip_samus_seg_tool(case_dir)
    print(">>> [StomachSeg] Running StomachSeg-nnUNet tool...")
    result3 = run_stomach_nnunet_seg_tool(case_dir)

    min_ratio = float(os.environ.get("AGENT_STOMACH_MIN_RATIO", "0.001"))
    min_area = int(os.environ.get("AGENT_STOMACH_MIN_AREA", "50"))

    final_results: Dict[str, Dict[str, Any]] = {}
    all_files = sorted(set(result1.per_image.keys()) | set(result2.per_image.keys()) | set(result3.per_image.keys()))

    for idx, fname in enumerate(all_files):
        r1 = result1.per_image.get(fname, {})
        r2 = result2.per_image.get(fname, {})
        r3 = result3.per_image.get(fname, {})
        p1 = r1.get("mask_path")
        p2 = r2.get("mask_path")
        p3 = r3.get("mask_path")

        raw_path = os.path.join(case_dir, fname)
        raw_img = _safe_load_pil(raw_path)
        m1 = _mask_to_raw_array(p1, raw_img, preprocess="pad_square")
        m2 = _mask_to_raw_array(p2, raw_img, preprocess="resize_direct")
        m3 = _mask_to_raw_array(p3, raw_img, preprocess="resize_direct")
        available = [m for m in [m1, m2, m3] if m is not None]

        final_mask_path = None
        decision_note = "no_mask_available"
        total_pixels = float(raw_img.size[0] * raw_img.size[1]) if raw_img is not None else 1.0
        if raw_img is not None and available:
            base = _majority_voting(available)
            base = _apply_postprocess(base, min_area=min_area)
            base_ratio = 0.0 if base is None else float(base.sum()) / total_pixels
            if base is not None and base_ratio >= min_ratio:
                final_mask_path = p2 or p3 or p1
                decision_note = "majority_minpass"
        if final_mask_path is None:
            r3_ratio = 0.0 if m3 is None else float(m3.sum()) / total_pixels
            r2_ratio = 0.0 if m2 is None else float(m2.sum()) / total_pixels
            r1_ratio = 0.0 if m1 is None else float(m1.sum()) / total_pixels
            if p3 and r3_ratio >= min_ratio:
                final_mask_path = p3
                decision_note = "fallback_tool3"
            elif p2 and r2_ratio >= min_ratio:
                final_mask_path = p2
                decision_note = "fallback_tool2"
            elif p1 and r1_ratio >= min_ratio:
                final_mask_path = p1
                decision_note = "fallback_tool1"
            else:
                final_mask_path = p3 or p2 or p1
                decision_note = "fallback_tool3_default"

        final_results[fname] = {
            "tool1": p1,
            "tool2": p2,
            "tool3": p3,
            "recommended_mask": final_mask_path,
            "note": decision_note,
        }

    structured: Dict[str, Any] = {}
    for fname in all_files:
        fr = final_results.get(fname, {})
        note = str(fr.get("note") or "")
        if note == "majority_minpass":
            cnote = "Majority mask passed shape-prior threshold"
        elif note.startswith("fallback_tool3"):
            cnote = "Fallback to tool3 per tiered rule"
        elif note == "fallback_tool2":
            cnote = "Fallback to tool2 per tiered rule"
        elif note == "fallback_tool1":
            cnote = "Fallback to tool1 per tiered rule"
        else:
            cnote = "Limited/invalid masks; fallback path used"
        structured[fname] = {
            "recommended": fr.get("recommended_mask"),
            "decision_note": cnote,
        }

    text = json.dumps(
        {
            "task": "stomach_segmentation",
            "format_version": "1.0",
            "per_image": structured,
            "decision_rule": {
                "type": "tiered_fallback_shape_prior",
                "min_ratio": min_ratio,
                "min_area": min_area,
                "fallback_order": ["tool3", "tool2", "tool1"],
            },
        },
        ensure_ascii=False,
    )

    return {
        "task": "stomach_segmentation",
        "algo_results": {
            "tool_1": {"name": result1.tool_name, "ok": result1.ok, "per_image": result1.per_image},
            "tool_2": {"name": result2.tool_name, "ok": result2.ok, "per_image": result2.per_image},
            "tool_3": {"name": result3.tool_name, "ok": result3.ok, "per_image": result3.per_image},
            "final_segmentations": final_results,
            "decision_rule": {
                "type": "tiered_fallback_shape_prior",
                "min_ratio": min_ratio,
                "min_area": min_area,
                "fallback_order": ["tool3", "tool2", "tool1"],
            },
        },
        "expert_text": text,
    }


async def run_abdomen_seg_expert(case_dir: str, vignette: str) -> Dict[str, Any]:
    """Run abdomen segmentation with tool2 only and compute AC from final masks."""
    print(">>> [AbdomenSeg] Running AbdomenSeg-FetalCLIP+SAMUS tool...")
    result2 = run_abdomen_fetalclip_samus_seg_tool(case_dir)

    # Build image set from both case_dir and tool outputs so missing predictions are visible.
    case_files = [
        n for n in sorted(os.listdir(case_dir))
        if n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"))
    ]
    all_files = sorted(set(case_files) | set(result2.per_image.keys()))
    pixel_map = parse_pixel_size_csv(os.path.join(case_dir, "pixel_size.csv"))

    final_results: Dict[str, Dict[str, Any]] = {}
    structured: Dict[str, Any] = {}
    for fname in all_files:
        p2 = (result2.per_image.get(fname) or {}).get("mask_path")
        raw_img = _safe_load_pil(os.path.join(case_dir, fname))
        mask_arr = _mask_to_raw_array(p2, raw_img, preprocess="resize_direct")
        ac_mm = _ac_mm_from_mask_array(mask_arr, pixel_map.get(fname))
        ga_weeks_hadlock = _hadlock_ga_weeks_from_ac_mm(ac_mm)

        final_results[fname] = {
            "tool2_mask_path": p2,
            "ac_mm": ac_mm,
            "ga_weeks_hadlock": ga_weeks_hadlock,
            "pixel_size_mm": pixel_map.get(fname),
            "decision_note": "direct_tool2",
        }

        if p2 and ac_mm is not None and ga_weeks_hadlock is not None:
            cnote = "Directly adopted tool2 mask; AC computed from contour-fitted ellipse; Hadlock GA derived from AC"
        elif p2 and ac_mm is not None:
            cnote = "Directly adopted tool2 mask; AC computed from contour-fitted ellipse; Hadlock GA unavailable"
        elif p2:
            cnote = "Directly adopted tool2 mask; AC unavailable (pixel size or ellipse fit issue)"
        else:
            cnote = "Tool2 mask unavailable"

        structured[fname] = {
            "recommended_mask_path": p2,
            "recommended_ac_mm": _round_1dp(ac_mm),
            "recommended_ga_weeks_from_ac": _round_1dp(ga_weeks_hadlock),
            "decision_note": cnote,
        }

    text = json.dumps(
        {
            "task": "abdomen_segmentation",
            "format_version": "1.0",
            "per_image": structured,
            "decision_rule": {"type": "single_tool_direct", "tool": "AbdomenSeg-FetalCLIP+SAMUS"},
        },
        ensure_ascii=False,
    )
    
    return {
        "task": "abdomen_segmentation",
        "algo_results": {
            "tool_2": {"name": result2.tool_name, "ok": result2.ok, "per_image": result2.per_image},
            "final_segmentations": final_results,
        },
        "expert_text": text,
    }


