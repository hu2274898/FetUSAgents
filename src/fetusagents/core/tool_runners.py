"""All 17 subprocess-based CV tool wrappers.

Each ``run_*_tool`` function takes a ``case_dir`` and (optionally) a
``ToolConfig``, launches the matching external Python script through
``run_tool_subprocess``, and returns a :class:`ToolResult` with
``per_image`` predictions and trimmed stdout/stderr logs. None of these
functions interact with LLMs; they are pure "model in subprocess" calls.

Also includes :func:`_weighted_vote_ensemble_ga`, the GA tool-vote
fusion helper.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:
    np = None

try:
    import cv2
except Exception:
    cv2 = None

from ._state import TOOL_CONFIG, ToolConfig, _FNAME_EXT_RE
from .llm import ToolResult, run_tool_subprocess
from .biometry import (
    ensure_pixel_csv,
    parse_pixel_size_csv,
    _hc_mm_from_mask_array,
)
from .image_utils import (
    _agent_outputs_dir,
    _mask_to_raw_array,
    _safe_load_pil,
)
from .parsing import (
    _parse_filename_colon_value,
    _parse_filename_label_probs,
    _parse_filename_colon_text,
    _normalize_video_plane_label,
)


def run_aop_sam_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    """Run AoP-SAM on images in case_dir."""
    script = os.path.join(config.agent_tools_dir, "aop_sam_step2_predict_agent.py")
    out_dir = _agent_outputs_dir("aop", "aop_sam_step2", case_dir)
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    result = run_tool_subprocess(
        python_path=config.hxt_base_python,
        script_path=script,
        args=[
            "--data_path",
            case_dir,
            "--sam_ckpt",
            config.aop_sam_ckpt,
            "--out_dir",
            out_dir,
            "--gpu",
            gpu_id,
        ],
        cwd=config.agent_tools_dir,
        timeout=config.default_timeout,
        log_prefix="AoP-SAM-step2",
        print_regexes=[r"\.png:\s*[\d.]+\s*deg"],
    )
    
    per_image = {}
    if result["ok"]:
        # Parse output: "filename.png: 123.45 deg | mask: /path/to/mask.png"
        for line in result["stdout"].splitlines():
            match = re.search(
                r"([^\s:]+\.(?:png|jpg|jpeg|bmp|tif|tiff|webp))\s*:\s*([-+]?\d*\.?\d+)\s*deg(?:\s*\|\s*mask:\s*(.+))?",
                line,
                flags=re.IGNORECASE,
            )
            if match:
                fname = match.group(1)
                aop = float(match.group(2))
                mask_path = (match.group(3) or "").strip() or None
                per_image[fname] = {"aop_deg": aop, "mask_path": mask_path}
    
    return ToolResult(
        tool_name="AoP-SAM",
        ok=result["ok"] and len(per_image) > 0,
        per_image=per_image,
        error=result.get("error") or (None if result["ok"] else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


def run_usfm_aop_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    """Run USFM AoP on images in case_dir."""
    result = run_tool_subprocess(
        python_path=config.usfm_python,
        script_path="predict_agent.py",
        args=["--data_path", case_dir],
        cwd=config.usfm_aop_dir,
        timeout=config.default_timeout,
        log_prefix="USFM-AoP",
        print_regexes=[r"\.png:\s*[\d.]+\s*deg"],
    )
    
    per_image = {}
    if result["ok"]:
        for line in result["stdout"].splitlines():
            match = re.search(
                r"([^\s:]+\.(?:png|jpg|jpeg|bmp|tif|tiff|webp))\s*:\s*([-+]?\d*\.?\d+)\s*deg(?:\s*\|\s*mask:\s*(.+))?",
                line,
                flags=re.IGNORECASE,
            )
            if match:
                fname = match.group(1)
                aop = float(match.group(2))
                mask_path = (match.group(3) or "").strip() or None
                per_image[fname] = {"aop_deg": aop, "mask_path": mask_path}
    
    return ToolResult(
        tool_name="USFM-AoP",
        ok=result["ok"] and len(per_image) > 0,
        per_image=per_image,
        error=result.get("error") or (None if result["ok"] else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


def run_upernet_aop_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    """Run UperNet AoP on images in case_dir."""
    script = os.path.join(config.agent_tools_dir, "upernet_aop_predict_agent.py")
    out_dir = _agent_outputs_dir("aop", "upernet", case_dir)
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    result = run_tool_subprocess(
        python_path=config.hxt_base_python,
        script_path=script,
        args=[
            "--data_path",
            case_dir,
            "--ckpt_path",
            config.upernet_aop_ckpt,
            "--out_dir",
            out_dir,
            "--gpu",
            gpu_id,
        ],
        cwd=config.agent_tools_dir,
        timeout=config.default_timeout,
        log_prefix="UperNet-AoP",
        print_regexes=[r"\.png:\s*[\d.]+\s*deg"],
    )

    per_image = {}
    if result["ok"]:
        for line in result["stdout"].splitlines():
            match = re.search(
                r"([^\s:]+\.(?:png|jpg|jpeg|bmp|tif|tiff|webp))\s*:\s*([-+]?\d*\.?\d+)\s*deg(?:\s*\|\s*mask:\s*(.+))?",
                line,
                flags=re.IGNORECASE,
            )
            if match:
                fname = match.group(1)
                aop = float(match.group(2))
                mask_path = (match.group(3) or "").strip() or None
                per_image[fname] = {"aop_deg": aop, "mask_path": mask_path}

    return ToolResult(
        tool_name="UperNet-AoP",
        ok=result["ok"] and len(per_image) > 0,
        per_image=per_image,
        error=result.get("error") or (None if result["ok"] else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


# HC Tools
def run_csm_hc_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    """Run CSM HC measurement on images in case_dir."""
    pixel_csv = ensure_pixel_csv(case_dir)
    out_dir = _agent_outputs_dir("head_circumference", "csm", case_dir)
    
    result = run_tool_subprocess(
        python_path=config.experiment_aaai_python,
        script_path="predict_agent.py",
        args=["--data_path", case_dir, "--pixel_csv", pixel_csv, "--output_dir", out_dir],
        cwd=config.csm_hc_dir,
        timeout=config.default_timeout,
        log_prefix="CSM-HC",
        # Suppress tool-emitted HC values (they use a different formula than our final pipeline).
        print_regexes=[r"^\bTHIS_REGEX_SHOULD_NOT_MATCH\b$"],
    )
    
    per_image = {}
    if result["ok"]:
        for line in result["stdout"].splitlines():
            match = re.search(r"([^\s:]+\.png)\s*:\s*([\d.]+)\s*mm", line)
            if match:
                fname = match.group(1)
                hc = float(match.group(2))
                mask_path = os.path.join(out_dir, "predictions", fname)
                per_image[fname] = {"hc_mm": hc, "mask_path": mask_path if os.path.exists(mask_path) else None}
    
    return ToolResult(
        tool_name="CSM-HC",
        ok=result["ok"] and len(per_image) > 0,
        per_image=per_image,
        error=result.get("error") or (None if result["ok"] else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


def run_usfm_hc_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    """Run USFM HC on images in case_dir."""
    pixel_csv = ensure_pixel_csv(case_dir)
    
    result = run_tool_subprocess(
        python_path=config.usfm_python,
        script_path="predict_agent.py",
        args=["--data_path", case_dir, "--pixel_csv", pixel_csv],
        cwd=config.usfm_hc_dir,
        timeout=config.default_timeout,
        log_prefix="USFM-HC",
        print_regexes=[r"\.png:\s*[\d.]+\s*mm"],
    )
    
    per_image = {}
    if result["ok"]:
        for line in result["stdout"].splitlines():
            match = re.search(r"([^\s:]+\.png)\s*:\s*([\d.]+)\s*mm", line)
            if match:
                fname = match.group(1)
                hc = float(match.group(2))
                per_image[fname] = {"hc_mm": hc}
    
    return ToolResult(
        tool_name="USFM-HC",
        ok=result["ok"] and len(per_image) > 0,
        per_image=per_image,
        error=result.get("error") or (None if result["ok"] else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


def run_nnunet_hc_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    """Run nnUNet HC segmentation and derive HC(mm) from predicted mask."""
    script = os.path.join(config.agent_tools_dir, "nnunet_hc_seg_predict_agent.py")
    out_dir = _agent_outputs_dir("head_circumference", "nnunet", case_dir)
    result = run_tool_subprocess(
        python_path=config.hxt_base_python,
        script_path=script,
        args=[
            "--data_path",
            case_dir,
            "--out_dir",
            out_dir,
            "--nnunet_predict",
            config.nnunet_predict,
            "--timeout",
            str(config.default_timeout),
        ],
        cwd=config.agent_tools_dir,
        timeout=config.default_timeout + 120,
        log_prefix="HC-nnUNet",
        print_regexes=[rf"\.{_FNAME_EXT_RE}\s*:\s*"],
    )

    per_image: Dict[str, Dict[str, Any]] = {}
    if result["ok"]:
        mask_paths = _parse_filename_colon_value(result["stdout"])
        pixel_map = parse_pixel_size_csv(os.path.join(case_dir, "pixel_size.csv"))
        for fname, mask_path in mask_paths.items():
            raw_img = _safe_load_pil(os.path.join(case_dir, fname))
            mask_arr = _mask_to_raw_array(mask_path, raw_img, preprocess="resize_direct")
            hc_mm = _hc_mm_from_mask_array(mask_arr, pixel_map.get(fname))
            per_image[fname] = {"mask_path": mask_path, "hc_mm": hc_mm}

    ok = result["ok"] and len(per_image) > 0
    return ToolResult(
        tool_name="HC-nnUNet",
        ok=ok,
        per_image=per_image,
        error=result.get("error") or (None if ok else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


# GA Tools
def run_ga_algo1_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    """Run GA Algorithm 1 (RadImageNet-based) on images in case_dir."""
    pixel_csv = ensure_pixel_csv(case_dir)
    
    result = run_tool_subprocess(
        python_path=config.hxt_base_python,
        script_path="predict_agent.py",
        args=["--img_dir", case_dir, "--pixel_csv", pixel_csv],
        cwd=config.ga_algo1_dir,
        timeout=config.default_timeout,
        log_prefix="GA-RadImageNet",
        print_regexes=[r"\.png:\s*[-+]?(?:\d+\.?\d*|\.\d+)\s*$"],
    )
    
    per_image = {}
    if result["ok"]:
        # Parse: "filename.png: 25.1234"
        for line in result["stdout"].splitlines():
            match = re.search(r"([^\s:]+\.png)\s*:\s*([\d.]+)", line)
            if match:
                fname = match.group(1)
                ga_weeks = float(match.group(2))
                weeks = int(ga_weeks)
                days = int((ga_weeks - weeks) * 7)
                per_image[fname] = {"ga_weeks": weeks, "ga_days": days, "total_weeks": ga_weeks}
    
    return ToolResult(
        tool_name="GA-RadImageNet",
        ok=result["ok"] and len(per_image) > 0,
        per_image=per_image,
        error=result.get("error") or (None if result["ok"] else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


def run_ga_algo2_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    """Run GA Algorithm 2 (FetalCLIP-based) on images in case_dir."""
    pixel_csv = ensure_pixel_csv(case_dir)
    
    result = run_tool_subprocess(
        python_path=config.fetalclip_python,
        script_path="predict_agent.py",
        args=["--img_dir", case_dir, "--pixel_csv", pixel_csv],
        cwd=config.ga_algo2_dir,
        timeout=config.default_timeout,
        log_prefix="GA-FetalCLIP",
        print_regexes=[r"^\[[^\]]+\.png\]\s+Predicted GA"],
    )
    
    per_image = {}
    if result["ok"]:
        # Parse: "[filename.png] Predicted GA ≈ 25 weeks + 3 days (25.4286 days total)"
        for line in result["stdout"].splitlines():
            match = re.search(
                r"\[([^\]]+\.png)\]\s+Predicted GA[^0-9]*(\d+)\s+weeks?\s*\+\s*(\d+)\s+days?.*\(([\d.]+)",
                line
            )
            if match:
                fname = match.group(1)
                weeks = int(match.group(2))
                days = int(match.group(3))
                total = float(match.group(4))
                per_image[fname] = {"ga_weeks": weeks, "ga_days": days, "total_weeks": total}
    
    return ToolResult(
        tool_name="GA-FetalCLIP",
        ok=result["ok"] and len(per_image) > 0,
        per_image=per_image,
        error=result.get("error") or (None if result["ok"] else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


def run_ga_algo3_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    """Run GA Algorithm 3 (ConvNeXt-based) on images in case_dir."""
    pixel_csv = ensure_pixel_csv(case_dir)

    result = run_tool_subprocess(
        python_path=config.hxt_base_python,
        script_path="predict_agent.py",
        args=["--img_dir", case_dir, "--pixel_csv", pixel_csv],
        cwd=config.ga_algo3_dir,
        timeout=config.default_timeout,
        log_prefix="GA-ConvNeXt",
        print_regexes=[r"\.png:\s*[-+]?(?:\d+\.?\d*|\.\d+)\s*$"],
    )

    per_image = {}
    if result["ok"]:
        for line in result["stdout"].splitlines():
            match = re.search(r"([^\s:]+\.png)\s*:\s*([\d.]+)", line)
            if match:
                fname = match.group(1)
                ga_weeks = float(match.group(2))
                weeks = int(ga_weeks)
                days = int((ga_weeks - weeks) * 7)
                per_image[fname] = {"ga_weeks": weeks, "ga_days": days, "total_weeks": ga_weeks}

    return ToolResult(
        tool_name="GA-ConvNeXt",
        ok=result["ok"] and len(per_image) > 0,
        per_image=per_image,
        error=result.get("error") or (None if result["ok"] else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


# Plane Classification Tools
def run_plane_fetalclip_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    """Run FetalCLIP plane classification on images in case_dir."""
    result = run_tool_subprocess(
        python_path=config.fetalclip_python,
        script_path="predict_agent.py",
        args=["--data_path", case_dir],
        cwd=config.plane_fetalclip_dir,
        timeout=config.default_timeout,
        log_prefix="Plane-FetalCLIP",
        print_regexes=[rf"\.{_FNAME_EXT_RE}\s*→\s*Predicted plane:"],
    )
    
    per_image = {}
    if result["ok"]:
        # Parse:
        #   filename.png → Predicted plane: brain | probs: abdomen:0.1,brain:0.8,...
        # or fallback without probs.
        for line in result["stdout"].splitlines():
            match = re.search(rf"([^\s→]+\.{_FNAME_EXT_RE})\s*→\s*Predicted plane:\s*([^|]+)\|\s*probs:\s*(.+)", line, flags=re.IGNORECASE)
            if match:
                fname = match.group(1).strip()
                plane = match.group(2).strip()
                prob_str = match.group(3).strip()
                probs: Dict[str, float] = {}
                for kv in prob_str.split(","):
                    if ":" in kv:
                        k, v = kv.split(":", 1)
                        try:
                            probs[k.strip()] = float(v.strip())
                        except Exception:
                            pass
                per_image[fname] = {"plane": plane, "probs": probs}
                continue
            match2 = re.search(rf"([^\s→]+\.{_FNAME_EXT_RE})\s*→\s*Predicted plane:\s*(.+)", line, flags=re.IGNORECASE)
            if match2:
                fname = match2.group(1).strip()
                plane = match2.group(2).strip()
                per_image[fname] = {"plane": plane, "probs": {}}
    
    return ToolResult(
        tool_name="Plane-FetalCLIP",
        ok=result["ok"] and len(per_image) > 0,
        per_image=per_image,
        error=result.get("error") or (None if result["ok"] else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


def run_plane_fulora_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    """Run FU-LoRA plane classification on images in case_dir."""
    result = run_tool_subprocess(
        python_path=config.experiment_aaai_python,
        script_path="predict_agent.py",
        args=["--data_path", case_dir],
        cwd=config.plane_fulora_dir,
        timeout=config.default_timeout,
        log_prefix="Plane-FU-LoRA",
        print_regexes=[rf"\.{_FNAME_EXT_RE}\s*:\s*"],
    )
    
    per_image = {}
    if result["ok"]:
        # Parse:
        #   filename.png: Fetal brain | probs: Other:0.01,Fetal abdomen:0.02,...
        # or fallback without probs.
        for line in result["stdout"].splitlines():
            match = re.search(rf"([^\s:]+\.{_FNAME_EXT_RE})\s*:\s*([^|]+)\|\s*probs:\s*(.+)", line, flags=re.IGNORECASE)
            if match:
                fname = match.group(1).strip()
                plane = match.group(2).strip()
                prob_str = match.group(3).strip()
                probs: Dict[str, float] = {}
                for kv in prob_str.split(","):
                    if ":" in kv:
                        parts = kv.rsplit(":", 1)
                        if len(parts) == 2:
                            k, v = parts
                            try:
                                probs[k.strip()] = float(v.strip())
                            except Exception:
                                pass
                per_image[fname] = {"plane": plane, "probs": probs}
                continue
            match2 = re.search(rf"([^\s:]+\.{_FNAME_EXT_RE})\s*:\s*(.+)", line, flags=re.IGNORECASE)
            if match2:
                fname = match2.group(1).strip()
                plane = match2.group(2).strip()
                per_image[fname] = {"plane": plane, "probs": {}}
    
    return ToolResult(
        tool_name="Plane-FU-LoRA",
        ok=result["ok"] and len(per_image) > 0,
        per_image=per_image,
        error=result.get("error") or (None if result["ok"] else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


# New Tools: Brain subplane / Stomach seg / Abdomen seg
def run_video_keyframe_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    """Run key-frame + 6-plane classifier in non-interactive agent mode."""
    script = os.path.join(config.agent_tools_dir, "video_keyframe_cls6_predict_agent.py")
    out_dir = _agent_outputs_dir("video_summary", "keyframe_cls6", case_dir)
    result = run_tool_subprocess(
        python_path=config.fetalclip2_python,
        script_path=script,
        args=[
            "--data_path",
            case_dir,
            "--test_script",
            os.path.join(config.keyframe_cls6_dir, "test.py"),
            "--config",
            config.keyframe_cls6_config,
            "--output_csv",
            os.path.join(out_dir, "predictions.csv"),
        ],
        cwd=config.agent_tools_dir,
        timeout=config.default_timeout,
        log_prefix="VideoKeyFrame-Cls6",
        print_regexes=[rf"\.{_FNAME_EXT_RE}\s*:\s*"],
    )
    per_image: Dict[str, Dict[str, Any]] = {}
    if result["ok"]:
        kv = _parse_filename_colon_text(result["stdout"])
        for fname, label in kv.items():
            norm = _normalize_video_plane_label(label)
            per_image[fname] = {
                "pred_plane_raw": label,
                "pred_plane_norm": norm,
                "is_key_frame": norm != "other",
            }
    ok = result["ok"] and len(per_image) > 0
    return ToolResult(
        tool_name="VideoKeyFrame-Cls6",
        ok=ok,
        per_image=per_image,
        error=result.get("error") or (None if ok else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
        artifacts_dir=out_dir,
    )


def _weighted_vote_ensemble_ga(
    ga1: Optional[float],
    ga2: Optional[float],
    ga3: Optional[float],
    tolerance: float = 1.5,
) -> Tuple[Optional[float], str]:
    """
    Weighted vote ensemble (tool2 weight=2.0, tool1/tool3=1.0).
    Returns (final_ga_weeks, source_tag).
    """
    weights = {"tool1": 1.0, "tool2": 2.0, "tool3": 1.0}
    preds: Dict[str, float] = {}
    if ga1 is not None:
        preds["tool1"] = float(ga1)
    if ga2 is not None:
        preds["tool2"] = float(ga2)
    if ga3 is not None:
        preds["tool3"] = float(ga3)
    if not preds:
        return None, "none"
    if len(preds) == 1:
        k, v = next(iter(preds.items()))
        return float(v), f"only_{k}"

    pred_list = list(preds.items())
    best_agreement: Optional[float] = None
    best_weight_sum = 0.0
    best_pair: Optional[str] = None
    for i in range(len(pred_list)):
        for j in range(i + 1, len(pred_list)):
            k1, v1 = pred_list[i]
            k2, v2 = pred_list[j]
            if abs(v1 - v2) <= tolerance:
                w1 = weights.get(k1, 1.0)
                w2 = weights.get(k2, 1.0)
                w_sum = w1 + w2
                if w_sum > best_weight_sum:
                    best_weight_sum = w_sum
                    best_pair = f"{k1}_{k2}"
                    best_agreement = (v1 * w1 + v2 * w2) / w_sum

    if best_agreement is not None:
        return float(best_agreement), f"pair_vote_{best_pair}"

    total_w = sum(weights.get(k, 1.0) for k in preds)
    total_v = sum(v * weights.get(k, 1.0) for k, v in preds.items())
    return float(total_v / total_w), "weighted_mean_fallback"


def run_brain_subplane_fetalclip_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    """Brain subplane classification with FetalCLIP fine-tuned head."""
    script = os.path.join(config.brain_subplane_fetalclip_dir, "predict_agent.py")
    result = run_tool_subprocess(
        python_path=config.fetalclip_python,
        script_path=script,
        args=[
            "--data_path",
            case_dir,
            "--ckpt_path",
            config.brain_subplane_fetalclip_ckpt,
        ],
        cwd=config.brain_subplane_fetalclip_dir,
        timeout=config.default_timeout,
        log_prefix="BrainSubplane-FetalCLIP",
        print_regexes=[rf"\.{_FNAME_EXT_RE}\s*:\s*"],
    )

    per_image: Dict[str, Dict[str, Any]] = {}
    if result["ok"]:
        kv = _parse_filename_label_probs(result["stdout"])
        for fname, data in kv.items():
            per_image[fname] = {"subplane": data.get("label"), "probs": data.get("probs")}

    ok = result["ok"] and len(per_image) > 0
    return ToolResult(
        tool_name="BrainSubplane-FetalCLIP",
        ok=ok,
        per_image=per_image,
        error=result.get("error") or (None if ok else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


def run_brain_subplane_resnet_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    """Brain subplane classification with ResNet classifier."""
    script = os.path.join(config.agent_tools_dir, "resnet_cls_predict_agent.py")
    result = run_tool_subprocess(
        python_path=config.fetalclip_python,
        script_path=script,
        args=[
            "--data_path",
            case_dir,
            "--ckpt_path",
            config.brain_subplane_resnet_ckpt,
        ],
        cwd=config.agent_tools_dir,
        timeout=config.default_timeout,
        log_prefix="BrainSubplane-ResNet",
        print_regexes=[rf"\.{_FNAME_EXT_RE}\s*:\s*"],
    )

    per_image: Dict[str, Dict[str, Any]] = {}
    if result["ok"]:
        kv = _parse_filename_label_probs(result["stdout"])
        for fname, data in kv.items():
            per_image[fname] = {"subplane": data.get("label"), "probs": data.get("probs")}

    ok = result["ok"] and len(per_image) > 0
    return ToolResult(
        tool_name="BrainSubplane-ResNet",
        ok=ok,
        per_image=per_image,
        error=result.get("error") or (None if ok else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


def run_brain_subplane_vit_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    """Brain subplane classification with ViT classifier."""
    script = os.path.join(config.agent_tools_dir, "vit_cls_predict_agent.py")
    result = run_tool_subprocess(
        python_path=config.fetalclip_python,
        script_path=script,
        args=[
            "--data_path",
            case_dir,
            "--ckpt_path",
            config.brain_subplane_vit_ckpt,
        ],
        cwd=config.agent_tools_dir,
        timeout=config.default_timeout,
        log_prefix="BrainSubplane-ViT",
        print_regexes=[rf"\.{_FNAME_EXT_RE}\s*:\s*"],
    )

    per_image: Dict[str, Dict[str, Any]] = {}
    if result["ok"]:
        kv = _parse_filename_label_probs(result["stdout"])
        for fname, data in kv.items():
            per_image[fname] = {"subplane": data.get("label"), "probs": data.get("probs")}

    ok = result["ok"] and len(per_image) > 0
    return ToolResult(
        tool_name="BrainSubplane-ViT",
        ok=ok,
        per_image=per_image,
        error=result.get("error") or (None if ok else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


def run_stomach_fetalclip_seg_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    script = os.path.join(config.agent_tools_dir, "stomach_seg_fetalclip_predict_agent.py")
    out_dir = _agent_outputs_dir("stomach_segmentation", "fetalclip", case_dir)
    result = run_tool_subprocess(
        python_path=config.fetalclip_python,
        script_path=script,
        args=[
            "--data_path",
            case_dir,
            "--ckpt_path",
            config.stomach_fetalclip_ckpt,
            "--out_dir",
            out_dir,
        ],
        cwd=config.agent_tools_dir,
        timeout=config.default_timeout,
        log_prefix="StomachSeg-FetalCLIP",
        print_regexes=[rf"\.{_FNAME_EXT_RE}\s*:\s*"],
    )

    per_image: Dict[str, Dict[str, Any]] = {}
    if result["ok"]:
        kv = _parse_filename_colon_value(result["stdout"])
        for fname, val in kv.items():
            per_image[fname] = {"mask_path": val}

    ok = result["ok"] and len(per_image) > 0
    return ToolResult(
        tool_name="StomachSeg-FetalCLIP",
        ok=ok,
        per_image=per_image,
        error=result.get("error") or (None if ok else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


def run_stomach_fetalclip_samus_seg_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    script = os.path.join(config.agent_tools_dir, "stomach_seg_fetalclip_samus_predict_agent.py")
    out_dir = _agent_outputs_dir("stomach_segmentation", "fetalclip_samus", case_dir)
    result = run_tool_subprocess(
        python_path=config.fetalclip2_python,
        script_path=script,
        args=[
            "--data_path",
            case_dir,
            "--fetalclip_ckpt",
            config.stomach_fetalclip_ckpt,
            "--samus_ckpt",
            config.stomach_samus_ckpt,
            "--sam_base",
            config.samus_base_ckpt,
            "--out_dir",
            out_dir,
        ],
        cwd=config.agent_tools_dir,
        timeout=config.default_timeout,
        log_prefix="StomachSeg-SAMUS",
        print_regexes=[rf"\.{_FNAME_EXT_RE}\s*:\s*"],
    )

    per_image: Dict[str, Dict[str, Any]] = {}
    if result["ok"]:
        kv = _parse_filename_colon_value(result["stdout"])
        for fname, val in kv.items():
            per_image[fname] = {"mask_path": val}

    ok = result["ok"] and len(per_image) > 0
    return ToolResult(
        tool_name="StomachSeg-FetalCLIP+SAMUS",
        ok=ok,
        per_image=per_image,
        error=result.get("error") or (None if ok else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


def run_stomach_nnunet_seg_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    """Run stomach segmentation with nnUNet helper script."""
    script = os.path.join(config.agent_tools_dir, "nnunet_stomach_seg_predict_agent.py")
    out_dir = _agent_outputs_dir("stomach_segmentation", "nnunet", case_dir)
    result = run_tool_subprocess(
        python_path=config.hxt_base_python,
        script_path=script,
        args=[
            "--data_path",
            case_dir,
            "--out_dir",
            out_dir,
            "--nnunet_predict",
            config.nnunet_predict,
            "--timeout",
            str(config.default_timeout),
            "--progress_every",
            "25",
        ],
        cwd=config.agent_tools_dir,
        timeout=config.default_timeout + 120,
        log_prefix="StomachSeg-nnUNet",
        print_regexes=[rf"\.{_FNAME_EXT_RE}\s*:\s*"],
    )

    per_image: Dict[str, Dict[str, Any]] = {}
    if result["ok"]:
        kv = _parse_filename_colon_value(result["stdout"])
        for fname, val in kv.items():
            if val and os.path.exists(val):
                per_image[fname] = {"mask_path": val}

    ok = result["ok"] and len(per_image) > 0
    return ToolResult(
        tool_name="StomachSeg-nnUNet",
        ok=ok,
        per_image=per_image,
        error=result.get("error") or (None if ok else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


def run_abdomen_fetalclip_seg_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    script = os.path.join(config.agent_tools_dir, "abdomen_seg_fetalclip_predict_agent.py")
    out_dir = _agent_outputs_dir("abdomen_segmentation", "fetalclip", case_dir)
    result = run_tool_subprocess(
        python_path=config.fetalclip2_python,
        script_path=script,
        args=[
            "--data_path",
            case_dir,
            "--ckpt_path",
            config.abdomen_fetalclip_ckpt,
            "--out_dir",
            out_dir,
        ],
        cwd=config.agent_tools_dir,
        timeout=config.default_timeout,
        log_prefix="AbdomenSeg-FetalCLIP",
        print_regexes=[rf"\.{_FNAME_EXT_RE}\s*:\s*"],
    )

    per_image: Dict[str, Dict[str, Any]] = {}
    if result["ok"]:
        kv = _parse_filename_colon_value(result["stdout"])
        for fname, val in kv.items():
            per_image[fname] = {"mask_path": val}

    ok = result["ok"] and len(per_image) > 0
    return ToolResult(
        tool_name="AbdomenSeg-FetalCLIP",
        ok=ok,
        per_image=per_image,
        error=result.get("error") or (None if ok else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


def run_abdomen_fetalclip_samus_seg_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    script = os.path.join(config.agent_tools_dir, "abdomen_seg_fetalclip_samus_predict_agent.py")
    out_dir = _agent_outputs_dir("abdomen_segmentation", "fetalclip_samus", case_dir)
    result = run_tool_subprocess(
        python_path=config.fetalclip2_python,
        script_path=script,
        args=[
            "--data_path",
            case_dir,
            "--fetalclip_ckpt",
            config.abdomen_fetalclip_ckpt,
            "--samus_ckpt",
            config.abdomen_samus_ckpt,
            "--sam_base",
            config.samus_base_ckpt,
            "--out_dir",
            out_dir,
        ],
        cwd=config.agent_tools_dir,
        timeout=config.default_timeout,
        log_prefix="AbdomenSeg-SAMUS",
        print_regexes=[rf"\.{_FNAME_EXT_RE}\s*:\s*"],
    )

    per_image: Dict[str, Dict[str, Any]] = {}
    if result["ok"]:
        kv = _parse_filename_colon_value(result["stdout"])
        for fname, val in kv.items():
            per_image[fname] = {"mask_path": val}

    ok = result["ok"] and len(per_image) > 0
    return ToolResult(
        tool_name="AbdomenSeg-FetalCLIP+SAMUS",
        ok=ok,
        per_image=per_image,
        error=result.get("error") or (None if ok else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )



