"""Stdout / filename parsers used by tool runners and the orchestrator.

These functions translate the loose text output of various predict
scripts (FetalCLIP / FU-LoRA / ResNet / ViT / video keyframes / seg
judges) into structured ``Dict[str, ...]`` mappings keyed by filename.

Pure functions; depend only on :mod:`re` and :mod:`_state` for the
common filename-extension regex.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._state import _FNAME_EXT_RE


def _parse_filename_colon_value(stdout: str) -> Dict[str, str]:
    pat = re.compile(rf"^(?P<fname>.+\.{_FNAME_EXT_RE})\s*:\s*(?P<val>.+)\s*$", flags=re.IGNORECASE)
    out: Dict[str, str] = {}
    for line in stdout.splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        out[m.group("fname").strip()] = m.group("val").strip()
    return out


def _parse_filename_label_probs(stdout: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse lines in format:
      filename.png: Label [0.12 0.34 0.54]
      filename.png: Label
    """
    pat = re.compile(
        rf"(?P<fname>[^\s:]+\.(?:{_FNAME_EXT_RE}))\s*:\s*(?P<label>[^\[]+?)(?:\s*\[(?P<probs>[^\]]+)\])?\s*$",
        flags=re.IGNORECASE,
    )
    out: Dict[str, Dict[str, Any]] = {}
    for line in stdout.splitlines():
        m = pat.search(line.strip())
        if not m:
            continue
        fname = m.group("fname").strip()
        label = m.group("label").strip()
        probs_raw = m.group("probs")
        probs: Optional[List[float]] = None
        if probs_raw:
            try:
                probs = [float(x) for x in probs_raw.strip().split()]
            except Exception:
                probs = None
        out[fname] = {"label": label, "probs": probs}
    return out


def _parse_filename_colon_text(stdout: str) -> Dict[str, str]:
    pat = re.compile(rf"^(?P<fname>.+\.{_FNAME_EXT_RE})\s*:\s*(?P<val>.+?)\s*$", flags=re.IGNORECASE)
    out: Dict[str, str] = {}
    for line in stdout.splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        out[m.group("fname").strip()] = m.group("val").strip()
    return out


def _normalize_video_plane_label(label: Optional[str]) -> str:
    s = (label or "").strip().lower()
    if "biparietal" in s or "brain" in s:
        return "brain"
    if "abdominal" in s or "abdomen" in s:
        return "abdomen"
    if "femur" in s:
        return "femur"
    if "heart" in s or "thorax" in s:
        return "thorax"
    if "spine" in s:
        return "spine"
    if "no plane" in s or s == "no_plane":
        return "other"
    return "other"


def _is_video_summary_request(inquiry: str) -> bool:
    t = (inquiry or "").lower()
    has_video = any(k in t for k in ("video", "continuous screenshot", "continuous screenshots", "cine", "sequence"))
    has_summary = any(k in t for k in ("summary", "comprehensive", "caption"))
    return has_video and has_summary


def _resolve_image_key(pred_name: str, available_images: List[str]) -> Optional[str]:
    pred_stem = Path(pred_name).stem.lower()
    pred_full = pred_name.lower()
    for x in available_images:
        if x.lower() == pred_full:
            return x
    for x in available_images:
        if Path(x).stem.lower() == pred_stem:
            return x
    return None


def _parse_seg_judge_output(text: str) -> Dict[str, str]:
    """
    Parse VLLM judge picks. Accepts lines like:
      - 0041.png — tool1
      0041.png: tool2
      0041.png - none
    Returns: { "0041.png": "tool1" | "tool2" | "none" }
    """
    out: Dict[str, str] = {}
    if not text:
        return out
    rx = re.compile(
        rf"^\s*(?:[-*•]\s*)?(?P<fname>.+\.{_FNAME_EXT_RE})\s*(?:—|–|-|:)\s*(?P<pick>tool\s*1|tool\s*2|tool1|tool2|none)\b",
        flags=re.IGNORECASE,
    )
    for line in text.splitlines():
        m = rx.match(line.strip())
        if not m:
            continue
        fname = m.group("fname").strip()
        pick_raw = m.group("pick").strip().lower().replace(" ", "")
        if pick_raw in ("tool1", "tool2", "none"):
            out[fname] = pick_raw
        elif pick_raw == "tool1":
            out[fname] = "tool1"
        elif pick_raw == "tool2":
            out[fname] = "tool2"
    return out
