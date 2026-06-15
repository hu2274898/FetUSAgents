"""Answer parsers, option-to-text utilities, dataset I/O, and small
geometry helpers used by the specific VQA pipeline.

Pure-Python — no GPU, no LLM, no subprocess. Tested by the routing
suite under ``tests/`` (indirectly via specific_vqa.run_single).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except Exception:
    np = None

from autogen_core import Image
from autogen_agentchat.messages import MultiModalMessage

from ..core.label_normalize import (
    normalize_brain_subplane as _normalize_brain_subplane,
    normalize_plane as _normalize_plane,
)
from .types import AgentVote, DEFAULT_PIXEL_SIZE_MM, VQASample


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def make_mm_message(text: str, image_path: str) -> MultiModalMessage:
    return MultiModalMessage(
        content=[
            text,
            Image.from_file(Path(image_path)),
        ],
        source="user",
    )




def load_vqa_dataset(json_path: str, image_dir: str, task_name: str) -> List[VQASample]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: List[VQASample] = []
    for item in data:
        px = item.get("pixel_size", None)
        try:
            px = float(px) if px is not None else None
        except Exception:
            px = None

        out.append(
            VQASample(
                image_id=item["image_id"],
                question=item["question"],
                options=item.get("options", []),
                answer=str(item["answer"]).strip().upper(),
                image_dir=image_dir,
                task_name=task_name,
                pixel_size=px,
            )
        )
    return out


def count_votes(votes: List[AgentVote]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in votes:
        if v.answer_letter is None:
            continue
        out[v.answer_letter] = out.get(v.answer_letter, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def get_consensus_answer(vote_count: Dict[str, int], threshold: int = 4) -> Optional[str]:
    for k, v in vote_count.items():
        if v >= threshold:
            return k
    return None


def ensure_single_image_case_dir(
    image_path: str,
    pixel_size_mm: Optional[float] = None,
) -> str:
    tmp_dir = tempfile.mkdtemp(prefix="vqa_single_case_")
    image_name = os.path.basename(image_path)
    shutil.copy2(image_path, os.path.join(tmp_dir, image_name))

    px = DEFAULT_PIXEL_SIZE_MM if pixel_size_mm is None else float(pixel_size_mm)

    with open(os.path.join(tmp_dir, "pixel_size.csv"), "w", encoding="utf-8") as f:
        f.write("filename,pixel size(mm)\n")
        f.write(f"{image_name},{px}\n")

    return tmp_dir


def parse_plane_answer(text: str) -> Optional[str]:
    if not text:
        return None

    t = text.strip()
    patterns = [
        r"final answer\s*[:：]\s*\(?([A-D])\)?",
        r"answer\s*[:：]\s*\(?([A-D])\)?",
        r"option\s*[:：]\s*\(?([A-D])\)?",
        r"^\s*\(?([A-D])\)?\s*$",
    ]
    for p in patterns:
        m = re.search(p, t, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    tl = t.lower()
    if "abdomen" in tl:
        return "A"
    if "femur" in tl:
        return "B"
    if "brain" in tl:
        return "C"
    if "thorax" in tl or "heart" in tl:
        return "D"
    return None


def parse_brain_subplane_answer(text: str) -> Optional[str]:
    if not text:
        return None

    t = text.strip()
    patterns = [
        r"final answer\s*[:：]\s*\(?([A-C])\)?",
        r"answer\s*[:：]\s*\(?([A-C])\)?",
        r"option\s*[:：]\s*\(?([A-C])\)?",
        r"^\s*\(?([A-C])\)?\s*$",
    ]
    for p in patterns:
        m = re.search(p, t, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    tl = t.lower()
    if "cerebell" in tl:
        return "A"
    if "thalam" in tl:
        return "B"
    if "ventric" in tl:
        return "C"
    return None

def parse_yesno_answer(text: str) -> Optional[str]:
    if not text:
        return None

    t = text.strip()

    patterns = [
        r"final answer\s*[:：]\s*\(?([A-B])\)?",
        r"answer\s*[:：]\s*\(?([A-B])\)?",
        r"option\s*[:：]\s*\(?([A-B])\)?",
        r"^\s*\(?([A-B])\)?\s*$",
    ]
    for p in patterns:
        m = re.search(p, t, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    tl = t.lower()
    if re.search(r"\byes\b", tl):
        return "A"
    if re.search(r"\bno\b", tl):
        return "B"
    return None

def parse_abcd_answer(text: str) -> Optional[str]:
    if not text:
        return None

    t = text.strip()
    patterns = [
        r"final answer\s*[:：]\s*\(?([A-D])\)?",
        r"answer\s*[:：]\s*\(?([A-D])\)?",
        r"option\s*[:：]\s*\(?([A-D])\)?",
        r"^\s*\(?([A-D])\)?\s*$",
    ]
    for p in patterns:
        m = re.search(p, t, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None

def parse_numeric_options(options: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for opt in options:
        m = re.match(r"\(?([A-D])\)?\s*[\).]?\s*([0-9]+(?:\.[0-9]+)?)", opt.strip(), flags=re.IGNORECASE)
        if m:
            letter = m.group(1).upper()
            value = float(m.group(2))
            out[letter] = value
    return out

def parse_trimester_multi_answer(text: str) -> Optional[str]:
    if not text:
        return None

    t = text.strip()

    patterns = [
        r"final answer\s*[:：]\s*\(?([A-C])\)?",
        r"answer\s*[:：]\s*\(?([A-C])\)?",
        r"option\s*[:：]\s*\(?([A-C])\)?",
        r"^\s*\(?([A-C])\)?\s*$",
    ]
    for p in patterns:
        m = re.search(p, t, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    tl = t.lower()
    if "first trimester" in tl:
        return "A"
    if "second trimester" in tl:
        return "B"
    if "third trimester" in tl:
        return "C"

    return None


def normalize_plane_label(label: Optional[str]) -> str:
    return _normalize_plane(label)


def normalize_brain_subplane_label(label: Optional[str]) -> str:
    return _normalize_brain_subplane(label)

def extract_target_brain_subplane_from_question(question: str) -> Optional[str]:
    q = normalize_space(question).lower().replace("_", "-")

    if "trans-cerebell" in q:
        return "trans-cerebellum"
    if "trans-thalam" in q:
        return "trans-thalamic"
    if "trans-ventric" in q:
        return "trans-ventricular"

    return None

def extract_target_trimester_from_question(question: str) -> Optional[str]:
    q = normalize_space(question).lower()
    if "first trimester" in q:
        return "first"
    if "second trimester" in q:
        return "second"
    if "third trimester" in q:
        return "third"
    return None

def extract_target_plane_from_question(question: str) -> Optional[str]:
    q = normalize_space(question).lower()

    if "fetal abdomen" in q or re.search(r"\babdomen\b", q):
        return "abdomen"
    if "fetal femur" in q or re.search(r"\bfemur\b", q):
        return "femur"
    if "fetal brain" in q or re.search(r"\bbrain\b", q):
        return "brain"
    if "fetal thorax" in q or re.search(r"\bthorax\b", q) or re.search(r"\bheart\b", q):
        return "thorax"

    return None

def trimester_from_total_weeks(total_weeks: float) -> str:
    if total_weeks >= 28.0:
        return "third"
    elif total_weeks >= 14.0:
        return "second"
    else:
        return "first"

def pick_closest_option_letter(pred_value: float, option_map: Dict[str, float]) -> Optional[str]:
    if pred_value is None or not option_map:
        return None
    return min(option_map.items(), key=lambda kv: abs(kv[1] - pred_value))[0]

def _stomach_area_pixel_from_mask_array(mask: Optional[Any]) -> Optional[float]:
    if mask is None or np is None:
        return None
    try:
        return float(np.count_nonzero(mask))
    except Exception:
        return None

def _majority_vote_masks(masks: List[Any]) -> Optional[Any]:
    if np is None:
        return None
    valid = [m.astype("uint8") for m in masks if m is not None]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    stack = np.stack(valid, axis=0)
    thr = len(valid) / 2.0
    return (stack.sum(axis=0) >= thr).astype("uint8")
