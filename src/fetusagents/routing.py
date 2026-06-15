"""Query parsing utilities used by the Coordinator.

The routing logic is deliberately rule-based: it must be auditable, fast,
and runnable in unit tests without any GPU / LLM access. Anything genuinely
ambiguous is passed through ``parsed_options`` and ``overrides`` so that
downstream workflows still receive the user's verbatim question.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
_VIDEO_FILE_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def parse_options(query: str) -> Dict[str, str]:
    """Extract MC options from a natural-language question.

    Recognises three common formats:

    * ``(A) Foo (B) Bar (C) Baz``  - parenthesised letters
    * ``A. Foo  B. Bar``           - dotted letters
    * ``Yes / No``                 - explicit binary

    Returns a dict mapping letter to option text. The dict is empty when
    no option block can be detected.
    """
    if not query:
        return {}

    paren_pattern = re.compile(r"\(([A-Da-d])\)\s*([^()]+?)(?=\s*\(([A-Da-d])\)|$)", re.DOTALL)
    matches = paren_pattern.findall(query)
    if matches:
        return {letter.upper(): text.strip().rstrip(".") for letter, text, _ in matches}

    dotted_pattern = re.compile(
        r"(?<![A-Za-z])([A-D])[\.\)]\s+([^A-D\n]+?)(?=\s+[A-D][\.\)]\s+|$)", re.DOTALL
    )
    dotted = dotted_pattern.findall(query)
    if len(dotted) >= 2:
        return {letter.upper(): text.strip().rstrip(".") for letter, text in dotted}

    lower = query.lower()
    if "yes" in lower and "no" in lower and re.search(r"\byes\s*/\s*no\b|\byes\b.*\bno\b", lower):
        return {"A": "Yes", "B": "No"}

    return {}


def has_explicit_option_block(query: str) -> bool:
    return len(parse_options(query)) >= 2


def is_video_input(input_path: str) -> bool:
    """Return ``True`` if ``input_path`` looks like a video or frame folder."""
    if not input_path:
        return False
    if os.path.isfile(input_path) and input_path.lower().endswith(_VIDEO_FILE_EXTS):
        return True
    if os.path.isdir(input_path):
        try:
            children = [n for n in os.listdir(input_path) if not n.startswith(".")]
        except OSError:
            return False
        image_children = [n for n in children if n.lower().endswith(_IMAGE_EXTS)]
        if len(image_children) >= 3:
            return True
    return False


_GENERAL_TRIGGERS = (
    "comprehensive caption",
    "comprehensive summary",
    "describe this",
    "write a caption",
    "write a comprehensive",
    "provide a summary",
    "provide a comprehensive summary",
    "summarize this video",
    "summarize the video",
    "summarise",
    "generate",
)


def looks_like_open_ended(query: str) -> bool:
    if not query:
        return False
    q = query.lower()
    if any(trigger in q for trigger in _GENERAL_TRIGGERS):
        return True
    if "which option" in q or "choose the best answer" in q or "which is correct" in q:
        return False
    return False


_TASK_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("aop_binary", [
        "angle of progression", "aop", "vaginal delivery", "cesarean",
        "instrumental delivery", "120", "instrumental",
    ]),
    ("hc_estimation_pixel", [
        "head circumference", "hc circumference", " hc ", "fetal head circumference",
        "circumference in pixels of the fetal head",
    ]),
    ("ac_estimation_pixel", [
        "abdominal circumference", " ac ", "abdomen circumference",
        "circumference in pixels of the fetal abdomen", "ac estimation",
    ]),
    ("stomach_volume_estimation", [
        "stomach", "gastric bubble", "fetal stomach", "stomach area", "stomach volume",
    ]),
    ("brain_subplane", [
        "trans-thalamic", "trans-cerebellar", "trans-ventricular",
        "brain plane", "cranial plane", "brain subplane",
    ]),
    ("ga_trimester_multi", [
        "which trimester", "pregnancy stage", "trimester is this",
        "first trimester", "second trimester", "third trimester",
    ]),
    ("plane_classification", [
        "anatomical plane", "which plane", "what plane",
        "fetal abdomen", "fetal femur", "fetal brain", "fetal thorax",
    ]),
]


def keyword_score(query: str, keywords: List[str]) -> int:
    q = " " + query.lower() + " "
    return sum(1 for kw in keywords if kw in q)


def infer_specific_task(query: str, options: Dict[str, str]) -> Tuple[str, float, str]:
    """Rank candidate specific task types from textual cues.

    Returns ``(task_id, confidence, reason)``. ``confidence`` is a rough
    score in ``[0, 1]`` based on how many keywords matched.
    """
    if not query and not options:
        return "plane_classification", 0.0, "no signal available"

    haystack = query.lower()
    options_blob = " ".join(options.values()).lower()
    if options_blob:
        haystack = haystack + " " + options_blob

    binary_opts = set(o.lower() for o in options.values()) == {"yes", "no"} if options else False

    best_task = "plane_classification"
    best_score = 0
    best_reason = ""
    for task_id, keywords in _TASK_KEYWORDS:
        score = keyword_score(haystack, keywords)
        if score > best_score:
            best_score = score
            best_task = task_id
            best_reason = (
                f"matched {score} keyword(s) for '{task_id}': "
                + ", ".join(kw for kw in keywords if kw in haystack)[:200]
            )

    if binary_opts and best_task == "plane_classification" and "fetal " in haystack:
        best_task = "plane_binary"
        best_reason = "Yes/No options on a plane question → plane_binary"
        best_score = max(best_score, 1)
    if binary_opts and best_task == "brain_subplane":
        best_task = "brain_subplane_binary"
        best_reason = "Yes/No options on a brain-subplane question → brain_subplane_binary"
    if binary_opts and best_task == "ga_trimester_multi":
        best_task = "ga_trimester_binary"
        best_reason = "Yes/No options on a trimester question → ga_trimester_binary"

    confidence = min(1.0, 0.3 + 0.2 * best_score)
    if not best_reason:
        best_reason = "no strong keyword match; defaulting to plane_classification"
    return best_task, confidence, best_reason
