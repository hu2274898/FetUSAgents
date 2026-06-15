"""Shared label normalisation for plane / brain-subplane outputs.

Both the general workflow's experts and the specific VQA tool runners
have to take a raw label coming out of a CV tool (``"Fetal Abdomen"``,
``"trans_thalamic"``, ``"BRAIN"``, …) and bucket it into a small,
canonical vocabulary so downstream voting and option-lookup work.

The two workflows want subtly different bucketing rules:

* specific VQA needs the strict mapping — the buckets are the answer
  options, so a tool emitting ``"kidney"`` should fall through to
  ``"other"`` rather than silently rewrite into ``"abdomen"``.
* the general workflow wants the extended mapping — ``"kidney"`` is
  effectively an abdominal hint, ``"heart"`` a thoracic one, and
  losing them as ``"other"`` would needlessly skip plane-dependent
  experts.

:func:`normalize_plane` takes an ``extended`` flag for that switch.

All normalisers return **lowercase** canonical labels — that's what
``PLANE_CANONICAL_TO_OPTION`` / ``BRAIN_CANONICAL_TO_OPTION`` in the
specific pipeline expect, and the general workflow only does string
comparisons so case doesn't matter to it. When the general workflow
needs a human-display variant (``"Trans-thalamic"`` in the report
text), :func:`title_brain_subplane` is the conversion to apply.
"""
from __future__ import annotations

import re
from typing import Optional


def _squash(label: Optional[str]) -> str:
    """Collapse whitespace and lowercase. Empty inputs become ``""``."""
    return re.sub(r"\s+", " ", str(label or "")).strip().lower()


def normalize_plane(label: Optional[str], *, extended: bool = False) -> str:
    """Bucket a plane label into ``abdomen / femur / brain / thorax / other``.

    ``"heart"`` always counts as a thoracic hint — both workflows treat
    a cardiac view as a thorax plane.

    ``extended=False`` (the default, used by specific VQA) keeps
    abdomen strictly to ``"abdomen"``-mentioning labels, since the
    buckets are the answer options and a tool emitting ``"kidney"``
    should fall through to ``"other"`` rather than silently rewrite
    into ``"abdomen"``.

    ``extended=True`` (used by the general workflow) widens that
    bucket — ``"kidney"`` also maps to ``"abdomen"`` — so the
    orchestrator can still pick plane-dependent experts when a tool
    returns a near-synonym instead of failing closed.
    """
    s = _squash(label)
    if "abdomen" in s or (extended and "kidney" in s):
        return "abdomen"
    if "femur" in s:
        return "femur"
    if "brain" in s:
        return "brain"
    if "thorax" in s or "heart" in s:
        return "thorax"
    return "other"


def normalize_brain_subplane(label: Optional[str]) -> str:
    """Bucket a brain subplane label into the three canonical names or ``other``."""
    s = _squash(label).replace("_", "-")
    if "cerebell" in s:
        return "trans-cerebellum"
    if "thalam" in s:
        return "trans-thalamic"
    if "ventric" in s:
        return "trans-ventricular"
    return "other"


def title_brain_subplane(label: Optional[str]) -> str:
    """Convert a canonical brain-subplane label to its display form.

    ``"trans-thalamic"`` → ``"Trans-thalamic"``. Non-canonical strings
    (including ``"other"`` or empty) are returned unchanged so this
    helper is safe to call on anything ``normalize_brain_subplane``
    might produce.
    """
    s = str(label or "")
    if not s.startswith("trans-"):
        return s
    return "Trans-" + s[len("trans-"):]
