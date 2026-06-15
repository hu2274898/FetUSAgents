"""Image / mask helpers shared across tool runners and the orchestrator.

Everything in this module is pure-image manipulation:
- PIL / OpenCV loaders and converters
- Square padding, overlays, side-by-side concatenation
- Mask resizing, majority voting, connected-component cleanup
- Ellipse-residual quality score

It depends only on PIL / cv2 / numpy, plus :mod:`_state` for
``_FNAME_EXT_RE`` and ``_SCRIPT_DIR``.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image as PILImage
from autogen_core import Image as AGImage

try:
    import numpy as np
except Exception:
    np = None

try:
    import cv2
except Exception:
    cv2 = None

from ._state import _FNAME_EXT_RE, _SCRIPT_DIR
from .biometry import parse_pixel_size_csv


def _agent_outputs_dir(task: str, tool: str, case_dir: str) -> str:
    """Writable output folder under ``<package>/outputs_agent/``."""
    out = _SCRIPT_DIR / "outputs_agent" / task / tool / Path(case_dir).name
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


def _safe_load_pil(path: str) -> Optional[PILImage.Image]:
    try:
        return PILImage.open(path).convert("RGB")
    except Exception:
        return None


def _make_square_pil(image: PILImage.Image) -> PILImage.Image:
    """Pad to square with black background (matches tool-side preprocessing)."""
    width, height = image.size
    max_side = max(width, height)
    new_image = PILImage.new("RGB", (max_side, max_side), (0, 0, 0))
    padding_left = (max_side - width) // 2
    padding_top = (max_side - height) // 2
    new_image.paste(image.convert("RGB"), (padding_left, padding_top))
    return new_image


def _make_overlay(
    raw_img: PILImage.Image,
    mask_path: Optional[str],
    preprocess: str = "resize_direct",  # "resize_direct" | "pad_square"
    color=(0, 255, 0, 120),
) -> Optional[PILImage.Image]:
    """
    Overlay a predicted mask onto the ORIGINAL raw image coordinate system.

    preprocess:
      - "resize_direct": tool resized raw (H,W) -> (S,S) without padding
      - "pad_square":    tool padded raw to square then resized to (S,S)
    """
    if not mask_path or not os.path.exists(mask_path) or raw_img is None:
        return None

    try:
        mask = PILImage.open(mask_path).convert("L")
    except Exception:
        return None

    w, h = raw_img.size

    if preprocess == "resize_direct":
        mask_on_raw = mask.resize((w, h), resample=PILImage.NEAREST)

    elif preprocess == "pad_square":
        max_side = max(w, h)
        pad_left = (max_side - w) // 2
        pad_top = (max_side - h) // 2
        mask_sq = mask.resize((max_side, max_side), resample=PILImage.NEAREST)
        mask_on_raw = mask_sq.crop((pad_left, pad_top, pad_left + w, pad_top + h))

    else:
        mask_on_raw = mask.resize((w, h), resample=PILImage.NEAREST)

    raw_rgba = raw_img.convert("RGBA")
    alpha = mask_on_raw.point(lambda p: color[3] if p > 0 else 0)
    color_layer = PILImage.new("RGBA", raw_rgba.size, (color[0], color[1], color[2], 0))
    color_layer.putalpha(alpha)
    return PILImage.alpha_composite(raw_rgba, color_layer)


def _concat_side_by_side(images: List[PILImage.Image]) -> Optional[PILImage.Image]:
    imgs = [im for im in images if im is not None]
    if not imgs:
        return None
    widths, heights = zip(*(i.size for i in imgs))
    total_width = sum(widths)
    max_height = max(heights)
    canvas = PILImage.new("RGB", (total_width, max_height), (0, 0, 0))
    x = 0
    for im in imgs:
        canvas.paste(im.convert("RGB"), (x, 0))
        x += im.size[0]
    return canvas


def _pil_to_agimage(img: PILImage.Image) -> AGImage:
    return AGImage(img)


# Single-image case_dir helper (uses pixel_size.csv)
def _make_single_image_case_dir(case_dir: str, image_name: str) -> str:
    tmp_dir = tempfile.mkdtemp(prefix="agent_single_case_")
    src_img = os.path.join(case_dir, image_name)
    dst_img = os.path.join(tmp_dir, image_name)
    shutil.copy2(src_img, dst_img)

    src_csv = os.path.join(case_dir, "pixel_size.csv")
    if os.path.exists(src_csv):
        pixel_map = parse_pixel_size_csv(src_csv)
        px = pixel_map.get(image_name)
        with open(os.path.join(tmp_dir, "pixel_size.csv"), "w", encoding="utf-8") as f:
            f.write("filename,pixel size(mm)\n")
            if px is not None:
                f.write(f"{image_name},{px}\n")
            else:
                f.write(f"{image_name},0.15\n")
    else:
        with open(os.path.join(tmp_dir, "pixel_size.csv"), "w", encoding="utf-8") as f:
            f.write("filename,pixel size(mm)\n")
            f.write(f"{image_name},0.15\n")
    return tmp_dir


# Mask manipulation
def _mask_to_raw_array(mask_path: Optional[str], raw_img: Optional[PILImage.Image], preprocess: str) -> Optional[Any]:
    """Rescale predicted binary mask back to the raw image size."""
    if np is None or raw_img is None or not mask_path or not os.path.exists(mask_path):
        return None
    try:
        mask = PILImage.open(mask_path).convert("L")
    except Exception:
        return None

    w, h = raw_img.size
    if preprocess == "resize_direct":
        mask_on_raw = mask.resize((w, h), resample=PILImage.NEAREST)
    elif preprocess == "pad_square":
        max_side = max(w, h)
        pad_left = (max_side - w) // 2
        pad_top = (max_side - h) // 2
        mask_sq = mask.resize((max_side, max_side), resample=PILImage.NEAREST)
        mask_on_raw = mask_sq.crop((pad_left, pad_top, pad_left + w, pad_top + h))
    else:
        mask_on_raw = mask.resize((w, h), resample=PILImage.NEAREST)
    arr = np.array(mask_on_raw)  # type: ignore[arg-type]
    return (arr > 0).astype("uint8")


def _load_mask_binary_cv2(mask_path: Optional[str], target_shape: Optional[Tuple[int, int]] = None) -> Optional[Any]:
    """Load a binary mask with OpenCV. If target_shape=(h,w), resize NEAREST."""
    if cv2 is None or np is None or not mask_path or not os.path.exists(mask_path):
        return None
    try:
        img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        if target_shape is not None and img.shape[:2] != target_shape:
            img = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        return (img > 0).astype("uint8")
    except Exception:
        return None


def _dice_masks(a: Optional[Any], b: Optional[Any]) -> Optional[float]:
    if np is None or a is None or b is None:
        return None
    a_sum = int(a.sum())
    b_sum = int(b.sum())
    if a_sum == 0 and b_sum == 0:
        return 1.0
    if a_sum == 0 or b_sum == 0:
        return 0.0
    inter = int((a & b).sum())
    return (2.0 * inter) / float(a_sum + b_sum)


def _majority_voting(masks: List[Any]) -> Optional[Any]:
    if np is None:
        return None
    valid = [m for m in masks if m is not None]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0].copy()
    stacked = np.stack(valid, axis=0)  # type: ignore[arg-type]
    thr = len(valid) / 2.0
    return (stacked.sum(axis=0) >= thr).astype("uint8")


def _keep_largest_component(mask: Any, min_area: int = 50) -> Any:
    if np is None or cv2 is None or mask is None or int(mask.sum()) == 0:
        return mask
    mask_u8 = (mask * 255).astype("uint8")
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    largest_area = int(stats[largest_idx, cv2.CC_STAT_AREA])
    if largest_area < min_area:
        return np.zeros_like(mask, dtype="uint8")
    return (labels == largest_idx).astype("uint8")


def _apply_postprocess(mask: Optional[Any], min_area: int = 50) -> Optional[Any]:
    if mask is None:
        return None
    return _keep_largest_component(mask, min_area=min_area)


def _compute_ellipse_residual(mask: Optional[Any]) -> float:
    """Inference-time mask quality score: lower is better."""
    if np is None or cv2 is None or mask is None or int(mask.sum()) == 0:
        return float("nan")
    try:
        mask_bin = (mask > 0).astype("uint8")
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return float("nan")
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 5:
            return float("nan")
        ellipse = cv2.fitEllipse(contour)
        (cx, cy), (major, minor), angle = ellipse
        a = max(major, minor) / 2.0
        b = min(major, minor) / 2.0
        if a <= 0 or b <= 0:
            return float("nan")
        angle_rad = math.radians(angle)
        distances: List[float] = []
        step = max(1, len(contour) // 100)
        for pt in contour[::step]:
            x, y = pt[0]
            dx = float(x) - float(cx)
            dy = float(y) - float(cy)
            x_rot = dx * math.cos(-angle_rad) - dy * math.sin(-angle_rad)
            y_rot = dx * math.sin(-angle_rad) + dy * math.cos(-angle_rad)
            ellipse_val = (x_rot / a) ** 2 + (y_rot / b) ** 2
            dist = abs(ellipse_val - 1.0) * min(a, b)
            distances.append(float(dist))
        return float(sum(distances) / len(distances)) if distances else float("nan")
    except Exception:
        return float("nan")
