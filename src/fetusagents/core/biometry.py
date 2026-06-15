"""Fetal biometry: pixel-size CSV, HC/AC measurements, GA conversion,
Hadlock formula, percentile assessment from reference tables.

All functions here are pure: they consume floats/arrays and return
floats/dicts. None of them touch the LLM or the subprocess runner.
"""
from __future__ import annotations

import csv
import json
import math
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


# Pixel Size CSV Utilities
def parse_pixel_size_csv(csv_path: str) -> Dict[str, float]:
    """Parse pixel_size.csv -> {filename: pixel_size_mm}."""
    pixel_sizes = {}
    if not os.path.exists(csv_path):
        return pixel_sizes
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("filename", "").strip()
                pixel_size_str = row.get("pixel size(mm)", "").strip()
                if filename and pixel_size_str:
                    try:
                        pixel_sizes[filename] = float(pixel_size_str)
                    except ValueError:
                        pass
    except Exception as e:
        print(f"[CSV] Error reading {csv_path}: {e}")
    return pixel_sizes


def ensure_pixel_csv(case_dir: str) -> str:
    """Ensure pixel_size.csv exists in case_dir. Return path."""
    csv_path = os.path.join(case_dir, "pixel_size.csv")
    if os.path.exists(csv_path):
        return csv_path

    images = [f for f in os.listdir(case_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("filename,pixel size(mm)\n")
        for img in images:
            f.write(f"{img},0.15\n")
    print(f"[INFO] Created default pixel_size.csv with {len(images)} images")
    return csv_path


# GA ↔ HC cross-check utilities
# HC(mm) = exp(b0 + b1*t + b2*t^2 + b3*t^3 + b4*t^4); t = GA(weeks).
GA_TO_HC_PARAMS: Dict[str, List[float]] = {
    "0.025": [1.59317517131532, 2.9459800552433e-1, -7.3860372566707e-3, 6.56951770216148e-5, 0.0],
    "0.5":   [2.09924879247164, 2.53373656106037e-1, -6.05647816678282e-3, 5.14256072059917e-5, 0.0],
    "0.975": [2.50074069629423, 2.20067854715719e-1, -4.93623111462443e-3, 3.89066000946519e-5, 0.0],
}


def hc_from_ga_weeks(t_weeks: float, params: List[float]) -> float:
    """HC(mm) from GA(weeks) using the polynomial-in-weeks model."""
    b0, b1, b2, b3, b4 = params
    return float(math.exp(b0 + b1 * t_weeks + b2 * (t_weeks ** 2) + b3 * (t_weeks ** 3) + b4 * (t_weeks ** 4)))


def hc_range_from_ga_weeks(t_weeks: float) -> Dict[str, float]:
    """HC(mm) at 2.5 / 50 / 97.5 percentile for a given GA."""
    return {
        "p2_5": hc_from_ga_weeks(t_weeks, GA_TO_HC_PARAMS["0.025"]),
        "p50": hc_from_ga_weeks(t_weeks, GA_TO_HC_PARAMS["0.5"]),
        "p97_5": hc_from_ga_weeks(t_weeks, GA_TO_HC_PARAMS["0.975"]),
    }


def weeks_days_to_float_weeks(weeks: int, days: int) -> float:
    return float(weeks) + float(days) / 7.0


def float_weeks_to_weeks_days(t_weeks: float) -> Tuple[int, int]:
    w = int(t_weeks)
    d = int(round((t_weeks - w) * 7))
    if d >= 7:
        w += 1
        d -= 7
    if d < 0:
        d = 0
    return w, d


# Mask -> circumference (HC / AC share the same routine)
def _largest_component_edge(mask: Any) -> Optional[Any]:
    if cv2 is None or np is None or mask is None or mask.size == 0:
        return None
    try:
        img = (mask > 0).astype("uint8") * 255
        retval, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=4)
        if retval <= 1:
            return None
        sort_label = np.argsort(-stats[:, 4])
        idx = labels == int(sort_label[1])
        max_connect = (idx * 255).astype("uint8")
        return cv2.Canny(max_connect, 50, 250)
    except Exception:
        return None


def _ellipse_circumference_mm_from_mask_array(mask: Optional[Any], pixel_size_mm: Optional[float]) -> Optional[float]:
    """
    Compute circumference (mm) from a binary mask by contour->fitEllipse->Ramanujan-II.
    Used for both HC and AC, aligned with the eval scripts.
    """
    if cv2 is None or np is None or mask is None or pixel_size_mm is None:
        return None
    try:
        mask_bin = (mask > 0).astype("uint8")
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        if contour is None or len(contour) < 5:
            return None
        (_, _), (major, minor), _ = cv2.fitEllipse(contour)
        a = max(float(major), float(minor)) / 2.0
        b = min(float(major), float(minor)) / 2.0
        if a <= 0 or b <= 0:
            return None
        circ_px = math.pi * (3.0 * (a + b) - math.sqrt((3.0 * a + b) * (a + 3.0 * b)))
        return float(circ_px * float(pixel_size_mm))
    except Exception:
        return None


def _hc_mm_from_mask_array(mask: Optional[Any], pixel_size_mm: Optional[float]) -> Optional[float]:
    return _ellipse_circumference_mm_from_mask_array(mask, pixel_size_mm)


def _ac_mm_from_mask_array(mask: Optional[Any], pixel_size_mm: Optional[float]) -> Optional[float]:
    return _ellipse_circumference_mm_from_mask_array(mask, pixel_size_mm)


def _round_1dp(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), 1)
    except Exception:
        return None


# Hadlock GA-from-AC + formatting helpers
def _hadlock_ga_weeks_from_ac_mm(ac_mm: Optional[float]) -> Optional[float]:
    """
    Hadlock formula (AC-based):
      GA(weeks) = 8.14 + 0.0753 * AC(mm) + 0.000036 * AC(mm)^2
    """
    if ac_mm is None:
        return None
    try:
        ac = float(ac_mm)
        return 8.14 + 0.0753 * ac + 0.000036 * (ac ** 2)
    except Exception:
        return None


def _format_ga_weeks_days(t_weeks: Optional[float]) -> Optional[str]:
    if t_weeks is None:
        return None
    try:
        w, d = float_weeks_to_weeks_days(float(t_weeks))
        return f"{w}w {d}d"
    except Exception:
        return None


def _ga_label_to_weeks(ga_label: str) -> Optional[float]:
    if not ga_label:
        return None
    m = re.search(r"(\d+)\s*w\s*(\d+)\s*d", str(ga_label), flags=re.IGNORECASE)
    if not m:
        return None
    weeks = int(m.group(1))
    days = int(m.group(2))
    return weeks + days / 7.0


# Percentile reference tables (CSV-driven)
def _load_ga_reference_table(csv_path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(csv_path):
        return rows
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ga_label = (row.get("GA(weeks/days)") or "").strip()
                ga_weeks = _ga_label_to_weeks(ga_label)
                if ga_weeks is None:
                    continue
                percentiles: Dict[float, float] = {}
                for k, v in row.items():
                    if k == "GA(weeks/days)":
                        continue
                    try:
                        pk = float(str(k).strip())
                        pv = float(str(v).strip())
                    except Exception:
                        continue
                    percentiles[pk] = pv
                if percentiles:
                    rows.append(
                        {
                            "ga_label": ga_label,
                            "ga_weeks": ga_weeks,
                            "percentiles": percentiles,
                        }
                    )
    except Exception as e:
        print(f"[Reference] Error reading {csv_path}: {e}")
        return []
    rows.sort(key=lambda x: float(x["ga_weeks"]))
    return rows


def _nearest_ga_row(table: List[Dict[str, Any]], ga_weeks: float) -> Optional[Dict[str, Any]]:
    if not table:
        return None
    return min(table, key=lambda r: abs(float(r["ga_weeks"]) - float(ga_weeks)))


def _fmt_percentile(p: float) -> str:
    if abs(p - round(p)) < 1e-9:
        return str(int(round(p)))
    return f"{p:.1f}".rstrip("0").rstrip(".")


def _percentile_assessment(
    measurement_mm: Optional[float],
    ga_weeks: Optional[float],
    table: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if measurement_mm is None or ga_weeks is None:
        return None
    row = _nearest_ga_row(table, ga_weeks)
    if not row:
        return None
    perc_map = row.get("percentiles") or {}
    if not isinstance(perc_map, dict) or not perc_map:
        return None
    sorted_p = sorted(float(p) for p in perc_map.keys())
    if not sorted_p:
        return None

    value = float(measurement_mm)
    p_low = 2.5 if 2.5 in perc_map else sorted_p[0]
    p_high = 97.5 if 97.5 in perc_map else sorted_p[-1]
    v_low = float(perc_map[p_low])
    v_high = float(perc_map[p_high])

    if value < v_low:
        status = "smaller"
    elif value > v_high:
        status = "larger"
    else:
        status = "within"

    band_lo = sorted_p[0]
    band_hi = sorted_p[-1]
    for i in range(len(sorted_p) - 1):
        p1 = sorted_p[i]
        p2 = sorted_p[i + 1]
        v1 = float(perc_map[p1])
        v2 = float(perc_map[p2])
        lo_v, hi_v = (v1, v2) if v1 <= v2 else (v2, v1)
        if lo_v <= value <= hi_v:
            band_lo = p1
            band_hi = p2
            break
        if value < float(perc_map[sorted_p[0]]):
            band_lo = sorted_p[0]
            band_hi = sorted_p[0]
        if value > float(perc_map[sorted_p[-1]]):
            band_lo = sorted_p[-1]
            band_hi = sorted_p[-1]

    return {
        "status": status,
        "band_lo": band_lo,
        "band_hi": band_hi,
        "band_text": f"{_fmt_percentile(band_lo)}th-{_fmt_percentile(band_hi)}th Percentile",
        "normal_text": f"{_fmt_percentile(p_low)}th-{_fmt_percentile(p_high)}th Percentile",
        "ga_label_used": row.get("ga_label"),
    }


def _hc_percentile_sanity_check(
    recommended_hc_mm: Optional[float],
    alt_hc_mm: Optional[float],
    ga_weeks: Optional[float],
    hc_table: List[Dict[str, Any]],
    rec_source: str = "",
    alt_source: str = "",
) -> Tuple[Optional[float], str, str]:
    """Post-hoc HC plausibility check against GA-based percentile reference.

    Returns (final_hc_mm, final_source, note).
    If the recommended HC is outside 2.5-97.5 percentile for the given GA
    but the alternative tool's HC is within range, switch to the alternative.
    """
    if recommended_hc_mm is None or ga_weeks is None:
        return recommended_hc_mm, rec_source, "no_check"
    rec_assess = _percentile_assessment(recommended_hc_mm, ga_weeks, hc_table)
    if rec_assess is None:
        return recommended_hc_mm, rec_source, "no_reference_data"
    if rec_assess["status"] == "within":
        return recommended_hc_mm, rec_source, "in_range"
    if alt_hc_mm is not None:
        alt_assess = _percentile_assessment(alt_hc_mm, ga_weeks, hc_table)
        if alt_assess is not None and alt_assess["status"] == "within":
            return (
                _round_1dp(alt_hc_mm),
                alt_source,
                f"switched: {rec_source} HC {recommended_hc_mm:.1f} mm out of range "
                f"({rec_assess['status']}), {alt_source} HC {alt_hc_mm:.1f} mm is within range",
            )
    return recommended_hc_mm, rec_source, f"kept: both tools out of range ({rec_assess['status']})"


def _extract_lmp_ga_weeks(text: str) -> Optional[float]:
    if not text:
        return None
    t = text.lower()
    m = re.search(r"ga\s*\(\s*lmp\s*\)\s*is\s*([0-9]+(?:\.[0-9]+)?)", t)
    if m:
        return float(m.group(1))
    m = re.search(
        r"(?:lmp|last menstrual period)[^0-9]{0,30}(\d+)\s*w(?:eeks?)?(?:[^0-9]{0,10}(\d+)\s*d(?:ays?)?)?",
        t,
    )
    if m:
        w = int(m.group(1))
        d = int(m.group(2)) if m.group(2) else 0
        return w + d / 7.0
    return None


# Plane-name display
def _plane_display_name(plane: Optional[str]) -> str:
    p = (plane or "").strip().lower()
    if p == "brain":
        return "Fetal Brain"
    if p == "abdomen":
        return "Fetal Abdomen"
    if p == "thorax":
        return "Fetal Thorax"
    if p == "femur":
        return "Fetal Femur"
    return "Unknown"


def _parse_expert_per_image(expert_outputs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    parsed: Dict[str, Dict[str, Any]] = {}
    for item in expert_outputs:
        task = str(item.get("task") or "")
        txt = item.get("expert_text") or ""
        try:
            data = json.loads(txt)
        except Exception:
            continue
        if isinstance(data, dict) and isinstance(data.get("per_image"), dict):
            parsed[task] = data["per_image"]
    return parsed
