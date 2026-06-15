#!/usr/bin/env python3
"""
Non-interactive agent wrapper for fetalclip_cls_6 key-frame/plane classifier.

It calls the original script and feeds the data path via stdin, then parses:
  File: <filename> | Pred: <plane>
and prints:
  <filename>: <plane>
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from typing import Dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--test_script", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output_csv", required=False, default="")
    return p.parse_args()


def _is_key_from_label(label: str) -> int:
    return 0 if "no plane" in label.lower() else 1


def main() -> int:
    args = parse_args()
    cmd = [sys.executable, args.test_script, "--config", args.config]
    child_env = os.environ.copy()
    child_env["FETALAGENT_KEYFRAME_DATA_PATH"] = os.path.abspath(args.data_path)
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=child_env,
        )
    except Exception as e:
        print(f"[ERROR] Failed to run key-frame classifier: {e}")
        return 1

    text = proc.stdout or ""
    pat = re.compile(r"^\s*File:\s*(?P<fname>.+?)\s*\|\s*Pred:\s*(?P<pred>.+?)\s*$")
    per_image: Dict[str, str] = {}
    for line in text.splitlines():
        m = pat.match(line)
        if not m:
            continue
        fname = m.group("fname").strip()
        pred = m.group("pred").strip()
        per_image[fname] = pred

    if not per_image:
        print("[ERROR] No predictions parsed from key-frame classifier output.")
        if text:
            print(text[-3000:])
        return 2

    for fname, pred in per_image.items():
        print(f"{fname}: {pred}")

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["filename", "pred_plane", "is_key_frame"])
            w.writeheader()
            for fname, pred in per_image.items():
                w.writerow(
                    {
                        "filename": fname,
                        "pred_plane": pred,
                        "is_key_frame": _is_key_from_label(pred),
                    }
                )

    return 0 if proc.returncode == 0 else proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
