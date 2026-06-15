#!/usr/bin/env python3
"""
nnUNet HC segmentation - non-interactive agent wrapper.

This script wraps the nnUNet inference pipeline for HC segmentation,
providing a command-line interface compatible with the FetalAgent system.

Interface:
    --data_path: Directory containing input images
    --out_dir: Directory to save output masks
    --nnunet_predict: Path to nnUNetv2_predict executable
    --dataset_id: nnUNet dataset ID (default: 503)
    --configuration: nnUNet configuration (default: 2d)
    --fold: nnUNet fold (default: 0)
    --checkpoint: nnUNet checkpoint name (default: checkpoint_best.pth)

Output format (stdout):
    filename.png: /path/to/mask.png
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CKPT_ROOT = Path(os.environ.get("FETALAGENT_CKPT_DIR", str(_PROJECT_ROOT.parent / "FetalAgent_ckpt"))).resolve()
DEFAULT_NNUNET_PREDICT = "nnUNetv2_predict"
DEFAULT_NNUNET_RAW = str(_CKPT_ROOT / "nnUNet" / "nnUNet_raw")
DEFAULT_NNUNET_PREPROCESSED = str(_CKPT_ROOT / "nnUNet" / "nnUNet_preprocessed")
DEFAULT_NNUNET_RESULTS = str(_CKPT_ROOT / "nnUNet" / "nnUNet_results")

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def iter_image_files(data_path: str) -> List[str]:
    return [n for n in sorted(os.listdir(data_path)) if n.lower().endswith(IMAGE_EXTS)]


def to_channel0_name(stem: str) -> str:
    if stem.endswith("_0000"):
        return stem + ".png"
    return stem + "_0000.png"


def preprocess_images(input_dir: str, proceed_dir: str, progress_every: int = 25) -> dict:
    Path(proceed_dir).mkdir(parents=True, exist_ok=True)
    mapping = {}  # nnunet_name -> original_name

    files = iter_image_files(input_dir)
    total = len(files)
    for idx, fname in enumerate(files, start=1):
        fpath = os.path.join(input_dir, fname)
        try:
            img = Image.open(fpath).convert("L")
        except Exception as e:
            print(f"WARN: Failed to load {fname}: {e}", file=sys.stderr, flush=True)
            continue

        stem = Path(fname).stem
        nnunet_name = to_channel0_name(stem)
        out_path = os.path.join(proceed_dir, nnunet_name)
        img.save(out_path)
        mapping[nnunet_name] = fname
        if progress_every > 0 and (idx % progress_every == 0 or idx == total):
            print(f"[nnUNet] Preprocess {idx}/{total}: {fname}", file=sys.stderr, flush=True)

    return mapping


def postprocess_masks(raw_out_dir: str, final_dir: str, name_mapping: dict, progress_every: int = 25) -> dict:
    Path(final_dir).mkdir(parents=True, exist_ok=True)
    results = {}

    reverse_map = {}
    for nnunet_name, orig_name in name_mapping.items():
        base = nnunet_name.replace("_0000.png", ".png")
        reverse_map[base] = orig_name

    out_files = sorted(Path(raw_out_dir).glob("*.png"))
    total = len(out_files)
    for idx, fpath in enumerate(out_files, start=1):
        fname = fpath.name
        orig_name = reverse_map.get(fname)
        if not orig_name:
            stem = fpath.stem
            for nnunet_name, orig in name_mapping.items():
                if stem in nnunet_name or nnunet_name.startswith(stem):
                    orig_name = orig
                    break

        if not orig_name:
            print(f"WARN: Cannot map nnUNet output {fname} to original", file=sys.stderr, flush=True)
            continue

        try:
            arr = np.array(Image.open(fpath))
            arr_post = arr.copy()
            arr_post[arr_post == 1] = 255

            orig_stem = Path(orig_name).stem
            out_name = f"{orig_stem}_prediction.png"
            out_path = os.path.join(final_dir, out_name)
            Image.fromarray(arr_post.astype(np.uint8)).save(out_path)
            results[orig_name] = out_path
            if progress_every > 0 and (idx % progress_every == 0 or idx == total):
                print(f"[nnUNet] Postprocess {idx}/{total}: {orig_name}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"WARN: Failed to postprocess {fname}: {e}", file=sys.stderr, flush=True)

    return results


def run_nnunet_predict(
    nnunet_predict_path: str,
    input_dir: str,
    output_dir: str,
    dataset_id: int,
    configuration: str,
    fold: int,
    checkpoint: str,
    timeout: Optional[int] = None,
) -> bool:
    env = os.environ.copy()
    env["nnUNet_raw"] = DEFAULT_NNUNET_RAW
    env["nnUNet_preprocessed"] = DEFAULT_NNUNET_PREPROCESSED
    env["nnUNet_results"] = DEFAULT_NNUNET_RESULTS

    cmd = [
        nnunet_predict_path,
        "-i", input_dir,
        "-o", output_dir,
        "-d", str(dataset_id),
        "-c", configuration,
        "-f", str(fold),
        "-chk", checkpoint,
    ]

    print(f"Running: {' '.join(cmd)}", file=sys.stderr, flush=True)

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            print(f"nnUNet stderr: {result.stderr}", file=sys.stderr, flush=True)
            return False
        return True
    except subprocess.TimeoutExpired:
        print("ERROR: nnUNet prediction timed out", file=sys.stderr, flush=True)
        return False
    except Exception as e:
        print(f"ERROR: Failed to run nnUNet: {e}", file=sys.stderr, flush=True)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="nnUNet HC segmentation - agent entrypoint")
    parser.add_argument("--data_path", required=True, help="Directory containing input images")
    parser.add_argument("--out_dir", default="", help="Output directory for masks")
    parser.add_argument("--nnunet_predict", default=DEFAULT_NNUNET_PREDICT)
    parser.add_argument("--dataset_id", type=int, default=503)
    parser.add_argument("--configuration", default="2d")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--checkpoint", default="checkpoint_best.pth")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--keep_temp", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.data_path):
        print(f"ERROR: data_path not found: {args.data_path}", flush=True)
        sys.exit(2)

    nnunet_predict_path = args.nnunet_predict
    if os.path.sep not in nnunet_predict_path:
        resolved = shutil.which(nnunet_predict_path)
        if resolved:
            nnunet_predict_path = resolved
    if not os.path.isfile(nnunet_predict_path):
        print(f"ERROR: nnUNetv2_predict not found: {args.nnunet_predict}", flush=True)
        sys.exit(2)

    out_dir = args.out_dir.strip()
    if not out_dir:
        agent_root = Path(__file__).resolve().parents[1]
        out_dir = str(agent_root / "outputs_agent" / "hc" / "nnunet")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    temp_base = tempfile.mkdtemp(prefix="nnunet_hc_")
    proceed_dir = os.path.join(temp_base, "proceed")
    raw_out_dir = os.path.join(temp_base, "raw_output")
    Path(raw_out_dir).mkdir(parents=True, exist_ok=True)

    try:
        print(f"Preprocessing images from {args.data_path}...", file=sys.stderr, flush=True)
        name_mapping = preprocess_images(args.data_path, proceed_dir, progress_every=args.progress_every)
        if not name_mapping:
            print("ERROR: No images found to process", flush=True)
            sys.exit(3)

        print("Running nnUNet prediction...", file=sys.stderr, flush=True)
        success = run_nnunet_predict(
            nnunet_predict_path=nnunet_predict_path,
            input_dir=proceed_dir,
            output_dir=raw_out_dir,
            dataset_id=args.dataset_id,
            configuration=args.configuration,
            fold=args.fold,
            checkpoint=args.checkpoint,
            timeout=args.timeout,
        )
        if not success:
            print("ERROR: nnUNet prediction failed", flush=True)
            sys.exit(4)

        print("Post-processing masks...", file=sys.stderr, flush=True)
        results = postprocess_masks(raw_out_dir, out_dir, name_mapping, progress_every=args.progress_every)
        if not results:
            print("ERROR: No masks generated", flush=True)
            sys.exit(5)

        for orig_name, mask_path in sorted(results.items()):
            print(f"{orig_name}: {mask_path}", flush=True)

        print(f"Done. Generated {len(results)} masks.", file=sys.stderr, flush=True)

    finally:
        if not args.keep_temp and os.path.exists(temp_base):
            shutil.rmtree(temp_base, ignore_errors=True)


if __name__ == "__main__":
    main()
