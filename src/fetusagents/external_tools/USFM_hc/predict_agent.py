"""
USFM HC segmentation - Agent-ready version (no interactive input)
Original: test.sh (kept as backup)
Usage: python predict_agent.py --data_path /path/to/images [--pixel_csv /path/to/pixel_size.csv]

This script wraps the USFM framework to run HC segmentation and compute head circumference.
"""
import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

_CKPT_ROOT = Path(os.environ.get("FETALAGENT_CKPT_DIR", str(Path(__file__).resolve().parents[3] / "FetalAgent_ckpt"))).resolve()

# Add the project path for imports
sys.path.insert(0, os.path.dirname(__file__))

try:
    from usdsgen.utils.modules import ellip_fit, mcc_edge
except ImportError:
    # Fallback implementations
    def mcc_edge(img):
        """Extract edge using morphological operations."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(img, kernel, iterations=1)
        eroded = cv2.erode(img, kernel, iterations=1)
        edge = dilated - eroded
        return edge
    
    def ellip_fit(edge_img):
        """Fit ellipse to edge image."""
        contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            return 0, 0, 0, 0, 0
        
        largest = max(contours, key=cv2.contourArea)
        if len(largest) < 5:
            return 0, 0, 0, 0, 0
        
        ellipse = cv2.fitEllipse(largest)
        (xc, yc), (width, height), theta = ellipse
        a = max(width, height) / 2
        b = min(width, height) / 2
        return xc, yc, theta, a, b


def load_pixel_sizes(csv_path):
    """Load pixel sizes from CSV file."""
    pixel_dict = {}
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            pixel_dict[row['filename']] = float(row['pixel size(mm)'])
    return pixel_dict


def stage_images(src_dir, dst_dir):
    """Copy/link images to the expected USFM test directory."""
    os.makedirs(dst_dir, exist_ok=True)
    
    # Clear existing PNG files in dst_dir
    for f in os.listdir(dst_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            fp = os.path.join(dst_dir, f)
            if os.path.isfile(fp):
                os.remove(fp)
    
    # Copy images directly to dst_dir
    count = 0
    for f in os.listdir(src_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))
            count += 1
    
    return count


def run_usfm_inference(work_dir, data_path, ckpt_path, pretrained_path, python_path):
    """Run USFM inference using main.py."""
    cmd = [
        python_path, "main.py",
        "experiment=task/Seg",
        "data=Seg/toy_dataset",
        "data.batch_size=10",
        "data.num_workers=4",
        f"data.path.root={data_path}",
        "data.path.split.test=",
        "model=Seg/SegVit",
        f"model.model_cfg.backbone.pretrained={pretrained_path}",
        f"model.resume={ckpt_path}",
        "mode=test",
        "L.devices=1",
        "tag=USFM_agent",
        f"ckpt_path={ckpt_path}",
    ]
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = env.get('CUDA_VISIBLE_DEVICES', '0')
    # Disable interactive post-processing inside USFM (it prompts via input())
    env["USFM_DISABLE_PROCESS_PREDICTION"] = "1"
    
    # Keep logs captured (not printed) to avoid noisy model-init output
    proc = subprocess.run(
        cmd,
        cwd=work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=3600,
        env=env,
    )
    return proc.returncode, proc.stdout, proc.stderr


def find_latest_output(logs_dir):
    """Find the latest inference_results directory."""
    base = Path(logs_dir) / "finetune" / "Seg" / "toy_dataset" / "SegVit" / "USFM_agent"
    if not base.exists():
        base = Path(logs_dir) / "finetune" / "Seg" / "toy_dataset" / "SegVit" / "USFM"
    
    if not base.exists():
        return None
    
    # Find most recent run
    runs = sorted(base.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
    for run in runs:
        results_dir = run / "outputs" / "inference_results"
        if results_dir.exists():
            return results_dir
    
    return None


def main():
    parser = argparse.ArgumentParser(description='USFM HC Segmentation (Agent-ready)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to directory containing PNG images')
    parser.add_argument('--pixel_csv', type=str, default=None,
                        help='Path to pixel_size.csv (default: <data_path>/pixel_size.csv)')
    parser.add_argument('--ckpt_path', type=str,
                        default=str(_CKPT_ROOT / 'USFM_hc' / 'best235.pth'),
                        help='Path to model checkpoint')
    parser.add_argument('--pretrained_path', type=str,
                        default=str(_CKPT_ROOT / 'USFM' / 'USFM_latest.pth'),
                        help='Path to pretrained backbone')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    args = parser.parse_args()

    work_dir = os.path.dirname(os.path.abspath(__file__))
    python_path = sys.executable
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Load pixel sizes
    pixel_csv = args.pixel_csv if args.pixel_csv else os.path.join(args.data_path, "pixel_size.csv")
    pixel_sizes = load_pixel_sizes(pixel_csv)

    # Stage images to expected location
    test_dir = os.path.join(work_dir, "datasets", "Seg", "toy_dataset", "test_set")
    num_images = stage_images(args.data_path, test_dir)

    if num_images == 0:
        print("ERROR: no images found.", flush=True)
        raise SystemExit(2)

    # Run inference
    retcode, stdout, stderr = run_usfm_inference(
        work_dir=work_dir,
        # Point USFM directly to the folder that contains the staged images
        data_path=test_dir,
        ckpt_path=args.ckpt_path,
        pretrained_path=args.pretrained_path,
        python_path=python_path
    )

    if retcode != 0:
        print(f"ERROR: USFM inference failed (code={retcode}).", flush=True)
        if stderr:
            print(stderr[-3000:], flush=True)
        raise SystemExit(retcode)

    # Find results
    results_dir = find_latest_output(os.path.join(work_dir, "logs"))
    if results_dir is None:
        print("Could not find inference results!")
        return

    # Get original image directory for size info (images are directly in test_dir)
    orig_img_dir = test_dir

    # Process predictions and calculate HC
    results = []
    for pred_file in results_dir.glob("*_pred.png"):
        pred_img = cv2.imread(str(pred_file), cv2.IMREAD_GRAYSCALE)
        if pred_img is None:
            continue
        
        # Extract edge and fit ellipse
        edge_img = mcc_edge(pred_img)
        xc, yc, theta, a, b = ellip_fit(edge_img)
        
        # Get original filename and dimensions
        orig_name = pred_file.stem.replace("_pred", "") + ".png"
        orig_path = os.path.join(orig_img_dir, orig_name)
        
        if os.path.exists(orig_path):
            orig_img = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
            gt_h, gt_w = orig_img.shape[:2]
        else:
            gt_h, gt_w = 512, 512  # Default
        
        # Get pixel size
        pixel_size = pixel_sizes.get(orig_name, 0.15)
        
        # Scale factors (prediction is 512x512)
        pred_h, pred_w = 512, 512
        scale_x = gt_w / pred_w
        scale_y = gt_h / pred_h
        
        # Scale ellipse parameters
        a_scaled = a * scale_x
        b_scaled = b * scale_y
        
        # Calculate HC using ellipse approximation
        hc = 2 * np.pi * b_scaled + 4 * (a_scaled - b_scaled)
        hc_mm = hc * pixel_size
        
        results.append({
            'filename': orig_name,
            'hc_mm': hc_mm,
            'pixel_size': pixel_size
        })

    for r in results:
        print(f"{r['filename']}: {r['hc_mm']:.2f} mm", flush=True)

    if len(results) == 0:
        print("ERROR: no HC results produced (no *_pred.png found or ellipse fit failed).", flush=True)
        raise SystemExit(3)


if __name__ == '__main__':
    main()

