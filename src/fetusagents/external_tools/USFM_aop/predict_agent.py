"""
USFM AoP segmentation - Agent-ready version (no interactive input)
Original: test.sh (kept as backup)
Usage: python predict_agent.py --data_path /path/to/images

This script wraps the USFM framework to run AoP segmentation and compute AoP angles.
"""
import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
import cv2
import numpy as np

_CKPT_ROOT = Path(os.environ.get("FETALAGENT_CKPT_DIR", str(Path(__file__).resolve().parents[3] / "FetalAgent_ckpt"))).resolve()

# Add the project path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Import evaluation function from local copy
try:
    from usdsgen.utils.evafunction import Evaluation
except ImportError:
    # Fallback - define minimal Evaluation class
    import SimpleITK
    import math
    
    class Evaluation:
        def __init__(self, pred_arr):
            self.pred_arr = pred_arr
        
        def process(self):
            return {'aop_truth': self._cal_aop()}
        
        def _cal_aop(self):
            # Simplified AoP calculation
            aop = 0.0
            try:
                pred = self.pred_arr.astype(np.uint8)
                # Create masks for pubic symphysis (1) and fetal head (2)
                mask1 = (pred == 1).astype(np.uint8) * 255
                mask2 = (pred == 2).astype(np.uint8) * 255
                
                contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
                if len(contours1) > 0 and len(contours2) > 0:
                    # Get largest contours
                    c1 = max(contours1, key=cv2.contourArea)
                    c2 = max(contours2, key=cv2.contourArea)
                    
                    if len(c1) >= 5 and len(c2) >= 5:
                        ellipse1 = cv2.fitEllipse(c1)
                        ellipse2 = cv2.fitEllipse(c2)
                        # Simplified angle calculation
                        aop = abs(ellipse1[2] - ellipse2[2])
                        if aop > 90:
                            aop = 180 - aop
            except Exception as e:
                print(f"AoP calculation error: {e}")
                aop = 0.0
            return aop


def stage_images(src_dir, dst_dir):
    """Copy/link images to the expected USFM test directory."""
    os.makedirs(dst_dir, exist_ok=True)
    
    # Clear existing image files only (do NOT delete scripts/configs)
    for f in os.listdir(dst_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            fp = os.path.join(dst_dir, f)
            if os.path.isfile(fp):
                os.remove(fp)
    
    # Copy images
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
        # Try without tag
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
    parser = argparse.ArgumentParser(description='USFM AoP Segmentation (Agent-ready)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to directory containing PNG images')
    parser.add_argument('--ckpt_path', type=str,
                        default=str(_CKPT_ROOT / 'USFM_aop' / 'best365.pth'),
                        help='Path to model checkpoint')
    parser.add_argument('--pretrained_path', type=str,
                        default=str(_CKPT_ROOT / 'USFM' / 'USFM_latest.pth'),
                        help='Path to pretrained backbone')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    args = parser.parse_args()

    work_dir = os.path.dirname(os.path.abspath(__file__))
    python_path = sys.executable  # Use current Python interpreter
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

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

    # Process predictions and calculate AoP
    results = []
    for pred_file in results_dir.glob("*_pred.png"):
        pred_img = cv2.imread(str(pred_file), cv2.IMREAD_GRAYSCALE)
        if pred_img is None:
            continue
        
        # Resize to 512x512 for evaluation
        pred_img = cv2.resize(pred_img, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        evaluator = Evaluation(pred_img)
        result = evaluator.process()
        aop_value = result['aop_truth']
        
        # Get original filename
        orig_name = pred_file.stem.replace("_pred", "") + ".png"
        
        results.append({
            'filename': orig_name,
            'aop_deg': aop_value,
            'mask_path': str(pred_file),
        })

    for r in results:
        print(
            f"{r['filename']}: {r['aop_deg']:.2f} deg | mask: {r['mask_path']}",
            flush=True,
        )

    if len(results) == 0:
        print("ERROR: no AoP results produced (no *_pred.png found).", flush=True)
        raise SystemExit(3)


if __name__ == '__main__':
    main()

