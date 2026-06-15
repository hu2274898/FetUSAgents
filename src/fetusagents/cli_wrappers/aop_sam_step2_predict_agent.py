import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CKPT_ROOT = Path(os.environ.get("FETALAGENT_CKPT_DIR", str(_PROJECT_ROOT.parent / "FetalAgent_ckpt"))).resolve()
STEP2_DIR = _PROJECT_ROOT / "external_tools" / "AoP_SAM"
if str(STEP2_DIR) not in sys.path:
    sys.path.insert(0, str(STEP2_DIR))

from models.model_dict import get_model  # type: ignore
from utils.data_us import JointTransform2D, PNGDataset  # type: ignore
from evafunction import Evaluation  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(description="AoP-SAM step2 prediction (agent-ready, non-interactive)")
    parser.add_argument("--data_path", required=True, help="Folder with input images")
    parser.add_argument(
        "--sam_ckpt",
        default=str(_CKPT_ROOT / "aop_sam_fold0.pth"),
        help="Checkpoint path for AoP-SAM step2 model",
    )
    parser.add_argument("--out_dir", required=True, help="Directory to save predicted masks")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument("--encoder_input_size", type=int, default=256)
    parser.add_argument("--low_image_size", type=int, default=128)
    parser.add_argument("--task", type=str, default="PSFH")
    parser.add_argument("--vit_name", type=str, default="vit_b")
    args = parser.parse_args()

    if not os.path.isdir(args.data_path):
        print(f"ERROR: data_path not found: {args.data_path}", flush=True)
        raise SystemExit(2)
    if not os.path.isfile(args.sam_ckpt):
        print(f"ERROR: checkpoint not found: {args.sam_ckpt}", flush=True)
        raise SystemExit(3)

    os.makedirs(args.out_dir, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    seed_value = 2023
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf_val = JointTransform2D(
        img_size=args.encoder_input_size,
        low_img_size=args.low_image_size,
        ori_size=256,
        crop=None,
        p_flip=0.0,
        color_jitter_params=None,
        long_mask=True,
    )
    test_ds = PNGDataset(args.data_path, joint_transform=tf_val, img_size=args.encoder_input_size)
    if len(test_ds) == 0:
        print("ERROR: no images found.", flush=True)
        raise SystemExit(4)

    testloader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

    model = get_model(args=args)
    ckpt = torch.load(args.sam_ckpt, map_location=device)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    with torch.no_grad():
        for datapack in testloader:
            imgs = datapack["image"].to(dtype=torch.float32, device=device)
            preds = model(imgs, None, None, None)["masks"]
            names = datapack.get("image_name", [])

            for i in range(preds.shape[0]):
                pred = preds[i].argmax(dim=0).detach().cpu().numpy().astype(np.uint8)
                fname = str(names[i]) if i < len(names) else f"sample_{i}.png"
                out_path = os.path.join(args.out_dir, fname)
                cv2.imwrite(out_path, pred)

                aop_value = float(Evaluation(pred).process()["aop_truth"])
                print(f"{fname}: {aop_value:.2f} deg | mask: {out_path}", flush=True)


if __name__ == "__main__":
    main()
