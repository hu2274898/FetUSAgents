import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPERNET_DIR = _PROJECT_ROOT / "external_tools" / "UperNet"
if str(UPERNET_DIR) not in sys.path:
    sys.path.insert(0, str(UPERNET_DIR))

from datasets.dataset_aop import JointTransform2D, PNGDataset  # type: ignore
from evafunction import Evaluation  # type: ignore
from nets.upernet import UPerNet  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(description="UperNet AoP prediction (agent-ready, non-interactive)")
    parser.add_argument("--data_path", required=True, help="Folder with input images")
    parser.add_argument(
        "--ckpt_path",
        default=str(_PROJECT_ROOT / "checkpoints" / "upernet_aop_fold0.pth"),
        help="Checkpoint path for UperNet model",
    )
    parser.add_argument("--out_dir", required=True, help="Directory to save predicted masks")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument("--img_size", type=int, default=256, help="Input size")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    args = parser.parse_args()

    if not os.path.isdir(args.data_path):
        print(f"ERROR: data_path not found: {args.data_path}", flush=True)
        raise SystemExit(2)
    if not os.path.isfile(args.ckpt_path):
        print(f"ERROR: checkpoint not found: {args.ckpt_path}", flush=True)
        raise SystemExit(3)

    os.makedirs(args.out_dir, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf_val = JointTransform2D(
        img_size=args.img_size,
        low_img_size=max(1, args.img_size // 2),
        ori_size=args.img_size,
        crop=None,
        p_flip=0.0,
        color_jitter_params=None,
        long_mask=True,
    )
    test_ds = PNGDataset(args.data_path, joint_transform=tf_val, img_size=args.img_size)
    if len(test_ds) == 0:
        print("ERROR: no images found.", flush=True)
        raise SystemExit(4)

    testloader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = UPerNet(3, [3, 4, 6, 3]).to(device)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    with torch.no_grad():
        for datapack in testloader:
            imgs = datapack["image"].to(dtype=torch.float32, device=device)
            preds = model(imgs)
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
