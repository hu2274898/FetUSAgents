import argparse
import json
import os
import sys
from typing import List
from pathlib import Path

import cv2
import numpy as np
import torch
import open_clip
from PIL import Image
import albumentations as A

_CKPT_ROOT = Path(os.environ.get("FETALAGENT_CKPT_DIR", str(Path(__file__).resolve().parents[3] / "FetalAgent_ckpt"))).resolve()

def _iter_image_files(data_path: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    files = []
    for name in sorted(os.listdir(data_path)):
        if name.lower().endswith(exts):
            files.append(name)
    return files


_preprocessing = A.Compose(
    [
        A.Resize(224, 224, interpolation=cv2.INTER_CUBIC, mask_interpolation=0, p=1.0),
    ]
)


class _ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, preprocess_img, make_square_fn):
        self.root = root
        self.preprocess_img = preprocess_img
        self._make_square = make_square_fn
        self.files = _iter_image_files(root)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        fn = self.files[idx]
        path = os.path.join(self.root, fn)
        data = Image.open(path)
        if data is None:
            raise FileNotFoundError(f"image not found: {path}")
        img = self._make_square(data)
        img = np.array(img)
        img = _preprocessing(image=img)["image"]
        img = Image.fromarray(img)
        img = self.preprocess_img(img)
        return img, fn


def main() -> None:
    # Avoid pyc writes into potentially non-writable __pycache__ folders
    sys.dont_write_bytecode = True
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # utils1.py expects config.json in CWD; make the script self-contained
    os.chdir(script_dir)
    sys.path.insert(0, script_dir)

    from utils1 import LitModel, DICT_CLSNAME_TO_CLSINDEX, make_image_square_with_zero_padding  # noqa: E402

    parser = argparse.ArgumentParser(description="Brain subplane classification (FetalCLIP) - agent entrypoint")
    parser.add_argument("--data_path", required=True, help="Folder containing images")
    parser.add_argument(
        "--ckpt_path",
        default=str(_CKPT_ROOT / "brain_subplane_fetalclip.ckpt"),
        help="Lightning checkpoint for the linear head",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    args = parser.parse_args()

    if not os.path.isdir(args.data_path):
        print(f"ERROR: data_path not found: {args.data_path}", flush=True)
        raise SystemExit(2)

    # Resolve device
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")

    # Load repo config.json (paths to FetalCLIP weights/config)
    with open(os.path.join(script_dir, "config.json"), "r") as f:
        cfg = json.load(f)
    path_fetalclip_weight = cfg["paths"]["path_fetalclip_weight"]
    path_fetalclip_config = cfg["paths"]["path_fetalclip_config"]

    # Load FetalCLIP visual encoder
    if not os.path.isabs(path_fetalclip_config):
        rel = os.path.join(script_dir, path_fetalclip_config)
        path_fetalclip_config = rel if os.path.exists(rel) else str(_CKPT_ROOT / "FetalCLIP_config.json")
    if not os.path.isabs(path_fetalclip_weight):
        rel = os.path.join(script_dir, path_fetalclip_weight)
        path_fetalclip_weight = rel if os.path.exists(rel) else str(_CKPT_ROOT / "FetalCLIP_weights.pt")
    with open(path_fetalclip_config, "r") as f:
        fetalclip_cfg = json.load(f)
    open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = fetalclip_cfg
    clip_model, _, preprocess_img = open_clip.create_model_and_transforms(
        "FetalCLIP", pretrained=path_fetalclip_weight
    )
    encoder = clip_model.visual.eval().to(device)

    # Build classifier head + load checkpoint
    input_dim = int(getattr(encoder, "proj").shape[1])
    num_classes = len(DICT_CLSNAME_TO_CLSINDEX)
    head = LitModel(input_dim, num_classes).eval().to(device)

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = head.load_state_dict(state, strict=False)
    # If the checkpoint is wrong, surface as a clean error line
    if len(unexpected) > 0 or (len(missing) > 0 and any(k.startswith("head.") for k in missing)):
        print(f"ERROR: failed to load head checkpoint: {args.ckpt_path}", flush=True)
        raise SystemExit(3)

    # Dataset / loader (filter images only; ignore CSV/etc)
    ds = _ImageFolderDataset(args.data_path, preprocess_img, make_image_square_with_zero_padding)
    if len(ds) == 0:
        print("ERROR: no images found.", flush=True)
        raise SystemExit(4)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    idx_to_name = {v: k for k, v in DICT_CLSNAME_TO_CLSINDEX.items()}

    # Inference
    with torch.no_grad():
        for imgs, filenames in dl:
            imgs = imgs.to(device)
            feats = encoder(imgs)
            logits = head(feats)
            probs = torch.softmax(logits, dim=1).detach().cpu()
            pred_idx = torch.argmax(logits, dim=1).detach().cpu().tolist()
            for fn, pi, prob_vec in zip(filenames, pred_idx, probs):
                label = idx_to_name.get(int(pi), "UNKNOWN")
                prob_str = " ".join([f"{p:.6f}" for p in prob_vec.tolist()])
                print(f"{fn}: {label} [{prob_str}]", flush=True)


if __name__ == "__main__":
    main()


