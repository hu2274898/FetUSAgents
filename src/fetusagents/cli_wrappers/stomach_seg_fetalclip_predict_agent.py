import argparse
import json
import os
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from PIL import Image
import torch
import open_clip


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CKPT_ROOT = Path(os.environ.get("FETALAGENT_CKPT_DIR", str(_PROJECT_ROOT.parent / "FetalAgent_ckpt"))).resolve()
SEG_PROJ_DIR = str(_PROJECT_ROOT / "external_tools" / "FetalCLIP_seg_stomach")
DEFAULT_CKPT = str(_CKPT_ROOT / "stomach_fetalclip.ckpt")


def _iter_image_files(data_path: str):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    return [n for n in sorted(os.listdir(data_path)) if n.lower().endswith(exts)]


def _make_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    max_side = max(width, height)
    if image.mode == "RGBA":
        image = image.convert("RGB")
    if image.mode == "RGB":
        padding_color = (0, 0, 0)
    else:
        padding_color = 0
    new_image = Image.new(image.mode, (max_side, max_side), padding_color)
    padding_left = (max_side - width) // 2
    padding_top = (max_side - height) // 2
    new_image.paste(image, (padding_left, padding_top))
    return new_image


_preprocessing = A.Compose(
    [
        A.Resize(224, 224, interpolation=cv2.INTER_CUBIC, mask_interpolation=0, p=1.0),
    ]
)


class _ImageFolder(torch.utils.data.Dataset):
    def __init__(self, root: str, preprocess_img):
        self.root = root
        self.preprocess_img = preprocess_img
        self.files = _iter_image_files(root)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        fn = self.files[idx]
        path = os.path.join(self.root, fn)
        data = Image.open(path)
        if data is None:
            raise FileNotFoundError(f"image not found: {path}")
        img = _make_square(data)
        img = np.array(img)
        img = _preprocessing(image=img)["image"]
        img = Image.fromarray(img)
        img = self.preprocess_img(img)
        return img, fn


def main() -> None:
    # Avoid writing pyc into non-writable project dirs
    sys.dont_write_bytecode = True

    parser = argparse.ArgumentParser(description="Stomach segmentation (FetalCLIP) - agent entrypoint")
    parser.add_argument("--data_path", required=True, help="Folder containing images")
    parser.add_argument("--ckpt_path", default=DEFAULT_CKPT, help="FetalCLIP segmentation checkpoint")
    parser.add_argument("--out_dir", default="", help="Output directory for masks")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    args = parser.parse_args()

    if not os.path.isdir(args.data_path):
        print(f"ERROR: data_path not found: {args.data_path}", flush=True)
        raise SystemExit(2)

    # Ensure we can import segmentation_stomach modules that assume cwd==project dir
    os.chdir(SEG_PROJ_DIR)
    sys.path.insert(0, SEG_PROJ_DIR)

    from utilsseg import LitModel, INIT_FILTERS  # noqa: E402
    from utils_get_embeddings import EncoderWrapper  # noqa: E402

    # Load config.json for base FetalCLIP weights/config
    with open(os.path.join(SEG_PROJ_DIR, "config.json"), "r") as f:
        cfg = json.load(f)
    path_fetalclip_weight = str(_CKPT_ROOT / "FetalCLIP_weights.pt")
    path_fetalclip_config = str(_CKPT_ROOT / "FetalCLIP_config.json")

    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")

    with open(path_fetalclip_config, "r") as f:
        fetalclip_cfg = json.load(f)
    open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = fetalclip_cfg
    clip_model, _, preprocess_img = open_clip.create_model_and_transforms(
        "FetalCLIP", pretrained=path_fetalclip_weight
    )
    visual = clip_model.visual.eval().to(device)
    encoder = EncoderWrapper(visual).eval().to(device)

    seg_model = LitModel(encoder.transformer.width, 1, 3, INIT_FILTERS).eval().to(device)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    seg_model.load_state_dict(state, strict=False)

    out_dir = args.out_dir.strip()
    if not out_dir:
        # Default to a writable folder under FetalAgent_hjw/
        agent_root = Path(__file__).resolve().parents[1]
        out_dir = str(agent_root / "outputs_agent" / "stomach_seg" / "fetalclip")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    ds = _ImageFolder(args.data_path, preprocess_img)
    if len(ds) == 0:
        print("ERROR: no images found.", flush=True)
        raise SystemExit(3)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    with torch.no_grad():
        for imgs, filenames in dl:
            imgs = imgs.to(device)
            embs = encoder(imgs)
            logits = seg_model([imgs, embs["z3"], embs["z6"], embs["z9"], embs["z12"]])
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).to(torch.uint8) * 255  # (B,1,H,W)
            preds = preds.squeeze(1).detach().cpu().numpy()
            for fn, mask in zip(filenames, preds):
                base = os.path.splitext(fn)[0]
                out_path = os.path.join(out_dir, f"{base}_prediction.png")
                Image.fromarray(mask).save(out_path)
                print(f"{fn}: {out_path}", flush=True)


if __name__ == "__main__":
    main()




