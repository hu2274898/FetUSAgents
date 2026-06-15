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
SAMUS_PROJ_DIR = str(_PROJECT_ROOT / "external_tools" / "fetalclip_pred_stomach")

DEFAULT_FETALCLIP_CKPT = str(_CKPT_ROOT / "stomach_fetalclip.ckpt")
DEFAULT_SAMUS_CKPT = str(_CKPT_ROOT / "stomach_samus.pth")
DEFAULT_SAM_BASE = str(_CKPT_ROOT / "SAMUS.pth")


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


_preprocessing224 = A.Compose(
    [
        A.Resize(224, 224, interpolation=cv2.INTER_CUBIC, mask_interpolation=0, p=1.0),
    ]
)


class _ImageFolder224(torch.utils.data.Dataset):
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
        img = _preprocessing224(image=img)["image"]
        img = Image.fromarray(img)
        img = self.preprocess_img(img)
        return img, fn


def _largest_cc(binary255: np.ndarray) -> np.ndarray:
    # binary255: uint8, 0/255
    if binary255 is None or binary255.size == 0:
        return binary255
    _, bin01 = cv2.threshold(binary255, 127, 1, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin01, connectivity=8)
    if num_labels <= 1:
        return (bin01 * 255).astype(np.uint8)
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    clean = (labels == largest_label).astype(np.uint8) * 255
    return clean


def _fixed_click(mask01: np.ndarray) -> tuple[np.ndarray, int]:
    # returns (pt_xy[1,2], point_label)
    idx = np.argwhere(mask01 == 1)
    point_label = 1
    if idx.shape[0] == 0:
        point_label = 0
        # fall back to a valid point
        pt = np.array([[128, 128]], dtype=np.int64)
        return pt, point_label
    # convert (y,x) -> (x,y)
    idx[:, [0, 1]] = idx[:, [1, 0]]
    pt = idx[len(idx) // 2][None, :]
    return pt, point_label


def main() -> None:
    sys.dont_write_bytecode = True

    parser = argparse.ArgumentParser(description="Stomach segmentation (FetalCLIP -> SAMUS) - agent entrypoint")
    parser.add_argument("--data_path", required=True, help="Folder containing images")
    parser.add_argument("--fetalclip_ckpt", default=DEFAULT_FETALCLIP_CKPT, help="FetalCLIP seg checkpoint")
    parser.add_argument("--samus_ckpt", default=DEFAULT_SAMUS_CKPT, help="SAMUS fine-tuned checkpoint")
    parser.add_argument("--sam_base", default=DEFAULT_SAM_BASE, help="Base SAMUS checkpoint (vit_b)")
    parser.add_argument("--out_dir", default="", help="Output directory for masks")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for coarse seg (default: 2)")
    args = parser.parse_args()

    if not os.path.isdir(args.data_path):
        print(f"ERROR: data_path not found: {args.data_path}", flush=True)
        raise SystemExit(2)

    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")

    # --- Load coarse FetalCLIP segmentation model (224x224) ---
    os.chdir(SEG_PROJ_DIR)
    sys.path.insert(0, SEG_PROJ_DIR)
    from utilsseg import LitModel, INIT_FILTERS  # noqa: E402
    from utils_get_embeddings import EncoderWrapper  # noqa: E402

    with open(os.path.join(SEG_PROJ_DIR, "config.json"), "r") as f:
        cfg = json.load(f)
    path_fetalclip_weight = str(_CKPT_ROOT / "FetalCLIP_weights.pt")
    path_fetalclip_config = str(_CKPT_ROOT / "FetalCLIP_config.json")

    with open(path_fetalclip_config, "r") as f:
        fetalclip_cfg = json.load(f)
    open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = fetalclip_cfg
    clip_model, _, preprocess_img = open_clip.create_model_and_transforms(
        "FetalCLIP", pretrained=path_fetalclip_weight
    )
    visual = clip_model.visual.eval().to(device)
    encoder = EncoderWrapper(visual).eval().to(device)

    coarse_model = LitModel(encoder.transformer.width, 1, 3, INIT_FILTERS).eval().to(device)
    ckpt = torch.load(args.fetalclip_ckpt, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    coarse_model.load_state_dict(state, strict=False)

    # --- Load SAMUS model (256x256) ---
    sys.path.insert(0, SAMUS_PROJ_DIR)
    from models.segment_anything_samus.build_sam_us import samus_model_registry  # noqa: E402

    samus = samus_model_registry["vit_b"](checkpoint=args.sam_base).to(device).eval()
    sam_state = torch.load(args.samus_ckpt, map_location="cpu")
    # strip 'module.' if present
    new_state = {k[7:]: v for k, v in sam_state.items()} if any(k.startswith("module.") for k in sam_state) else sam_state
    samus.load_state_dict(new_state, strict=False)

    out_dir = args.out_dir.strip()
    if not out_dir:
        # Default to a writable folder under FetalAgent_hjw/
        agent_root = Path(__file__).resolve().parents[1]
        out_dir = str(agent_root / "outputs_agent" / "stomach_seg" / "fetalclip_samus")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    ds = _ImageFolder224(args.data_path, preprocess_img)
    if len(ds) == 0:
        print("ERROR: no images found.", flush=True)
        raise SystemExit(3)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # We need original paths for SAMUS image loading
    with torch.no_grad():
        for imgs, filenames in dl:
            imgs = imgs.to(device)
            embs = encoder(imgs)
            logits = coarse_model([imgs, embs["z3"], embs["z6"], embs["z9"], embs["z12"]])
            probs = torch.sigmoid(logits)
            coarse_bin = (probs > 0.5).to(torch.uint8).squeeze(1).detach().cpu().numpy() * 255  # (B,H,W)

            for fn, coarse_mask in zip(filenames, coarse_bin):
                # Clean mask + resize to 256 for prompt
                clean_mask = _largest_cc(coarse_mask.astype(np.uint8))
                clean256 = cv2.resize(clean_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
                mask01 = (clean256 > 127).astype(np.uint8)
                pt_xy, point_label = _fixed_click(mask01)

                # Build SAMUS point prompt tensors: (B,N,2) and (B,N)
                coords = torch.as_tensor(pt_xy, dtype=torch.float32, device=device).unsqueeze(0)  # (1,1,2)
                labels = torch.as_tensor([[point_label]], dtype=torch.int64, device=device)  # (1,1)
                pt = (coords, labels)

                # Load image as grayscale -> tensor (1,1,256,256) in [0,1]
                img_path = os.path.join(args.data_path, fn)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"{fn}: ERROR unreadable image", flush=True)
                    continue
                img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
                img_t = torch.from_numpy(img).float().div(255.0).unsqueeze(0).unsqueeze(0).to(device)

                pred = samus(img_t, pt)
                sam_prob = torch.sigmoid(pred["masks"])
                sam_mask = (sam_prob > 0.5).to(torch.uint8).squeeze(0).squeeze(0).detach().cpu().numpy() * 255

                base = os.path.splitext(fn)[0]
                out_path = os.path.join(out_dir, f"{base}_samus.png")
                Image.fromarray(sam_mask.astype(np.uint8)).save(out_path)
                print(f"{fn}: {out_path}", flush=True)


if __name__ == "__main__":
    main()




