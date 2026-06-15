import argparse
import csv
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
from peft import get_peft_model, LoraConfig


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CKPT_ROOT = Path(os.environ.get("FETALAGENT_CKPT_DIR", str(_PROJECT_ROOT.parent / "FetalAgent_ckpt"))).resolve()
PROJ_DIR = str(_PROJECT_ROOT / "external_tools" / "fetalclip_seg_ac")
DEFAULT_CKPT = str(_CKPT_ROOT / "abdomen_fetalclip.ckpt")
DEFAULT_FETALCLIP_CONFIG = str(_PROJECT_ROOT / "external_tools" / "fetalclip_seg_ac" / "FetalCLIP_config.json")
DEFAULT_FETALCLIP_WEIGHTS = str(_CKPT_ROOT / "FetalCLIP_weights.pt")


def _iter_image_files(data_path: str):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    return [n for n in sorted(os.listdir(data_path)) if n.lower().endswith(exts)]


def _load_ac_info(csv_path: str) -> dict:
    info = {}
    if not csv_path or not os.path.exists(csv_path):
        return info
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = (row.get("Filename") or row.get("filename") or "").strip()
                if not name:
                    continue
                try:
                    pixel_size = float(row.get("pixel_size", ""))
                except Exception:
                    pixel_size = float("nan")
                try:
                    ac_gt_mm = float(row.get("AC_value_mm", ""))
                except Exception:
                    ac_gt_mm = float("nan")
                info[name] = {"pixel_size": pixel_size, "ac_gt_mm": ac_gt_mm}
    except Exception:
        return {}
    return info


def _mcc_edge(mask_255: np.ndarray) -> np.ndarray:
    if mask_255 is None or mask_255.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    img = mask_255.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=4)
    if n_labels <= 1:
        return np.zeros_like(img, dtype=np.uint8)
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    max_cc = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    return cv2.Canny(max_cc, 50, 250)


def _ellip_fit(edge_img: np.ndarray):
    ys, xs = np.where(edge_img == 255)
    if len(xs) < 20:
        return None
    x = ys.reshape(-1, 1).astype(np.float64)
    y = xs.reshape(-1, 1).astype(np.float64)
    x2 = x * x
    xy = 2 * x * y
    _2x = -2 * x
    _2y = -2 * y
    minus_1 = -np.ones_like(x)
    X = np.concatenate((x2, xy, _2x, _2y, minus_1), axis=1)
    rhs = -(y * y)
    coef, *_ = np.linalg.lstsq(X, rhs, rcond=None)
    k1, k2, k3, k4, k5 = [float(v) for v in coef.reshape(-1)]
    den = (k1 - k2 * k2)
    if abs(den) < 1e-12:
        return None
    xc = (k3 - k2 * k4) / den
    yc = (k1 * k4 - k2 * k3) / den
    theta = 0.5 * np.arctan2(2 * k2, (k1 - 1.0))
    T = np.tan(theta)
    den_k = (k1 - T * T)
    if abs(den_k) < 1e-12:
        return None
    K = (1 - k1 * T * T) / den_k
    den_b = (K * (T * T + 1))
    if abs(den_b) < 1e-12:
        return None
    p1 = -np.square(xc + T * yc)
    p2 = -np.square(xc * T - yc)
    b2 = (k5 * (T * T + K) - p1 - p2 * K) / den_b
    a2 = K * b2
    if a2 <= 0 or b2 <= 0:
        return None
    a = float(np.sqrt(a2))
    b = float(np.sqrt(b2))
    if a < b:
        a, b = b, a
    return a, b


def _compute_ac_px(mask_255: np.ndarray) -> float:
    edge_img = _mcc_edge(mask_255)
    fit = _ellip_fit(edge_img)
    if fit is None:
        return float("nan")
    a, b = fit
    return float(2.0 * np.pi * b + 4.0 * (a - b))


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


def main() -> None:
    sys.dont_write_bytecode = True

    parser = argparse.ArgumentParser(description="Abdomen segmentation (FetalCLIP) - agent entrypoint")
    parser.add_argument("--data_path", required=True, help="Folder containing images")
    parser.add_argument("--ckpt_path", default=DEFAULT_CKPT, help="FetalCLIP segmentation checkpoint")
    parser.add_argument("--out_dir", default="", help="Output directory for masks")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument(
        "--ac_info_csv",
        default="",
        help="Optional AC metadata CSV with columns Filename,pixel_size,AC_value_mm",
    )
    parser.add_argument(
        "--ac_out_csv",
        default="",
        help="Optional output CSV for AC predictions; defaults to <out_dir>/ac_predictions.csv",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_path):
        print(f"ERROR: data_path not found: {args.data_path}", flush=True)
        raise SystemExit(2)

    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")

    # Import project modules (for SegmentationModel + EncoderWrapper)
    os.chdir(PROJ_DIR)
    sys.path.insert(0, PROJ_DIR)
    from embeddings import EncoderWrapper  # noqa: E402
    from model import SegmentationModel  # noqa: E402

    # Load FetalCLIP base config/weights
    with open(DEFAULT_FETALCLIP_CONFIG, "r") as f:
        fetalclip_cfg = json.load(f)
    open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = fetalclip_cfg
    clip_model, _, preprocess_img = open_clip.create_model_and_transforms(
        "FetalCLIP", pretrained=DEFAULT_FETALCLIP_WEIGHTS
    )
    visual = clip_model.visual

    # Model init
    tmp = EncoderWrapper(visual)
    input_dim = int(tmp.transformer.width)
    model = SegmentationModel(
        visual,
        input_dim=input_dim,
        num_classes=1,
        freeze_encoder=False,
        root_path=args.data_path,
    )

    # Apply LoRA (matches training scripts)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_fc", "c_proj", "out_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    model.encoder = get_peft_model(model.encoder, lora_config)

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)

    model = model.to(device).eval()

    out_dir = args.out_dir.strip()
    if not out_dir:
        agent_root = Path(__file__).resolve().parents[1]
        out_dir = str(agent_root / "outputs_agent" / "abdomen_seg" / "fetalclip")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    ac_info_csv = args.ac_info_csv.strip()
    if not ac_info_csv:
        auto_csv = os.path.join(os.path.dirname(os.path.abspath(args.data_path)), "ac_detail_info.csv")
        ac_info_csv = auto_csv if os.path.exists(auto_csv) else ""
    ac_info = _load_ac_info(ac_info_csv)
    ac_rows = []

    ds = _ImageFolder224(args.data_path, preprocess_img)
    if len(ds) == 0:
        print("ERROR: no images found.", flush=True)
        raise SystemExit(3)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    n_out = 0
    with torch.no_grad():
        for imgs, filenames in dl:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            masks = (probs > 0.5).to(torch.uint8).squeeze(1).detach().cpu().numpy() * 255  # (B,H,W)

            for fn, mask in zip(filenames, masks):
                base = os.path.splitext(fn)[0]
                out_path = os.path.join(out_dir, f"{base}.png")
                mask_u8 = mask.astype(np.uint8)
                Image.fromarray(mask_u8).save(out_path)
                print(f"{fn}: {out_path}", flush=True)
                if fn in ac_info:
                    pixel_size = ac_info[fn].get("pixel_size", float("nan"))
                    ac_px = _compute_ac_px(mask_u8)
                    ac_mm = float("nan") if np.isnan(ac_px) or np.isnan(pixel_size) else float(ac_px * pixel_size)
                    ac_gt = ac_info[fn].get("ac_gt_mm", float("nan"))
                    ac_err = float("nan") if np.isnan(ac_mm) or np.isnan(ac_gt) else abs(ac_mm - ac_gt)
                    ac_rows.append(
                        {
                            "image_name": fn,
                            "mask_path": out_path,
                            "pixel_size": "" if np.isnan(pixel_size) else f"{pixel_size:.10f}",
                            "ac_pred_mm": "" if np.isnan(ac_mm) else f"{ac_mm:.6f}",
                            "ac_gt_mm": "" if np.isnan(ac_gt) else f"{ac_gt:.6f}",
                            "ac_abs_err_mm": "" if np.isnan(ac_err) else f"{ac_err:.6f}",
                        }
                    )
                    if not np.isnan(ac_mm):
                        print(f"[AC] {fn} = {ac_mm:.2f} mm", flush=True)
                n_out += 1

    if n_out == 0:
        print("ERROR: no masks produced.", flush=True)
        raise SystemExit(4)

    if ac_rows:
        ac_out_csv = args.ac_out_csv.strip() or os.path.join(out_dir, "ac_predictions.csv")
        os.makedirs(os.path.dirname(ac_out_csv), exist_ok=True)
        with open(ac_out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["image_name", "mask_path", "pixel_size", "ac_pred_mm", "ac_gt_mm", "ac_abs_err_mm"],
            )
            writer.writeheader()
            writer.writerows(ac_rows)
        print(f"[AC] Saved AC predictions to: {ac_out_csv}", flush=True)


if __name__ == "__main__":
    main()


