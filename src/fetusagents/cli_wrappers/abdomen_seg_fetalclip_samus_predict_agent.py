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
SEG_PROJ_DIR = str(_PROJECT_ROOT / "external_tools" / "fetalclip_seg_ac")
SAMUS_PROJ_DIR = str(_PROJECT_ROOT / "external_tools" / "fetalclip_pred_ac")

DEFAULT_FETALCLIP_CKPT = str(_CKPT_ROOT / "abdomen_fetalclip.ckpt")
DEFAULT_FETALCLIP_CONFIG = str(_PROJECT_ROOT / "external_tools" / "fetalclip_seg_ac" / "FetalCLIP_config.json")
DEFAULT_FETALCLIP_WEIGHTS = str(_CKPT_ROOT / "FetalCLIP_weights.pt")

DEFAULT_SAMUS_CKPT = str(_CKPT_ROOT / "abdomen_samus.pth")
DEFAULT_SAM_BASE = str(_CKPT_ROOT / "SAMUS.pth")


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


def _largest_cc(binary255: np.ndarray) -> np.ndarray:
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
    idx = np.argwhere(mask01 == 1)
    point_label = 1
    if idx.shape[0] == 0:
        point_label = 0
        pt = np.array([[128, 128]], dtype=np.int64)
        return pt, point_label
    idx[:, [0, 1]] = idx[:, [1, 0]]  # (y,x)->(x,y)
    pt = idx[len(idx) // 2][None, :]
    return pt, point_label


def main() -> None:
    sys.dont_write_bytecode = True

    parser = argparse.ArgumentParser(description="Abdomen segmentation (FetalCLIP -> SAMUS) - agent entrypoint")
    parser.add_argument("--data_path", required=True, help="Folder containing images")
    parser.add_argument("--fetalclip_ckpt", default=DEFAULT_FETALCLIP_CKPT, help="FetalCLIP seg checkpoint")
    parser.add_argument("--samus_ckpt", default=DEFAULT_SAMUS_CKPT, help="SAMUS fine-tuned checkpoint")
    parser.add_argument("--sam_base", default=DEFAULT_SAM_BASE, help="Base SAMUS checkpoint (vit_b)")
    parser.add_argument("--out_dir", default="", help="Output directory for masks")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for coarse seg (default: 2)")
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

    # --- Load coarse FetalCLIP segmentation model (224x224) ---
    os.chdir(SEG_PROJ_DIR)
    sys.path.insert(0, SEG_PROJ_DIR)
    from embeddings import EncoderWrapper  # noqa: E402
    from model import SegmentationModel  # noqa: E402

    with open(DEFAULT_FETALCLIP_CONFIG, "r") as f:
        fetalclip_cfg = json.load(f)
    open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = fetalclip_cfg
    clip_model, _, preprocess_img = open_clip.create_model_and_transforms(
        "FetalCLIP", pretrained=DEFAULT_FETALCLIP_WEIGHTS
    )
    visual = clip_model.visual

    tmp = EncoderWrapper(visual)
    input_dim = int(tmp.transformer.width)
    coarse_model = SegmentationModel(
        visual,
        input_dim=input_dim,
        num_classes=1,
        freeze_encoder=False,
        root_path=args.data_path,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_fc", "c_proj", "out_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    coarse_model.encoder = get_peft_model(coarse_model.encoder, lora_config)

    ckpt = torch.load(args.fetalclip_ckpt, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    coarse_model.load_state_dict(state, strict=False)
    coarse_model = coarse_model.to(device).eval()

    # --- Load SAMUS model (256x256) ---
    sys.path.insert(0, SAMUS_PROJ_DIR)
    from models.segment_anything_samus.build_sam_us import samus_model_registry  # noqa: E402

    samus = samus_model_registry["vit_b"](checkpoint=args.sam_base).to(device).eval()
    sam_state = torch.load(args.samus_ckpt, map_location="cpu")
    new_state = (
        {k[7:]: v for k, v in sam_state.items()} if any(k.startswith("module.") for k in sam_state) else sam_state
    )
    samus.load_state_dict(new_state, strict=False)

    out_dir = args.out_dir.strip()
    if not out_dir:
        agent_root = Path(__file__).resolve().parents[1]
        out_dir = str(agent_root / "outputs_agent" / "abdomen_seg" / "fetalclip_samus")
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
            logits = coarse_model(imgs)
            probs = torch.sigmoid(logits)
            coarse_bin = (probs > 0.5).to(torch.uint8).squeeze(1).detach().cpu().numpy() * 255  # (B,H,W)

            for fn, coarse_mask in zip(filenames, coarse_bin):
                clean = _largest_cc(coarse_mask.astype(np.uint8))
                clean256 = cv2.resize(clean, (256, 256), interpolation=cv2.INTER_NEAREST)
                mask01 = (clean256 > 127).astype(np.uint8)
                pt_xy, point_label = _fixed_click(mask01)

                coords = torch.as_tensor(pt_xy, dtype=torch.float32, device=device).unsqueeze(0)  # (1,1,2)
                labels = torch.as_tensor([[point_label]], dtype=torch.int64, device=device)  # (1,1)
                pt = (coords, labels)

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
                sam_mask_u8 = sam_mask.astype(np.uint8)
                Image.fromarray(sam_mask_u8).save(out_path)
                print(f"{fn}: {out_path}", flush=True)
                if fn in ac_info:
                    pixel_size = ac_info[fn].get("pixel_size", float("nan"))
                    ac_px = _compute_ac_px(sam_mask_u8)
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
        print("ERROR: no SAMUS masks produced.", flush=True)
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


