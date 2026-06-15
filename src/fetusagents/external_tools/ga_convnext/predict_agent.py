"""
GA estimation (ConvNeXt regression) - Agent-ready version (no interactive input)
Original: predict.py
Usage: python predict_agent.py --img_dir /path/to/images [--pixel_csv /path/to/pixel_size.csv]
"""
import argparse
import os
from pathlib import Path

import cv2
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

_CKPT_ROOT = Path(os.environ.get("FETALAGENT_CKPT_DIR", str(Path(__file__).resolve().parents[3] / "FetalAgent_ckpt"))).resolve()


def load_pixel_sizes(csv_path: str) -> dict:
    """Load pixel sizes from CSV file."""
    pixel_dict = {}
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            pixel_dict[str(row["filename"]).strip()] = float(row["pixel size(mm)"])
    return pixel_dict


class Dataset4test(Dataset):
    def __init__(self, img_dir: str, pixel_sizes: dict, max_images: int | None = None):
        self.img_dir = img_dir
        self.pixel_sizes = pixel_sizes
        self.target = 0.15
        self.transform_norm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.img_files = [
            f for f in sorted(os.listdir(img_dir)) if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if max_images is not None:
            self.img_files = self.img_files[: max_images]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        filename = self.img_files[index]
        pixel_size = float(self.pixel_sizes.get(filename, 0.15))
        img_path = os.path.join(self.img_dir, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"image not found: {img_path}")

        scale = pixel_size / self.target
        new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

        max_side = max(image.shape)
        pad_vert = (max_side - image.shape[0]) // 2
        pad_horz = (max_side - image.shape[1]) // 2
        image = cv2.copyMakeBorder(
            image,
            pad_vert,
            max_side - image.shape[0] - pad_vert,
            pad_horz,
            max_side - image.shape[1] - pad_horz,
            cv2.BORDER_CONSTANT,
            value=0,
        )

        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = self.transform_norm(image)
        return {"image": image, "filename": filename}


class ConvNeXtRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        # Avoid downloading ImageNet weights; checkpoint will provide trained weights
        self.model = models.convnext_tiny(weights=None)
        dim = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Sequential(nn.Linear(dim, 1))

    def forward(self, x):
        return self.model(x).squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="GA Estimation (ConvNeXt regression) - agent entrypoint")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument(
        "--pixel_csv",
        type=str,
        default=None,
        help="Path to pixel_size.csv (default: <img_dir>/pixel_size.csv)",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=str(_CKPT_ROOT / "best.pth"),
        help="Checkpoint path",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID (default: 0)")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size (default: 20)")
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Limit number of images to process (for quick integration testing)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.img_dir):
        print(f"ERROR: img_dir not found: {args.img_dir}", flush=True)
        raise SystemExit(2)

    pixel_csv = args.pixel_csv if args.pixel_csv else os.path.join(args.img_dir, "pixel_size.csv")
    if not os.path.exists(pixel_csv):
        print(f"ERROR: pixel_size.csv not found: {pixel_csv}", flush=True)
        raise SystemExit(3)
    pixel_sizes = load_pixel_sizes(pixel_csv)
    print(f"[INFO] Loaded {len(pixel_sizes)} pixel sizes from {pixel_csv}", flush=True)

    dataset = Dataset4test(args.img_dir, pixel_sizes, max_images=args.max_images)
    if len(dataset) == 0:
        print("ERROR: no images found.", flush=True)
        raise SystemExit(4)
    print(f"[INFO] Found {len(dataset)} images in {args.img_dir}", flush=True)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
    print(f"[INFO] Using device: {device}", flush=True)

    model = ConvNeXtRegressor()
    if not os.path.exists(args.ckpt_path):
        print(f"ERROR: checkpoint not found: {args.ckpt_path}", flush=True)
        raise SystemExit(5)
    print(f"[INFO] Loading checkpoint: {args.ckpt_path}", flush=True)
    state = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device).eval()
    print("[INFO] Model ready. Starting inference...", flush=True)

    with torch.no_grad():
        for batch in loader:
            data = batch["image"].to(device, dtype=torch.float)
            filenames = batch["filename"]
            outputs = model(data).detach().cpu().numpy()
            for f, pred in zip(filenames, outputs):
                try:
                    pred_val = float(pred)
                except Exception:
                    pred_val = float(pred[0])
                print(f"{f}: {pred_val:.4f}", flush=True)


if __name__ == "__main__":
    main()