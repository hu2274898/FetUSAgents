"""
GA estimation (RadImageNet-based) - Agent-ready version (no interactive input)
Original: predict.py / predict_without_pixel_size.py (kept as backup)
Usage: python predict_agent.py --img_dir /path/to/images [--pixel_csv /path/to/pixel_size.csv]
"""
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import os
import cv2
from pathlib import Path
from torchvision import transforms
from torchvision.models import resnet50
import torch.nn as nn

_CKPT_ROOT = Path(os.environ.get("FETALAGENT_CKPT_DIR", str(Path(__file__).resolve().parents[3] / "FetalAgent_ckpt"))).resolve()


def load_pixel_sizes(csv_path):
    """Load pixel sizes from CSV file."""
    pixel_dict = {}
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            pixel_dict[row['filename']] = float(row['pixel size(mm)'])
    return pixel_dict


class GADataset(Dataset):
    def __init__(self, img_dir, pixel_sizes=None, transform=None):
        self.img_dir = img_dir
        self.img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.target = 0.15  # target pixel size for normalization
        self.transform = transform if transform else transforms.Compose([transforms.ToTensor()])
        self.pixel_sizes = pixel_sizes if pixel_sizes else {}

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        filename = self.img_files[index]
        pixel_size = self.pixel_sizes.get(filename, 0.15)  # default 0.15 mm
        img_path = os.path.join(self.img_dir, filename)
        
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f'Image not found: {img_path}')

        # Scale based on pixel size
        scale = pixel_size / self.target
        new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

        # Pad to square
        max_side = max(image.shape)
        pad_vert = (max_side - image.shape[0]) // 2
        pad_horz = (max_side - image.shape[1]) // 2
        image = cv2.copyMakeBorder(
            image, pad_vert, max_side - image.shape[0] - pad_vert,
            pad_horz, max_side - image.shape[1] - pad_horz,
            cv2.BORDER_CONSTANT, value=0
        )

        # Resize to 224x224
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = (image - 127.5) * 2 / 255

        if self.transform is not None:
            image = self.transform(image)

        return {"image": image, "filename": filename}


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = resnet50(pretrained=False)
        encoder_layers = list(base_model.children())
        self.backbone = nn.Sequential(*encoder_layers[:-1])

    def forward(self, x):
        return self.backbone(x)


class Classifier(nn.Module):
    def __init__(self, num_class, in_features=2048, hidden=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden, num_class)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x).squeeze(-1)


def main():
    parser = argparse.ArgumentParser(description='GA Estimation Algorithm 1 (Agent-ready)')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--pixel_csv', type=str, default=None, 
                        help='Path to pixel_size.csv (default: <img_dir>/pixel_size.csv)')
    parser.add_argument('--model_path', type=str,
                        default=str(_CKPT_ROOT / 'acl_fold1_best_model_day_epoch226.pth'),
                        help='Path to model checkpoint')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    args = parser.parse_args()

    # Setup device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load pixel sizes
    pixel_csv = args.pixel_csv if args.pixel_csv else os.path.join(args.img_dir, "pixel_size.csv")
    pixel_sizes = load_pixel_sizes(pixel_csv)

    # Create dataset
    test_dataset = GADataset(img_dir=args.img_dir, pixel_sizes=pixel_sizes, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=0)

    # Build model
    backbone = Backbone()
    classifier = Classifier(num_class=1)
    model = nn.Sequential(backbone, classifier)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Run inference
    with torch.no_grad():
        for batch in test_loader:
            data = batch["image"].to(device, dtype=torch.float)
            filenames = batch["filename"]
            outputs = model(data).cpu().numpy()
            for f, pred in zip(filenames, outputs):
                # Print per-image result immediately so users see progress
                try:
                    pred_val = float(pred)
                except Exception:
                    pred_val = float(pred[0])
                print(f"{f}: {pred_val:.4f}", flush=True)


if __name__ == '__main__':
    main()

