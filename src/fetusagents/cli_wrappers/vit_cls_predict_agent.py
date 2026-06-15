import argparse
import os
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models


DICT_CLSNAME_TO_CLSINDEX = {
    "Trans-thalamic": 0,
    "Trans-cerebellum": 1,
    "Trans-ventricular": 2,
}


def make_image_square_with_zero_padding(image: Image.Image) -> Image.Image:
    width, height = image.size
    max_side = max(width, height)
    # Force 3-channel RGB to match ViT input expectation.
    if image.mode != "RGB":
        image = image.convert("RGB")
    padding_color = (0, 0, 0)
    new_image = Image.new(image.mode, (max_side, max_side), padding_color)
    padding_left = (max_side - width) // 2
    padding_top = (max_side - height) // 2
    new_image.paste(image, (padding_left, padding_top))
    return new_image


_preprocessing = A.Compose(
    [
        A.Resize(
            224,
            224,
            interpolation=cv2.INTER_CUBIC,
            mask_interpolation=0,
            p=1.0,
        ),
    ]
)


TEST_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class Dataset4test(Dataset):
    def __init__(self, root: str, preprocess_img):
        self.preprocess_img = preprocess_img
        self.data = []
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
        for filename in sorted(os.listdir(root)):
            if filename.lower().endswith(exts):
                self.data.append((os.path.join(root, filename), filename))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_path, filename = self.data[index]
        data = Image.open(data_path)
        if data is None:
            raise FileNotFoundError(f"image not found: {data_path}")
        img = make_image_square_with_zero_padding(data)
        img = np.array(img)
        img = _preprocessing(image=img)["image"]
        img = Image.fromarray(img)
        img = self.preprocess_img(img)
        ann = img.clone()
        return img, ann, filename


def main() -> None:
    parser = argparse.ArgumentParser(description="Brain subplane classification (ViT) - agent entrypoint")
    parser.add_argument("--data_path", required=True, help="Folder containing images")
    parser.add_argument(
        "--ckpt_path",
        default=str(Path(__file__).resolve().parent.parent / "checkpoints" / "vit_cls_best_model.pth"),
        help="Checkpoint path",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    args = parser.parse_args()

    if not os.path.isdir(args.data_path):
        print(f"ERROR: data_path not found: {args.data_path}", flush=True)
        raise SystemExit(2)

    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")

    test_dataset = Dataset4test(args.data_path, TEST_TRANSFORMS)
    if len(test_dataset) == 0:
        print("ERROR: no images found.", flush=True)
        raise SystemExit(3)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, 3)
    model = model.to(device)

    if not os.path.exists(args.ckpt_path):
        print(f"ERROR: checkpoint not found: {args.ckpt_path}", flush=True)
        raise SystemExit(4)
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()

    index_to_clsname = {v: k for k, v in DICT_CLSNAME_TO_CLSINDEX.items()}
    with torch.no_grad():
        for imgs, _, filenames in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1).detach().cpu()
            preds = outputs.argmax(dim=1).detach().cpu().tolist()
            for fn, pi, prob_vec in zip(filenames, preds, probs):
                label = index_to_clsname.get(int(pi), "UNKNOWN")
                prob_str = " ".join([f"{p:.6f}" for p in prob_vec.tolist()])
                print(f"{fn}: {label} [{prob_str}]", flush=True)


if __name__ == "__main__":
    main()
