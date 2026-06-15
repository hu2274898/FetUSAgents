"""
FetalCLIP GA estimation - Agent-ready version (no interactive input)
Original: predict.py (kept as backup)
Usage: python predict_agent.py --img_dir /path/to/images [--pixel_csv /path/to/pixel_size.csv]
"""
import os
import argparse
import json
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
import open_clip

_SCRIPT_DIR = Path(__file__).resolve().parent
_CKPT_ROOT = Path(os.environ.get("FETALAGENT_CKPT_DIR", str(_SCRIPT_DIR.parents[3] / "FetalAgent_ckpt"))).resolve()
PATH_FETALCLIP_WEIGHT = str(_CKPT_ROOT / "FetalCLIP_weights.pt")
PATH_FETALCLIP_CONFIG = str(_CKPT_ROOT / "FetalCLIP_config.json")
NUM_WORKERS = 0
INPUT_SIZE = 224

DICT_TEMPLATES = {
    "brain": [
        "Ultrasound image at {weeks} weeks and {day} days gestation focusing on the fetal brain, highlighting anatomical structures with a pixel spacing of {pixel_spacing} mm/pixel.",
        "Fetal ultrasound image at {weeks} weeks, {day} days of gestation, focusing on the developing brain, with a pixel spacing of {pixel_spacing} mm/pixel, highlighting the structures of the fetal brain.",
        "Fetal ultrasound image at {weeks} weeks and {day} days gestational age, highlighting the developing brain structures with a pixel spacing of {pixel_spacing} mm/pixel, providing important visual insights for ongoing prenatal assessments.",
        "Ultrasound image at {weeks} weeks and {day} days gestation, highlighting the fetal brain structures with a pixel spacing of {pixel_spacing} mm/pixel.",
        "Fetal ultrasound at {weeks} weeks and {day} days, showing a clear view of the developing brain, with an image pixel spacing of {pixel_spacing} mm/pixel."
    ],
}

list_ga_in_days = [weeks * 7 + days for weeks in range(14, 39) for days in range(0, 7)]


def load_pixel_sizes(csv_path):
    """Load pixel sizes from CSV file."""
    pixel_dict = {}
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            pixel_dict[row['filename']] = float(row['pixel size(mm)'])
    return pixel_dict


def find_median_from_top_n(text_dot_prods, n=20):
    assert len(text_dot_prods.shape) == 1
    tmp = [[i, t] for i, t in enumerate(text_dot_prods)]
    tmp = sorted(tmp, key=lambda x: x[1], reverse=True)[:n]
    tmp = sorted(tmp, key=lambda x: x[0])
    median_ind = tmp[n // 2][0]
    return median_ind


def make_image_square_with_zero_padding(image):
    width, height = image.size
    max_side = max(width, height)

    if image.mode == 'RGBA':
        image = image.convert('RGB')

    if image.mode == "RGB":
        padding_color = (0, 0, 0)
    elif image.mode == "L":
        padding_color = 0
    else:
        padding_color = 0

    new_image = Image.new(image.mode, (max_side, max_side), padding_color)
    padding_left = (max_side - width) // 2
    padding_top = (max_side - height) // 2
    new_image.paste(image, (padding_left, padding_top))
    return new_image


class HCDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, pixel_sizes, preprocess=None):
        self.root_dir = root_dir
        self.preprocess = preprocess
        self.image_files = [
            f for f in os.listdir(root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.data = []
        for filename in self.image_files:
            pixel_size = pixel_sizes.get(filename, 0.15)
            self.data.append({
                "filename": filename,
                "pixel size(mm)": pixel_size
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        name = data['filename']
        imagepath = os.path.join(self.root_dir, data['filename'])

        image = Image.open(imagepath)
        pixel_spacing = max(image.size) / INPUT_SIZE * data['pixel size(mm)']
        image = make_image_square_with_zero_padding(image)

        if self.preprocess:
            image = self.preprocess(image)

        return image, pixel_spacing, name


def get_text_prompts(template, pixel_spacing, tokenizer, model, device):
    prompts = []
    for weeks in range(14, 39):
        for days in range(0, 7):
            prompt = template.replace("{weeks}", str(weeks))
            prompt = prompt.replace("{day}", str(days))
            prompt = prompt.replace("{pixel_spacing}", f"{pixel_spacing:.2f}")
            prompts.append(prompt)
    with torch.no_grad():
        prompts_tokens = tokenizer(prompts).to(device)
        text_features = model.encode_text(prompts_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features


def get_unnormalized_dot_products(image_features, list_text_features):
    text_features = torch.cat(list_text_features, dim=0)
    text_dot_prods = (100.0 * image_features @ text_features.T)

    n_prompts = len(list_text_features)
    n_days = len(list_text_features[0])

    text_dot_prods = text_dot_prods.view(image_features.shape[0], n_prompts, n_days)
    text_dot_prods = text_dot_prods.mean(dim=1)
    return text_dot_prods


def main():
    parser = argparse.ArgumentParser(description='FetalCLIP GA Estimation (Agent-ready)')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--pixel_csv', type=str, default=None,
                        help='Path to pixel_size.csv (default: <img_dir>/pixel_size.csv)')
    parser.add_argument('--model_path', type=str, default=PATH_FETALCLIP_WEIGHT,
                        help='Path to FetalCLIP weights')
    parser.add_argument('--config_path', type=str, default=PATH_FETALCLIP_CONFIG,
                        help='Path to FetalCLIP config')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model configuration
    with open(args.config_path, "r") as file:
        config_fetalclip = json.load(file)
    open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = config_fetalclip

    # Load model
    model, _, preprocess = open_clip.create_model_and_transforms("FetalCLIP", pretrained=args.model_path)
    tokenizer = open_clip.get_tokenizer("FetalCLIP")
    model.eval()
    model.to(device)

    # Load pixel sizes
    pixel_csv = args.pixel_csv if args.pixel_csv else os.path.join(args.img_dir, "pixel_size.csv")
    pixel_sizes = load_pixel_sizes(pixel_csv)

    # Create dataset
    ds = HCDataset(args.img_dir, pixel_sizes, preprocess)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    for imgs, pixel_spacing, name in dl:
        assert imgs.shape[0] == 1
        pixel_spacing = float(pixel_spacing)
        img_name = name[0]
        imgs = imgs.to(device)

        with torch.no_grad():
            image_features = model.encode_image(imgs)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            key = 'brain'
            values = DICT_TEMPLATES[key]
            values = [get_text_prompts(val, pixel_spacing, tokenizer, model, device) for val in values]

            text_dot_prods = get_unnormalized_dot_products(image_features, values)

        text_dot_prods_1d = text_dot_prods.detach().cpu().numpy()[0]
        med_index = find_median_from_top_n(text_dot_prods_1d, n=15)
        pred_days = list_ga_in_days[med_index]
        pred_weeks = pred_days // 7
        pred_remaining_days = pred_days % 7
        pred_weeks_float = pred_days / 7.0
        # Print per-image result immediately
        print(f"[{img_name}] Predicted GA ≈ {pred_weeks} weeks + {pred_remaining_days} days ({pred_weeks_float:.4f} weeks total)", flush=True)


if __name__ == '__main__':
    main()

