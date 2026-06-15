"""
FetalCLIP plane classification - Agent-ready version (no interactive input)
Original: predict.py (kept as backup)
Usage: python predict_agent.py --data_path /path/to/images
"""
import os
import json
import argparse
import torch
import numpy as np
import open_clip
from pathlib import Path
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
_CKPT_ROOT = Path(os.environ.get("FETALAGENT_CKPT_DIR", str(_SCRIPT_DIR.parents[3] / "FetalAgent_ckpt"))).resolve()
PATH_FETALCLIP_WEIGHT = str(_CKPT_ROOT / "FetalCLIP_weights.pt")
PATH_FETALCLIP_CONFIG = str(_CKPT_ROOT / "FetalCLIP_config.json")
PATH_TEXT_PROMPTS = "test_five_planes_prompts.json"
BATCH_SIZE = 1
NUM_WORKERS = 4


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


class DatasetFetalPlanesDB(torch.utils.data.Dataset):
    def __init__(self, dir_images, preprocess):
        self.root = dir_images
        self.preprocess = preprocess
        self.data = [
            os.path.join(self.root, f)
            for f in os.listdir(self.root)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = Image.open(self.data[index])
        img = make_image_square_with_zero_padding(img)
        img = self.preprocess(img)
        filename = os.path.basename(self.data[index])
        return img, filename


def main():
    parser = argparse.ArgumentParser(description='FetalCLIP Plane Classification (Agent-ready)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to directory containing images')
    parser.add_argument('--model_path', type=str, default=PATH_FETALCLIP_WEIGHT, help='Path to FetalCLIP weights')
    parser.add_argument('--config_path', type=str, default=PATH_FETALCLIP_CONFIG, help='Path to FetalCLIP config')
    parser.add_argument('--prompts_path', type=str, default=PATH_TEXT_PROMPTS, help='Path to text prompts JSON')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model configuration
    config_path = args.config_path
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    
    with open(config_path, "r") as file:
        config_fetalclip = json.load(file)
    open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = config_fetalclip

    # Load text prompts
    prompts_path = args.prompts_path
    if not os.path.isabs(prompts_path):
        prompts_path = os.path.join(os.path.dirname(__file__), prompts_path)
    
    with open(prompts_path, 'r') as json_file:
        text_prompts = json.load(json_file)

    planename_to_index = {key: i for i, key in enumerate(text_prompts.keys())}
    index_to_planename = {val: key for key, val in planename_to_index.items()}

    # Load model
    model, _, preprocess = open_clip.create_model_and_transforms("FetalCLIP", pretrained=args.model_path)
    tokenizer = open_clip.get_tokenizer("FetalCLIP")
    model.eval()
    model.to(device)

    # Create dataset
    ds = DatasetFetalPlanesDB(args.data_path, preprocess)
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Encode text features
    list_text_features = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for plane, prompts in text_prompts.items():
            prompts_tokens = tokenizer(prompts).to(device)
            text_features = model.encode_text(prompts_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.mean(dim=0).unsqueeze(0)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            list_text_features.append(text_features)

    # Get class names in order
    class_names = list(text_prompts.keys())
    
    # Run inference
    for imgs, filename in dl:
        with torch.no_grad(), torch.cuda.amp.autocast():
            imgs = imgs.to(device)
            image_features = model.encode_image(imgs)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            list_text_logits = []
            for text_features in list_text_features:
                text_logits = (100.0 * image_features @ text_features.T).mean(dim=-1)[:, None]
                list_text_logits.append(text_logits)
            text_probs = torch.cat(list_text_logits, dim=1).softmax(dim=-1)
            pred_indices = torch.argmax(text_probs, dim=1).cpu().tolist()
            probs_np = text_probs.cpu().numpy()

            for i, (name, idx) in enumerate(zip(filename, pred_indices)):
                # Print per-image result with probabilities
                prob_dict = {class_names[j]: float(probs_np[i, j]) for j in range(len(class_names))}
                prob_str = ",".join([f"{k}:{v:.6f}" for k, v in prob_dict.items()])
                print(f"{name} → Predicted plane: {index_to_planename[idx]} | probs: {prob_str}", flush=True)


if __name__ == '__main__':
    main()

