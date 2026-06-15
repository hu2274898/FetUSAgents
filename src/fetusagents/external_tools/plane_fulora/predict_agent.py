"""
FU-LoRA plane classification - Agent-ready version (no interactive input)
Original: predict.py (kept as backup)
Usage: python predict_agent.py --data_path /path/to/images
"""
import argparse
import torch
import os
from pathlib import Path
import PIL.Image
import torch.utils.data
import cv2
import numpy as np
import torchvision.transforms as transforms

_CKPT_ROOT = Path(os.environ.get("FETALAGENT_CKPT_DIR", str(Path(__file__).resolve().parents[3] / "FetalAgent_ckpt"))).resolve()

# Class mappings
class2idx = {
    "Other": 0,
    "Fetal abdomen": 1,
    "Fetal brain": 2,
    "Fetal femur": 3,
    "Fetal thorax": 4
}
idx2class = {v: k for k, v in class2idx.items()}

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
])


def load_model(model_name, device):
    model = torch.load(model_name, map_location=device)
    model.eval()
    return model


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        super(Dataset, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.paths = []
        for f in os.listdir(root_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.paths.append(os.path.join(self.root_dir, f))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path = self.paths[index]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = PIL.Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        image = torch.from_numpy(image.numpy()[..., ::])
        img_name = os.path.basename(img_path)
        return image, img_name


def main():
    parser = argparse.ArgumentParser(description='FU-LoRA Plane Classification (Agent-ready)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to directory containing images')
    parser.add_argument('--model_path', type=str, 
                        default=str(_CKPT_ROOT / 'spain_synthesis_6148_5000_vit_07102025.pth'),
                        help='Path to model checkpoint')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    args = parser.parse_args()

    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    # Load model
    model = load_model(args.model_path, device)
    model = model.to(device)
    model.eval()

    # Create dataset and dataloader
    test_dataset = Dataset(root_dir=args.data_path, transform=VAL_TRANSFORMS)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    for image, name in test_loader:
        img_name = name[0] if isinstance(name, (list, tuple)) else str(name)
        image = image.to(device)
        output = model(image)
        probs = torch.softmax(output, dim=1)  # Use softmax for probabilities
        _, pred = torch.max(probs, dim=1)
        pred_idx = int(pred.item())
        pred_label = idx2class.get(pred_idx, "Other")
        probs_np = probs.cpu().detach().numpy()[0]
        # Build prob string: Other:0.01,Fetal abdomen:0.90,...
        prob_str = ",".join([f"{idx2class[j]}:{probs_np[j]:.6f}" for j in range(len(idx2class))])
        # Print per-image result with probabilities
        print(f"{img_name}: {pred_label} | probs: {prob_str}", flush=True)


if __name__ == '__main__':
    main()

