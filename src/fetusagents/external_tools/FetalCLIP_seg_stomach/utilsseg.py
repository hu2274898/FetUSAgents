import os
import json

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms as T
import albumentations as A
import open_clip
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils

from PIL import Image
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent

with open('config.json', 'r') as file:
    config = json.load(file)

DIR_TRAIN = os.path.join(config['paths']['dir_preprocessed'], 'train')
DIR_VAL = os.path.join(config['paths']['dir_preprocessed'], 'val')
DIR_TEST = str(_SCRIPT_DIR / 'img')


DIR_SAVE_CSV_EXP = config['paths']['dir_experiment_logs']

NUM_WORKERS = config['params']['num_workers']
BATCH_SIZE = config['params']['batch_size']
MAX_EPOCHS = config['params']['max_epochs']

IMG_SIZE = 224
INIT_FILTERS = 32
N_RUNS_PER_EXP = 1
CHECK_VAL_EVERY_N_EPOCH = 1
PIN_MEMORY = True
def make_image_square_with_zero_padding(image):
    width, height = image.size

    # Determine the size of the square
    max_side = max(width, height)

    # Create a new square image with black padding (0 for black in RGB or L modes)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    if image.mode == "RGB":
        padding_color = (0, 0, 0)  # Black for RGB images
    elif image.mode == "L":
        padding_color = 0  # Black for grayscale images

    # Create a new square image
    new_image = Image.new(image.mode, (max_side, max_side), padding_color)

    # Calculate padding
    padding_left = (max_side - width) // 2
    padding_top = (max_side - height) // 2

    # Paste the original image in the center of the new square image
    new_image.paste(image, (padding_left, padding_top))

    return new_image
preprocessing = A.Compose([
    A.Resize(
        224, 224, interpolation=cv2.INTER_CUBIC,
        mask_interpolation=0,
        p=1.0
    ),
])

os.makedirs(DIR_SAVE_CSV_EXP, exist_ok=True)

class DatasetHC18(torch.utils.data.Dataset):
    def __init__(
            self,
            root,
            preprocess_img,
            preprocess_mask,
            dict_embeddings=None,
        ):
        
        self.preprocess_img = preprocess_img
        self.preprocess_mask = preprocess_mask
        self.dict_embeddings = dict_embeddings


        self.data = []
        for filename in os.listdir(root):

            if self.dict_embeddings:
                self.data.append((
                    os.path.join(root, filename),
                    dict_embeddings[filename.split('.')[0]],
                    filename
                ))
            else:
                self.data.append((os.path.join(root, filename),filename))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if self.dict_embeddings:
            data_path, embs, filename = self.data[index]
            data = Image.open(data_path)
        else:
            data_path, filename = self.data[index]
            data = Image.open(data_path)
        if data is None:
            raise FileNotFoundError(f'image not found:{data_path}')

        img = make_image_square_with_zero_padding(data)
        img = np.array(img)
        img_ann = preprocessing(image=img)
        img = img_ann["image"]
        
        img = Image.fromarray(img)
        img = self.preprocess_img(img)
        ann = img.clone()

        
        if self.dict_embeddings:
            return img, ann, embs,filename
        else:
            return img, ann,filename

class LitModel(pl.LightningModule):
    def __init__(self, transformer_width, num_classes, input_dim, init_filters):
        super().__init__()
        self.num_classes = num_classes
        self.model = UNETR(transformer_width, num_classes, input_dim, init_filters)
        self.criterion = smp.losses.DiceLoss(mode='multilabel', from_logits=True)
        self.validation_step_outputs = []
    
    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y, embs = batch

        pred = self.forward([x, *embs])
        
        loss = self.criterion(pred, y)
        
        dsc = smp_utils.metrics.Fscore(activation='sigmoid')(pred, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_dsc', dsc, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_nb):
        x, y, embs, filename= batch
        
        pred = self.forward([x, *embs])
        
        self.validation_step_outputs.append((pred, filename))
        return pred, filename
    
    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def on_validation_epoch_end(self):
        preds = []
        targets = []


        for outs in self.validation_step_outputs:
            preds.append(outs[0])
            targets.append(outs[1])
      
        
        preds = torch.cat(preds).cpu()
        targets = torch.cat(targets).cpu()

        loss = self.criterion(preds, targets)
        
        dsc = smp_utils.metrics.Fscore(activation='sigmoid')(preds, targets)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_dsc', dsc, on_step=False, on_epoch=True, prog_bar=True)
        
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        preds = []
        filenames = []

        for outs in self.validation_step_outputs:
            preds.append(outs[0])
            filenames.extend(outs[1])
        
        preds = torch.cat(preds).cpu()

        # ===================== 保存预测结果 =====================
        save_dir = str(_SCRIPT_DIR / 'prediction')
        os.makedirs(save_dir, exist_ok=True)
        for i in range(preds.shape[0]):
            pred = torch.sigmoid(preds[i]).squeeze().numpy()
            pred_binary = (pred > 0.5).astype(np.uint8)
            pred_binary = pred_binary * 255
            base_name = os.path.splitext(filenames[i])[0]
            Image.fromarray(pred_binary).save(os.path.join(save_dir, f"{base_name}_prediction.png"))

        print(f"Saved predictions to: {save_dir}")
        # ===========================================================

        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        trainable_params = (
            param for name, param in self.named_parameters()
            if not name.startswith('model.transformer')
        )
        for name, param in self.named_parameters():
            if name.startswith('model.transformer'):
                param.requires_grad = False
        return torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)

'''
REFERENCES:
- https://github.com/tamasino52/UNETR/blob/main/unetr.py#L171
'''

class SingleDeconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, groups=1):
        super().__init__()
        self.block = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0, groups=groups)

    def forward(self, x):
        return self.block(x)


class SingleConv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1):
        super().__init__()
        self.block = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                               padding=((kernel_size - 1) // 2), groups=groups)

    def forward(self, x):
        return self.block(x)


class Conv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv2DBlock(in_planes, in_planes, kernel_size, groups=in_planes),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(True),
            SingleConv2DBlock(in_planes, out_planes, 1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)


class Deconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv2DBlock(in_planes, in_planes, groups=in_planes),
            SingleConv2DBlock(in_planes, in_planes, kernel_size, groups=in_planes),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(True),
            SingleConv2DBlock(in_planes, out_planes, 1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)

class SingleDWConv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv2DBlock(in_planes, in_planes, groups=in_planes),
            SingleConv2DBlock(in_planes, out_planes, 1),
        )

    def forward(self, x):
        return self.block(x)

class UNETR(nn.Module):
    def __init__(self, transformer_width, output_dim, input_dim, init_filters):
        super().__init__()

        self.decoder0 = \
            nn.Sequential(
                Conv2DBlock(input_dim, init_filters, 3),
                Conv2DBlock(init_filters, init_filters, 3)
            )

        self.decoder3 = \
            nn.Sequential(
                Deconv2DBlock(transformer_width, 8*init_filters),
                Deconv2DBlock(8*init_filters, 4*init_filters),
                Deconv2DBlock(4*init_filters, 2*init_filters)
            )

        self.decoder6 = \
            nn.Sequential(
                Deconv2DBlock(transformer_width, 8*init_filters),
                Deconv2DBlock(8*init_filters, 4*init_filters),
            )

        self.decoder9 = \
            Deconv2DBlock(transformer_width, 8*init_filters)

        self.decoder12_upsampler = \
            SingleDWConv2DBlock(transformer_width, 8*init_filters)

        self.decoder9_upsampler = \
            nn.Sequential(
                Conv2DBlock(16*init_filters, 8*init_filters),
                Conv2DBlock(8*init_filters, 8*init_filters),
                Conv2DBlock(8*init_filters, 8*init_filters),
                SingleDWConv2DBlock(8*init_filters, 4*init_filters)
            )

        self.decoder6_upsampler = \
            nn.Sequential(
                Conv2DBlock(8*init_filters, 4*init_filters),
                Conv2DBlock(4*init_filters, 4*init_filters),
                SingleDWConv2DBlock(4*init_filters, 2*init_filters)
            )

        self.decoder3_upsampler = \
            nn.Sequential(
                Conv2DBlock(4*init_filters, 2*init_filters),
                Conv2DBlock(2*init_filters, 2*init_filters),
                SingleDWConv2DBlock(2*init_filters, init_filters)
            )

        self.decoder0_header = \
            nn.Sequential(
                Conv2DBlock(2*init_filters, init_filters),
                Conv2DBlock(init_filters, init_filters),
                SingleConv2DBlock(init_filters, output_dim, 1)
            )
    
    def forward(self, x):
        z0, z3, z6, z9, z12 = x
        
        # print(z0.shape, z3.shape, z6.shape, z9.shape, z12.shape)
        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(z0)
        # print(z0.shape, z3.shape, z6.shape, z9.shape, z12.shape)
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))

        return output