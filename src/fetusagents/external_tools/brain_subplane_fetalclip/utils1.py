import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import albumentations as A
import cv2
from torchmetrics import Accuracy, F1Score
from PIL import Image

with open('config.json', 'r') as file:
    config = json.load(file)

DIR_TRAIN = os.path.join(config['paths']['dir_preprocessed'], 'train')
DIR_VAL = os.path.join(config['paths']['dir_preprocessed'], 'val')
DIR_TEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_images')

PATH_TRAIN_VAL_SPLIT = config['paths']['path_train_val_split']
PATH_TEST_SPLIT = config['paths']['path_test_split']
DIR_SAVE_CSV_EXP = config['paths']['dir_experiment_logs']

NUM_WORKERS = config['params']['num_workers']
BATCH_SIZE = config['params']['batch_size']
MAX_EPOCHS = config['params']['max_epochs']

IMG_SIZE = 224
N_RUNS_PER_EXP = 5
CHECK_VAL_EVERY_N_EPOCH = 1
PIN_MEMORY = True

DICT_CLSNAME_TO_CLSINDEX = {
    'Trans-thalamic': 0,
    'Trans-cerebellum': 1,
    'Trans-ventricular': 2,
}
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
with open(PATH_TRAIN_VAL_SPLIT, 'r') as file:
    DICT_LIST_PID = json.load(file)

with open(PATH_TEST_SPLIT, 'r') as file:
    LIST_TEST_PID = json.load(file)

NUM_CLASSES  = len(DICT_CLSNAME_TO_CLSINDEX)

os.makedirs(DIR_SAVE_CSV_EXP, exist_ok=True)

class Dataset4test(torch.utils.data.Dataset):
    def __init__(
            self,
            root,
            preprocess_img,
            dict_embeddings=None,
        ):
        
        self.preprocess_img = preprocess_img
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
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.head = torch.nn.Linear(input_dim, num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.validation_step_outputs = []
    
    def forward(self, x):
        x = self.head(x)
        return x

    def training_step(self, batch, batch_nb):
        _, y, embs= batch

        pred = self.forward(embs)
        
        loss = self.criterion(pred, y)
        pred = torch.argmax(pred, 1)
        acc = Accuracy(task="multiclass", num_classes=self.num_classes, top_k=1, average='macro')(pred.to('cpu'), y.to('cpu')).item()
        f1  = F1Score(task="multiclass", num_classes=self.num_classes, top_k=1, average='macro')(pred.to('cpu'), y.to('cpu')).item()

        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_f1', f1, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_nb):
        _, y, embs, filename= batch

        pred = self.forward(embs)
        
        self.validation_step_outputs.append((pred,filename))
        return pred,filename
    
    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def on_validation_epoch_end(self):
        preds = []


        for outs in self.validation_step_outputs:
            preds.append(outs[0])

        
        preds = torch.cat(preds).cpu()

        
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        preds = []
        filenames = []


        for outs in self.validation_step_outputs:
            preds.append(outs[0])
            filenames.extend(outs[1])
        
        preds = torch.cat(preds).cpu()
        pred_classes = torch.argmax(preds, dim=1).numpy()
        index_to_clsname = {v: k for k, v in DICT_CLSNAME_TO_CLSINDEX.items()}
        pred_labels = [index_to_clsname[int(i)] for i in pred_classes]

        for i in range(len(filenames)):
            print(f"{filenames[i]}  -->  {pred_labels[i]}")

        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.head.parameters(), lr=3e-4, weight_decay=0.01)