import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import lightning as L
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random
from utils import init_logger

logger = logging.getLogger(__name__)
init_logger()
csv_path = Path(__file__).resolve().parent / 'data' / 'us_gt_updated.csv'
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

class AcouslicAIDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        num_workers: int,
        batch_size: int = 32,
        use_augmentation: bool = False,
        few_shot_list: Optional[list] = None,
        image_transform=None,
        mask_transform=None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.use_augmentation = use_augmentation

        self.few_shot_list = few_shot_list

        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # please either download the raw dataset from https://zenodo.org/records/12697994
        # or use the preprocessed dataset using the link on our github repository
        ...

    def setup(self,stage: Optional[str] = None):
        self.test_dataset = AcoudslicAIDataset(
            data_dir=self.data_dir,
            image_transform=self.image_transform,
            mask_transform=self.mask_transform,
            few_shot_list=self.few_shot_list,
            use_augmentation=self.use_augmentation,
        ) 



    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class AcoudslicAIDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        image_transform=None,
        mask_transform=None,
        few_shot_list: Optional[list] = None,
        use_augmentation: bool = True,
    ):
        self.data_dir = data_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        if csv_path.exists():
            self.df = pd.read_csv(csv_path)
            self.name_to_value = dict(zip(self.df["file_name"], self.df["value"]))
        else:
            # In packaged inference-only use, the training metadata CSV is not shipped.
            self.df = None
            self.name_to_value = {}

        self.data = []
        for data_path in list(sorted(self.data_dir.iterdir())):
            if not data_path.is_file() or data_path.suffix.lower() not in _IMAGE_EXTS:
                continue
            # using augmented data or not
            if not use_augmentation and len(data_path.stem.split("_")) >= 3:
                continue

            # few-shot learning is not implemented but is possible
            if few_shot_list is not None and data_path.stem not in few_shot_list:
                continue

            self.data.append(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_path = self.data[idx]
        image = Image.open(data_path)
        stem = data_path.stem
        if "_aug" in stem:
            original_stem = stem.rsplit("_aug", 1)[0]  # 分割最后一个 _aug，取前面部分
        else:
            original_stem = stem
        label_lookup_name = original_stem + ".jpeg"
        label = self.name_to_value.get(data_path.name, self.name_to_value.get(label_lookup_name, -1))
        if self.image_transform:
            image = image.convert("RGB")
            image = self.image_transform(image)
        mask = 1
        item = {"image": image, "mask": mask, "label": label, 'file_name': data_path.name}
        return item




class AcoudslicAIDatasettrain(Dataset):
    def __init__(
        self,
        data_dir: Path,
        image_transform=None,
        mask_transform=None,
        few_shot_list: Optional[list] = None,
        use_augmentation: bool = True,
        zero_ratio: float = 0.3,  # 新增：控制保留多少比例的 0 样本
    ):
        self.data_dir = data_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        if csv_path.exists():
            self.df = pd.read_csv(csv_path)
            self.name_to_value = dict(zip(self.df["file_name"], self.df["value"]))
        else:
            self.df = None
            self.name_to_value = {}

        # --- Step 1. 收集所有数据路径 ---
        all_data = []
        for data_path in sorted(self.data_dir.iterdir()):
            if not data_path.is_file() or data_path.suffix.lower() not in _IMAGE_EXTS:
                continue
            if not use_augmentation and len(data_path.stem.split("_")) >= 3:
                continue
            if few_shot_list is not None and data_path.stem not in few_shot_list:
                continue
            all_data.append(data_path)

        # --- Step 2. 区分 label=5 ---
        data_else,data_5 = [], []
        for p in all_data:
            stem = p.stem
            if "_aug" in stem:
                stem = stem.rsplit("_aug", 1)[0]
            file_name = stem + ".jpeg"
            label = self.name_to_value.get(p.name, self.name_to_value.get(file_name, -1))
            if label == 5:
                data_5.append(p)
            else:
                data_else.append(p)
        zero_ratio = 0.015
        # --- Step 3. 随机抽取部分 5 样本，全保留 else 样本 ---
        n_keep_5 = int(len(data_5) * zero_ratio)
        random.shuffle(data_5)
        selected_data_0 = data_5[:n_keep_5]
        self.data = selected_data_0 + data_else
        random.shuffle(self.data)

        print(f"数据集加载完成：保留 {len(data_else)} 个 label=0,1,2,3,4，全量；{n_keep_5}/{len(data_5)} 个 label=5。")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_path = self.data[idx]
        image = Image.open(data_path)
        stem = data_path.stem
        if "_aug" in stem:
            original_stem = stem.rsplit("_aug", 1)[0]
        else:
            original_stem = stem

        file_name = original_stem + ".jpeg"
        label = self.name_to_value.get(data_path.name, self.name_to_value.get(file_name, -1))

        if self.image_transform:
            image = image.convert("RGB")
            image = self.image_transform(image)

        mask = 1
        return {"image": image, "mask": mask, "label": label, "file_name": data_path.name}
