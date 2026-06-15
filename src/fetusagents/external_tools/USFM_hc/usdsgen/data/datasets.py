import os

import albumentations as A
import numpy as np
import torch
import torchvision.transforms as T
from albumentations.pytorch import ToTensorV2
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms import str_to_pil_interp
from torch.utils.data import Dataset
from torchvision import datasets


def build_cls_dataset(config, logger):
    train_transforms = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize(
                (config.data.img_size, config.data.img_size),
                interpolation=str_to_pil_interp(config.data.interpolation),
            ),
            # T.RandomHorizontalFlip(),
            # A.RandomRotate90(p=0.5),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(
                mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
                std=torch.tensor(IMAGENET_DEFAULT_STD),
            ),
        ]
    )
    val_transforms = T.Compose(
        [
            T.Resize(
                (config.data.img_size, config.data.img_size),
                interpolation=str_to_pil_interp(config.data.interpolation),
            ),
            T.ToTensor(),
            T.Normalize(
                mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
                std=torch.tensor(IMAGENET_DEFAULT_STD),
            ),
        ]
    )
    if config.data.type == "cls_imagenet":
        data_path = config.data.path
        dataset_train = datasets.ImageFolder(
            os.path.join(data_path.root, data_path.split.train),
            transform=train_transforms,
        )
        dataset_val = datasets.ImageFolder(
            os.path.join(data_path.root, data_path.split.val), transform=val_transforms
        )
        dataset_test = datasets.ImageFolder(
            os.path.join(data_path.root, data_path.split.test), transform=val_transforms
        )
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    logger.info(
        f"Build [Cls] dataset: train images = {len(dataset_train)}, val images = {len(dataset_val)}, test images = {len(dataset_test)}"
    )
    return dataset_train, dataset_val, dataset_test


def build_seg_dataset(config, logger):

    val_transforms = A.Compose(
        [
            A.Resize(width=config.data.img_size, height=config.data.img_size),
            A.ToFloat(max_value=255),
            A.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1
            ),
            ToTensorV2(),
        ]
    )
    Dataset_class = eval(config.data.type + "Dataset")


    dataset_test = Dataset_class(config.data, "test", val_transforms)
    logger.info(
        f"Build [Seg] dataset:  test images = {len(dataset_test)}"
    )

    return dataset_test


class SegBaseDataset(Dataset):
    def __init__(self, DataConfig, stage, transforms=None):
        super().__init__()
        data_folder = os.path.join(DataConfig.path.root, DataConfig.path.split[stage])
        self.num_classes = DataConfig.num_classes
        self.update_datalist(data_folder)
        self.transforms = transforms

    def __getitem__(self, index):
        image_file = self.image_list[index]

        #dummy mask for test
        image = np.array(Image.open(image_file).convert("RGB"))
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        if self.transforms is not None:
            image_mask = self.transforms(image=image, mask=mask)
            image_mask["img_path"] = image_file
            image_mask["mask_path"] = 'dummy'
            image_mask["img_name"] = os.path.basename(image_file) 
        return image_mask

    def update_datalist(self, folder):
        if any(f.endswith('.png') for f in os.listdir(folder)):
            # 直接使用文件夹中的图片文件
            self.image_list = [os.path.join(folder, f) for f in os.listdir(folder) 
                            if f.endswith('.png') and '_mask' not in f]
            self.mask_list = self.image_list.copy()  # 使用相同的伪掩码路径
            print(f"找到 {len(self.image_list)} 个图片文件")


    def __len__(self):
        return len(self.image_list)


class SegVocDataset(Dataset):
    def __init__(self, DataConfig, stage, transforms=None):
        super().__init__()
        self.update_datalist(DataConfig.path.root, stage, DataConfig.path.image_type)
        self.transforms = transforms

    def __getitem__(self, index):
        image_file = self.image_list[index]
        mask_file = self.mask_list[index]
        image = np.array(Image.open(image_file).convert("RGB"))
        mask = np.array(Image.open(mask_file).convert("1")).astype(int)
        if self.transforms is not None:
            image_mask = self.transforms(
                image=image, mask=mask, img_path=image_file, mask_path=mask_file
            )
        return image_mask

    def update_datalist(self, root, stage, image_type):
        filenames = np.loadtxt(
            os.path.join(root, "ImageSets", stage + ".txt"), dtype=str
        )
        image_filenames = [i + "." + image_type for i in filenames]
        mask_filenames = [i + ".png" for i in filenames]
        self.image_list = [os.path.join(root, "JPEGImages", i) for i in image_filenames]
        self.mask_list = [
            os.path.join(root, "SegmentationClass", i) for i in mask_filenames
        ]

    def __len__(self):
        return len(self.image_list)
