import os
from pathlib import Path
from typing import Union
import sys

import logging
import json
from datetime import datetime

import lightning as L
import open_clip
import pandas as pd
import timm
import torch
import torchvision.transforms as T
# import wandb
from jsonargparse import auto_cli
from lightning.pytorch.callbacks import ModelCheckpoint, Timer, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from peft import get_peft_model, LoraConfig
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from data import AcouslicAIDataModule
from model import ClassificationModel, SegmentationModel
from utils import Settings, init_logger

_CKPT_ROOT = Path(
    os.environ.get(
        "FETALAGENT_CKPT_DIR",
        str(Path(__file__).resolve().parents[3] / "FetalAgent_ckpt"),
    )
).resolve()


logger = logging.getLogger(__name__)
init_logger()


def get_encoder_and_transforms(
    model_name: str, config: dict[str, Union[str, Path]]
) -> tuple:
    # Load model configuration
    if model_name == "fetalclip":
        fetalclip_config_path = Path(config.fetalclip_config_path)
        fetalclip_weights_path = Path(config.fetalclip_weights_path)
        if not fetalclip_config_path.is_absolute():
            fetalclip_config_path = _CKPT_ROOT / fetalclip_config_path.name
        if not fetalclip_weights_path.is_absolute():
            fetalclip_weights_path = _CKPT_ROOT / fetalclip_weights_path.name

        with open(fetalclip_config_path, "r") as file:
            fetalclip_config = json.load(file)
        open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = fetalclip_config

        model, _, image_transform = open_clip.create_model_and_transforms(
            "FetalCLIP", pretrained=str(fetalclip_weights_path)
        )
        model = model.visual

        image_size = fetalclip_config["vision_cfg"]["image_size"]
    elif model_name == "densenet":
        model_name = "densenet121"
    elif model_name == "resnet":
        model_name = "resnet50"
    elif model_name == "mobilenet":
        model_name = "mobilenetv3_small_100"
    elif model_name == "efficientnet":
        model_name = "efficientnet_b0"
    elif model_name == "vgg":
        model_name = "vgg16"
    elif model_name == "vit":
        model_name = "vit_large_patch14_clip_224.openai_ft_in12k_in1k"
    elif model_name == "vit_laion":
        model_name = "vit_large_patch14_clip_224.laion2b_ft_in12k_in1k"
    elif model_name == "vit_small":
        model_name = "vit_small_patch16_224.augreg_in21k_ft_in1k"
    elif model_name == "deit":
        model_name = "deit3_large_patch16_224.fb_in22k_ft_in1k"
    elif model_name == "cait":
        model_name = "cait_s24_224.fb_dist_in1k"
    elif model_name == "medvit":
        # MedViT [1/2]: to test MedViT, you need to manually install medvit library and modify the relative path as needed
        import sys

        sys.path.append(str(Path("../MedViT").resolve().parent))
        #from MedViT.MedViT import MedViT
    elif model_name == "swin":
        model_name = "swin_large_patch4_window7_224.ms_in22k_ft_in1k"
    else:
        raise ValueError(f"Unsupported model name: {config['model_name']}")

    if model_name == "medvit":
        model = MedViT(stem_chs=[64, 32, 64], depths=[3, 4, 30, 3], path_dropout=0.2)
        image_size = 224
        image_transform = T.Compose(
            [
                T.Resize(image_size),
                T.Lambda(lambda image: image.convert("RGB")),
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
    elif model_name != "fetalclip":
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        model_config = resolve_data_config({}, model=model)
        image_transform = create_transform(**model_config)
        image_size = model.default_cfg["input_size"][-1]

    mask_transform = None
    if config["task"] == "segmentation":
        mask_transform = T.Compose(
            [
                T.Resize(
                    (image_size, image_size), interpolation=T.InterpolationMode.NEAREST
                ),
                T.ToTensor(),
            ]
        )

    return model, image_transform, mask_transform


def main(config: dict[str, Union[str, Path]]):
    task = config.task
    model_name = config.model_name.lower()
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # output paths
    exp_output_folder = config["output_dir"] / task / model_name / run_id
    exp_results_path = exp_output_folder / "results.csv"

    # print meta info
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Config: {json.dumps(config._to_dict(), indent=2)}")
    logger.info(f"Output folder: {exp_output_folder}")

    encoder, image_transform, mask_transform = get_encoder_and_transforms(
        model_name, config
    )

    if task == "classification":
        if model_name == "fetalclip":
            input_dim = encoder.proj.shape[1]
        elif model_name in ["resnet", "densenet", "efficientnet", "swin"]:
            input_dim = encoder.num_features
        elif model_name in ["mobilenet", "vgg"]:
            input_dim = encoder.head_hidden_size
        elif model_name in ["vit", "deit", "cait", "vit_tiny"]:
            input_dim = encoder.embed_dim
        elif model_name == "medvit":
            input_dim = encoder.proj_head[0].weight.shape[1]
            # MedViT [2/2]g: if use medvit, you need to resolve the relative path
            encoder.load_state_dict(
                torch.load("../MedViT/MedViT_large_im1k.pth")["model"], strict=False
            )
            encoder.proj_head = torch.nn.Identity()
    elif task == "segmentation":
        if model_name == "fetalclip":
            input_dim = encoder.transformer.width
        elif model_name == "vit":
            input_dim = encoder.embed_dim
    root_path = os.environ.get("FETALAGENT_KEYFRAME_DATA_PATH", "").strip()
    if not root_path:
        root_path = input("Please enter the data path: ").strip()
    if not root_path:
        raise ValueError("No data path provided for key-frame classification.")
    root_path = str(Path(root_path).resolve())
    # dataset preparation
    data_module = AcouslicAIDataModule(
        data_dir=root_path,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        use_augmentation=config.use_augmentation,
        image_transform=image_transform,
        mask_transform=mask_transform,
    )
    data_module.setup()

    test_dataloader = data_module.test_dataloader()

    # training
    use_lora = config.use_lora
    num_trials = config.num_trials
    num_classes = 6  # fixed - binary classification

    max_epochs = config.max_epochs

    torch.set_float32_matmul_precision("high")

    # debug mode
    limit_batches = None
    if config.debug_mode:
        limit_batches = 5
        num_trials = 2
        max_epochs = 1
        logger.info("Debug mode is ON. Limiting batches, epochs, and num of trials.")

    for trial in range(num_trials):
        logger.info(f"Starting trial {trial + 1}/{num_trials}")

        # network setup
        if task == "classification":
            model = ClassificationModel(
                encoder, input_dim, num_classes, freeze_encoder=config.freeze_encoder
            )
        elif task == "segmentation":
            model = SegmentationModel(
                encoder, input_dim, num_classes, freeze_encoder=config.freeze_encoder
            )

        if use_lora:
            if model_name == "fetalclip":
                target_modules = ["c_fc", "c_proj", "out_proj"]
            else:
                target_modules = ["proj", "fc1", "fc2"]

            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=target_modules,
                lora_dropout=0.1,
                bias="none",
            )
            model.encoder = get_peft_model(model.encoder, lora_config)
            model.encoder.print_trainable_parameters()

        # callbacks
        checkpoint = ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            dirpath=exp_output_folder / "checkpoints" / f"trial-{str(trial)}",
        )
        # early stopping is not used but kept for completeness
        earlystop = EarlyStopping(monitor="val/loss", patience=5, mode="min")
        timer = Timer()
        callbacks = [checkpoint, earlystop, timer]

        # wandb logger
        # wandb_logger = WandbLogger(
        #     project="FetalCLIP-IQA",
        #     group=f"{task}/{model_name}",
        #     name=f"{run_id}-trial_{trial}",
        #     save_dir=exp_output_folder,
        #     tags=[f"trial_{trial}", task, model_name, run_id],
        #     settings=wandb.Settings(reinit="finish_previous"),
        # )

        trainer = L.Trainer(
            devices=[int(os.environ.get("KEYFRAME_TRAINER_DEVICE", "0"))],
            accelerator="gpu",
            max_epochs=max_epochs,
            logger=False,  #wandb_logger
            callbacks=callbacks,
            precision="bf16-mixed",
            # only for debug purpose
            limit_train_batches=limit_batches,
            limit_val_batches=limit_batches,
            limit_test_batches=limit_batches,
        )

        # trainer.fit(model, datamodule=data_module)
        # model = ClassificationModel.load_from_checkpoint(
        #     "./checkpoints/epoch=0-step=150.ckpt",
        #     encoder=encoder,
        #     input_dim=input_dim,
        #     num_classes=num_classes,
        #     freeze_encoder=config.freeze_encoder,
        # )
        results = trainer.test(
            model,
            dataloaders=test_dataloader,
            ckpt_path=str(_CKPT_ROOT / "keyframe_cls6" / "epoch=14-step=540.ckpt"),
        )

        logger.info(f"Train time: {round(timer.time_elapsed('train'), 2)} s")
        logger.info(
            f"Inference time: {round(timer.time_elapsed('test') * 1000 / len(test_dataloader), 2)} ms"
        )

        # Save results to CSV
        results = {"trial": trial, **results[0]}
        results = pd.DataFrame([results])
        if exp_results_path.exists():
            results.to_csv(exp_results_path, mode="a", header=False, index=False)
        else:
            exp_results_path.parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(exp_results_path, mode="w", header=True, index=False)


if __name__ == "__main__":
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    config = auto_cli(Settings)
    main(config)
