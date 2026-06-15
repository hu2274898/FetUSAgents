import logging
import cv2
import os
from sklearn.linear_model import LinearRegression
import numpy as np
import torch
import torch.nn as nn
import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils
from torchmetrics import (
    MetricCollection,
    Accuracy,
    Recall,
    F1Score,
    Precision,
    Specificity,
    ConfusionMatrix,
)

from embeddings import EncoderWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def mcc_edge(in_img):
    """
    Extract max connected component and then extract edge.
    """

    img = in_img
    if in_img.dtype != 'uint8':
        in_img = in_img * 255
        img = in_img.astype('uint8')

    # Max connected component extraction
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=4)
    sort_label = np.argsort(-stats[:, 4], )

    idx = labels == sort_label[1]
    max_connect = idx * 255
    max_connect = max_connect.astype('uint8')

    # Edge detection
    edge_img = cv2.Canny(max_connect, 50, 250)

    return edge_img
def ellip_fit(in_img):
    """
    To fit the fetal head contour into an ellipse, output 5 ellipse parameters.
    Note: The unit of parameters obtained is pixel distance, and the coordinate system is as follows:
    the upper left corner of the image is the origin,vertically downward is the X-axis direction,
    horizontally right is the Y-axis direction.
    """

    edge_img = in_img
    if in_img.dtype != 'uint8':
        in_img = in_img * 255
        edge_img = in_img.astype('uint8')

    # Get coordinates of edge points
    edge_points = np.where(edge_img == 255)
    edge_x = edge_points[0]
    edge_y = edge_points[1]
    edge_x = edge_x.reshape(-1, 1)  # (N, 1)
    edge_y = edge_y.reshape(-1, 1)  # (N, 1)

    # least squares fitting
    x2 = edge_x * edge_x
    xy = 2 * edge_x * edge_y
    _2x = -2 * edge_x
    _2y = -2 * edge_y
    mine_1 = -np.ones(edge_x.shape)
    X = np.concatenate((x2, xy, _2x, _2y, mine_1), axis=1)
    y = -edge_y * edge_y


    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    k1 = model.coef_[0, 0]
    k2 = model.coef_[0, 1]
    k3 = model.coef_[0, 2]
    k4 = model.coef_[0, 3]
    k5 = model.coef_[0, 4]

    # Calculate parameters: xc,yc,theta,a,b
    xc = (k3 - k2 * k4) / (k1 - k2 * k2)
    yc = (k1 * k4 - k2 * k3) / (k1 - k2 * k2)
    theta = 0.5 * np.arctan(2 * k2 / (k1 - 1))


    T = np.tan(theta)
    K = (1 - k1 * T * T) / (k1 - T * T)                         # a^2 = K * b^2
    p1 = -np.square(xc + T * yc)
    p2 = -np.square(xc * T - yc)
    b_2 = (k5 * (T * T + K) - p1 - p2 * K) / (K * (T * T + 1))  # b^2
    a_2 = K * b_2                                               # a^2

    a = np.sqrt(a_2)
    b = np.sqrt(b_2)

    # Set a to the long half axis and b to the short half axis, and adjust the angle
    if a < b:
        t = b
        b = a
        a = t
        theta = theta + 0.5 * np.pi

    return xc,yc,theta,a,b
def process_prediction(pred_path, orig_img_root):
    # pred_path 是刚保存的预测图路径
    img_name = os.path.basename(pred_path)
    edge_img = mcc_edge(cv2.imread(pred_path, 0))
    orig_img_path = os.path.join(orig_img_root, img_name)
    orig_img = cv2.imread(orig_img_path, 0)
    if orig_img is None:
        print(f"Original image not found: {orig_img_path}")
        return
    gt_h, gt_w = orig_img.shape[:2]
    # 椭圆拟合
    xc, yc, theta, a, b = ellip_fit(edge_img)


    # 询问 pixel size
    default_pixel_size = 0.15
    user_input = input(f"Press input pixel size for {img_name}: ").strip()
    if user_input:
        try:
            pixel_size = float(user_input)
        except:
            print("Invalid input, using default.")
            pixel_size = default_pixel_size
    else:
        pixel_size = default_pixel_size
    pred_h, pred_w = 224,224
    scale_x = gt_w / pred_w
    scale_y = gt_h / pred_h
    xc = (xc + 0.5) * scale_x - 0.5
    yc = (yc + 0.5) * scale_y - 0.5
    a = a * scale_x
    b = b * scale_y

    # 计算 HC
    hc = 2 * np.pi * b + 4 * (a - b)
    print(f'prediction HC of {img_name} : {hc * pixel_size:.2f} mm')


class ClassificationModel(L.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int,
        num_classes: int,
        freeze_encoder: bool,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])

        self.num_classes = num_classes

        # network architecture
        self.encoder = encoder
        self.classifier = nn.Linear(input_dim, num_classes)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # loss function
        self.loss_fn = nn.BCEWithLogitsLoss()

        # optimizer
        self.lr = 1e-4

        # metrics
        self.val_metrics = MetricCollection(
            {
                "accuracy": Accuracy(task="binary"),
                "f1": F1Score(task="binary"),
                "recall": Recall(task="binary"),
                "precision": Precision(task="binary"),
                "specificity": Specificity(task="binary"),
            },
            prefix="val/",
        )
        self.test_metrics = self.val_metrics.clone(prefix="test/")
        self.test_metrics.add_metrics({"confmat": ConfusionMatrix(task="binary")})

        # Embedding visualization [1/3]: save test embeddings for visualization
        # self.test_embs = []
        # self.test_emb_labels = []

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]

        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]

        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.log("val/loss", loss, prog_bar=True)

        probs = torch.sigmoid(logits)
        self.val_metrics.update(probs, y)

    def on_validation_epoch_end(self):
        results = self.val_metrics.compute()
        self.val_metrics.reset()

        self.log("val/f1", results.pop("val/f1"), prog_bar=True)
        self.log_dict(results)

    def test_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]

        logits = self(x)

        probs = torch.sigmoid(logits)
        self.test_metrics.update(probs, y)

        # Embedding visualization [2/3]: save test embeddings for visualization
        # embeddings = self.encoder(x)
        # embeddings = embeddings.detach().cpu()
        # self.test_embs.append(embeddings)
        # self.test_emb_labels.append(y)

    def on_test_epoch_end(self):
        results = self.test_metrics.compute()
        self.test_metrics.reset()

        confmat = results.pop("test/confmat").cpu().numpy()
        self.log_dict(results)
        for i in range(2):
            for j in range(2):
                self.log(f"test/confmat_{i}_{j}", confmat[i, j])

        ## Embedding visualization [3/3]: save test embeddings for visualization
        # test_embs = torch.cat(self.test_embs, dim=0)
        # test_emb_labels = torch.cat(self.test_emb_labels, dim=0)
        # torch.save(test_embs, "test_embs.pt")
        # torch.save(test_emb_labels, "test_emb_labels.pt")
        # self.test_embs.clear()
        # self.test_emb_labels.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


class SegmentationModel(L.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int,
        num_classes: int,
        freeze_encoder: bool,
        root_path: str,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])
        self.num_classes = num_classes
        # network architecture
        self.encoder = EncoderWrapper(encoder)
        self.decoder = UNETR(input_dim, num_classes, input_dim=3, init_filters=32)
        self.root_path = root_path

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # loss function
        # self.loss_fn = smp.losses.FocalLoss(
        #     mode="binary",     # 你的任务是 multilabel
        #     alpha=0.25,            # 可设为 None 或按类频率自定义张量
        #     gamma=2.0,             # 聚焦难样本
        # )
        self.loss_fn = smp.losses.DiceLoss(mode="multilabel", from_logits=True)

        # optimizer
        self.lr = 3e-4

        # validation step outputs
        self.val_test_step_outputs = []

        # metric
        self.test_metrics = MetricCollection(
            {
                "accuracy": Accuracy(task="binary"),
                "f1": F1Score(task="binary"),
                "recall": Recall(task="binary"),
                "precision": Precision(task="binary"),
                "specificity": Specificity(task="binary"),
                "confmat": ConfusionMatrix(task="binary"),
            },
            prefix="test/",
        )

        # threshold
        self.threshold = 224 * 224 * 0.01  # = 501

    def forward(self, x):
        embs = self.encoder(x)
        x = self.decoder([x, *embs.values()])
        return x

    def training_step(self, batch, batch_nb):
        x = batch["image"]
        y = batch["mask"]

        logits = self(x)
        loss = self.loss_fn(logits, y)

        dice = smp_utils.metrics.Fscore(activation="sigmoid")(logits, y)

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/dice", dice, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_nb):
        x = batch["image"]
        y = batch["mask"]

        logits = self(x)

        self.val_test_step_outputs.append((logits, y))

        return logits, y

    def on_validation_epoch_end(self):
        preds = []
        targets = []
        for outs in self.val_test_step_outputs:
            preds.append(outs[0])
            targets.append(outs[1])

        preds = torch.cat(preds).cpu()
        targets = torch.cat(targets).cpu()

        loss = self.loss_fn(preds, targets)
        dice = smp_utils.metrics.Fscore(activation="sigmoid")(preds, targets)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/dice", dice, prog_bar=True)

        self.val_test_step_outputs.clear()

    def test_step(self, batch, batch_nb):
        x = batch["image"]

        filename = batch['filename']

        logits = self(x)

        self.val_test_step_outputs.append((logits,filename))

        probs = torch.sigmoid(logits)
        print(probs.shape)
        print("max, min values of probs")
        print(probs.max(), probs.min())


    def on_test_epoch_end(self):
        preds = []
        filename = []
        for outs in self.val_test_step_outputs:
            preds.append(outs[0])
            filename.extend(outs[1])

        preds = torch.cat(preds).cpu()


        # ---------- 保存预测结果和 GT ---------- 
        import torchvision.transforms.functional as TF 
        from pathlib import Path 
        save_dir = Path("outputs/test_predictions") 
        save_dir.mkdir(parents=True, exist_ok=True) # 对每个样本保存预测图和真值图 
        for i in range(preds.shape[0]):
             pred_mask = (torch.sigmoid(preds[i]) > 0.5).float() 

             base_name = Path(filename[i]).stem 
             # 去掉扩展名 
             pred_path = save_dir / f"{base_name}.png" 

             TF.to_pil_image(pred_mask).save(pred_path) 
             
        all_paths = sorted(
            list(save_dir.glob("*.png")) +
            list(save_dir.glob("*.jpg")) +
            list(save_dir.glob("*.jpeg"))
        )
        input_path = self.root_path
        for img_path in all_paths:
            img_name = img_path.stem

            # 对保存的结果做你的 postprocess
            process_prediction(str(img_path), input_path)

        print(f"Saved {preds.shape[0]} predicted masks and GT masks to {save_dir}")

        self.val_test_step_outputs.clear()

        ## save predicted masks and ground truth masks for visualization
        # from pathlib import Path

        # import torchvision.transforms.functional as TF

        # example_out_folder = Path("outputs/segmentation_examples")
        # example_out_folder.mkdir(exists_ok=True)

        # empty_counts = 0
        # for i in range(preds.size(0)):
        #     pred_mask = (torch.sigmoid(preds[i]) > 0.5).float()
        #     gt_mask = targets[i]
        #     if gt_mask.max() == 1:
        #         pred_save_path = example_out_folder / f"{i}_pred.png"
        #         gt_save_path = example_out_folder / f"{i}_gt.png"
        #         TF.to_pil_image(pred_mask).save(pred_save_path)
        #         TF.to_pil_image(gt_mask).save(gt_save_path)
        #     elif empty_counts < 50:
        #             pred_save_path = example_out_folder / f"empty_{i}_pred.png"
        #             gt_save_path = example_out_folder / f"empty_{i}_gt.png"
        #             TF.to_pil_image(pred_mask).save(pred_save_path)
        #             TF.to_pil_image(gt_mask).save(gt_save_path)
        #         empty_counts += 1

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


"""
REFERENCES:
- https://github.com/tamasino52/UNETR/blob/main/unetr.py#L171
"""


class SingleDeconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, groups=1):
        super().__init__()
        self.block = nn.ConvTranspose2d(
            in_planes,
            out_planes,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
            groups=groups,
        )

    def forward(self, x):
        return self.block(x)


class SingleConv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1):
        super().__init__()
        self.block = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=1,
            padding=((kernel_size - 1) // 2),
            groups=groups,
        )

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

        self.decoder0 = nn.Sequential(
            Conv2DBlock(input_dim, init_filters, 3),
            Conv2DBlock(init_filters, init_filters, 3),
        )

        self.decoder3 = nn.Sequential(
            Deconv2DBlock(transformer_width, 8 * init_filters),
            Deconv2DBlock(8 * init_filters, 4 * init_filters),
            Deconv2DBlock(4 * init_filters, 2 * init_filters),
        )

        self.decoder6 = nn.Sequential(
            Deconv2DBlock(transformer_width, 8 * init_filters),
            Deconv2DBlock(8 * init_filters, 4 * init_filters),
        )

        self.decoder9 = Deconv2DBlock(transformer_width, 8 * init_filters)

        self.decoder12_upsampler = SingleDWConv2DBlock(
            transformer_width, 8 * init_filters
        )

        self.decoder9_upsampler = nn.Sequential(
            Conv2DBlock(16 * init_filters, 8 * init_filters),
            Conv2DBlock(8 * init_filters, 8 * init_filters),
            Conv2DBlock(8 * init_filters, 8 * init_filters),
            SingleDWConv2DBlock(8 * init_filters, 4 * init_filters),
        )

        self.decoder6_upsampler = nn.Sequential(
            Conv2DBlock(8 * init_filters, 4 * init_filters),
            Conv2DBlock(4 * init_filters, 4 * init_filters),
            SingleDWConv2DBlock(4 * init_filters, 2 * init_filters),
        )

        self.decoder3_upsampler = nn.Sequential(
            Conv2DBlock(4 * init_filters, 2 * init_filters),
            Conv2DBlock(2 * init_filters, 2 * init_filters),
            SingleDWConv2DBlock(2 * init_filters, init_filters),
        )

        self.decoder0_header = nn.Sequential(
            Conv2DBlock(2 * init_filters, init_filters),
            Conv2DBlock(init_filters, init_filters),
            SingleConv2DBlock(init_filters, output_dim, 1),
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
