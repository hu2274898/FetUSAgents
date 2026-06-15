from tkinter import image_names
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from utils.generate_prompts import get_click_prompt
import time
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import SimpleITK
from tqdm import tqdm


class Evaluation:
    def __init__(self, a):
        self.results = []
        pred_arr = cv2.resize(a, (512, 512), interpolation=cv2.INTER_NEAREST)
        self.pre_image = SimpleITK.GetImageFromArray(pred_arr)

    def evaluation(self, pred: SimpleITK.Image):
        result = dict()
        pred_aop = self.cal_aop(pred)
        result['aop_truth'] = pred_aop
        return result

    def process(self):
        pre_image = self.pre_image
        result = self.evaluation(pre_image)
        return result



    def cal_aop(self, pred):
        aop = 0.0
        ellipse = None
        ellipse2 = None
        pred_data = SimpleITK.GetArrayFromImage(pred)
        aop_pred = np.array(self.onehot_to_mask(pred_data)).astype(np.uint8)
        contours, _ = cv2.findContours(cv2.medianBlur(aop_pred[:, :, 1], 1), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        contours2, _ = cv2.findContours(cv2.medianBlur(aop_pred[:, :, 2], 1), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)
        maxindex1 = 0
        maxindex2 = 0
        max1 = 0
        max2 = 0
        flag1 = 0
        flag2 = 0
        for j in range(len(contours)):
            if contours[j].shape[0] > max1:
                maxindex1 = j
                max1 = contours[j].shape[0]
            if j == len(contours) - 1:
                approxCurve = cv2.approxPolyDP(contours[maxindex1], 1, closed=True)
                if approxCurve.shape[0] > 5:
                    ellipse = cv2.fitEllipse(approxCurve)
                flag1 = 1
        for k in range(len(contours2)):
            if contours2[k].shape[0] > max2:
                maxindex2 = k
                max2 = contours2[k].shape[0]
            if k == len(contours2) - 1:
                approxCurve2 = cv2.approxPolyDP(contours2[maxindex2], 1, closed=True)
                if approxCurve2.shape[0] > 5:
                    ellipse2 = cv2.fitEllipse(approxCurve2)
                flag2 = 1
        if flag1 == 1 and flag2 == 1 and ellipse2 != None and ellipse != None:
            aop = drawline_AOD(ellipse2, ellipse)
        return aop

    def onehot_to_mask(self, mask):
        ret = np.zeros([3, 512, 512])
        tmp = mask.copy()
        tmp[tmp == 1] = 255
        tmp[tmp == 2] = 0
        ret[1] = tmp
        tmp = mask.copy()
        tmp[tmp == 2] = 255
        tmp[tmp == 1] = 0
        ret[2] = tmp
        b = ret[0]
        r = ret[1]
        g = ret[2]
        ret = cv2.merge([b, r, g])
        mask = ret.transpose([0, 1, 2])
        return mask


def drawline_AOD(element_, element_1):
    import math
    element = (element_[0], (element_[1][1], element_[1][0]), element_[2] - 90)
    element1 = (element_1[0], (element_1[1][1], element_1[1][0]), element_1[2] - 90)

    [d11, d12] = [element1[0][0] - element1[1][0] / 2 * math.cos(element1[2] * 0.01745),
                  element1[0][1] - element1[1][0] / 2 * math.sin(element1[2] * 0.01745)]
    [d21, d22] = [element1[0][0] + element1[1][0] / 2 * math.cos(element1[2] * 0.01745),
                  element1[0][1] + element1[1][0] / 2 * math.sin(element1[2] * 0.01745)]
    # cv2.line(background, (round(d11), round(d12)), (round(d21), round(d22)), (255, 255, 255), 2)
    a = element[1][0] / 2
    b = element[1][1] / 2
    angel = 2 * math.pi * element[2] / 360
    dp21 = d21 - element[0][0]
    dp22 = d22 - element[0][1]

    dp2 = np.array([[dp21], [dp22]])
    Transmat1 = np.array([[math.cos(-angel), -math.sin(-angel)],
                          [math.sin(-angel), math.cos(-angel)]])
    Transmat2 = np.array([[math.cos(angel), -math.sin(angel)],
                          [math.sin(angel), math.cos(angel)]])
    dpz2 = Transmat1 @ dp2
    dpz21 = dpz2[0][0]
    dpz22 = dpz2[1][0]
    if dpz21 ** 2 - a ** 2 == 0:
        dpz21 += 1
    if (b ** 2 * dpz21 ** 2 + a ** 2 * dpz22 ** 2 - a ** 2 * b ** 2) >= 0:
        xielv_aod = (dpz21 * dpz22 - math.sqrt(b ** 2 * dpz21 ** 2 + a ** 2 * dpz22 ** 2 - a ** 2 * b ** 2)) / (
                dpz21 ** 2 - a ** 2)
    else:
        xielv_aod = 0
    bias_aod = dpz22 - xielv_aod * dpz21
    qiepz1 = (-2 * xielv_aod * bias_aod / b ** 2) / (2 * (1 / a ** 2 + xielv_aod ** 2 / b ** 2))
    qiepz2 = qiepz1 * xielv_aod + bias_aod
    qiepz = np.array([[qiepz1], [qiepz2]])
    qiep = list(Transmat2 @ qiepz)
    qie1 = qiep[0][0] + element[0][0]
    qie2 = qiep[1][0] + element[0][1]

    ld1d3 = math.sqrt((d11 - d21) ** 2 + (d12 - d22) ** 2)
    ld3x4 = math.sqrt((d21 - qie1) ** 2 + (d22 - qie2) ** 2)
    ld1x4 = math.sqrt((d11 - qie1) ** 2 + (d12 - qie2) ** 2)

    aod = math.acos((ld1d3 ** 2 + ld3x4 ** 2 - ld1x4 ** 2) / (2 * ld1d3 * ld3x4)) / math.pi * 180  ##余弦定理
    # cv2.line(background, (round(d21), round(d22)), (int(qie1), int(qie2)), (255, 255, 255), 2)
    return aod


if __name__ == '__main__':
    pred_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_predictions")


    pred_files = [f for f in os.listdir(pred_dir) if f.endswith(".png")]
    results = []

    for f in pred_files:

        pred_path = os.path.join(pred_dir, f)



        # 读图并 resize
        pred_png = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        pred_png = cv2.resize(pred_png, (512, 512), interpolation=cv2.INTER_NEAREST)
        pred_arr = np.array(pred_png, dtype=np.uint8)

        evaluator = Evaluation(pred_arr)
        result = evaluator.process()   # 这里返回的是 {"score":..., "loss":..., "dice":..., ...}


        print(f'File: {f}, Predicted AOP: {result["aop_truth"]:.2f} degrees')
