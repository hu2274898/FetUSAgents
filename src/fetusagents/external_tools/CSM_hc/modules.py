"""
Components for data preprocessing, model training, prediction,
postprocessing, visulization and evalution.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import time
import os
import numpy as np
import pandas as pd
import cv2
from sklearn.linear_model import LinearRegression

# Deep model
class HcDataset(Dataset):
    """
    Create dataset for model training
    """

    def __init__(self, input_folder, label_folder, input_size=[192, 128]):

        self.input_w = int(input_size[0])
        self.input_h = int(input_size[1])

        self.label1_w = int(self.input_w / 8)
        self.label1_h = int(self.input_h / 8)

        self.label2_w = int(self.input_w / 4)
        self.label2_h = int(self.input_h / 4)

        self.input_list = os.listdir(input_folder)

        self.input_folder = input_folder
        self.label_folder = label_folder

        self.dataset_L = len(self.input_list)

    def __len__(self):
        return self.dataset_L

    def __getitem__(self, idx):

        input_name = self.input_list[idx]
        label_name = input_name[:-4] + '_Annotation.png'

        input_file_path = self.input_folder + input_name
        label_file_path = self.label_folder + label_name

        img = cv2.imread(input_file_path,0)
        label = cv2.imread(label_file_path,0)

        # Resize images and labels
        img = cv2.resize(img, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA)
        label1 = cv2.resize(label,(self.label1_w, self.label1_h),interpolation=cv2.INTER_AREA)
        label2 = cv2.resize(label,(self.label2_w, self.label2_h),interpolation=cv2.INTER_AREA)

        img = img / 255
        label1 = label1 / 255
        label2 = label2 / 255

        img = img.astype('float32')
        label1 = label1.astype('float32')
        label2 = label2.astype('float32')

        input = torch.from_numpy(img)
        label1 = torch.from_numpy(label1)
        label2 = torch.from_numpy(label2)

        input = input.unsqueeze(0)
        label1 = label1.unsqueeze(0)
        label2 = label2.unsqueeze(0)

        return input, label1,label2


class CSM(nn.Module):
    """
    Convolutional segmentation machine.
    """

    def __init__(self):
        super(CSM,self).__init__()
        self.stage1 = nn.Sequential(nn.Conv2d(1, 8, 9, padding=4),
                                   nn.BatchNorm2d(8),
                                   nn.ReLU(True),
                                   nn.MaxPool2d(2),
                                   nn.Conv2d(8, 8, 9, padding=4, groups=8),
                                   nn.Conv2d(8, 16, 1),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(True),
                                   nn.MaxPool2d(2),
                                   nn.Conv2d(16, 16, 9, padding=4, groups=16),
                                   nn.Conv2d(16, 32, 1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(True),
                                   nn.MaxPool2d(2),
                                   nn.Conv2d(32, 32, 5, padding=2, groups=32),
                                   nn.Conv2d(32, 64, 1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(True),
                                   nn.Conv2d(64, 64, 9, padding=4, groups=64),
                                   nn.Conv2d(64, 32, 1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(True),
                                   nn.Conv2d(32, 16, 1),
                                   nn.ReLU(True),
                                   nn.Conv2d(16, 1, 1),
                                   nn.Sigmoid())

        self.f1 = nn.Sequential(nn.Conv2d(1, 8, 9, padding=4),
                                nn.BatchNorm2d(8),
                                nn.ReLU(True),
                                nn.MaxPool2d(2),
                                nn.Conv2d(8, 16, 9, padding=4),
                                nn.BatchNorm2d(16),
                                nn.ReLU(True),
                                nn.MaxPool2d(2),
                                nn.Conv2d(16, 32, 9, padding=4),
                                nn.BatchNorm2d(32),
                                nn.ReLU(True))

        self.f2 = nn.Sequential(nn.MaxPool2d(2),
                                  nn.Conv2d(32, 16, 5, padding=2),
                                  nn.BatchNorm2d(16),
                                  nn.ReLU(True))

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.f3 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1),
                                  nn.BatchNorm2d(16),
                                  nn.ReLU(True),
                                  nn.Conv2d(16, 16, 3, padding=1),
                                  nn.BatchNorm2d(16),
                                  nn.ReLU(True))

        self.stage2 = CSM_stagen()
        self.stage3 = CSM_stagen()

    def forward(self,x):
        y1 = self.stage1(x)
        x_f1 = self.f1(x)

        x_f2 = self.f2(x_f1)
        x_f3 = self.f3(x_f1)

        x1 = torch.cat([y1,x_f2],1)
        y2 = self.stage2(x1)
        y2_up = self.up(y2)
        x2 = torch.cat([y2_up,x_f3],1)
        y3 = self.stage3(x2)

        return y1, y2, y3


class CSM_stagen(nn.Module):
    """
    Network of n(n>=2) stage in CSM.
    """
    def __init__(self):
        super(CSM_stagen,self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(17, 17, 11, padding=5,groups=17),
                                  nn.Conv2d(17, 32, 1),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU(True),
                                  nn.Conv2d(32, 32, 11, padding=5, groups=32),
                                  nn.Conv2d(32, 64, 1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(True),
                                  nn.Conv2d(64, 64, 11, padding=5, groups=64),
                                  nn.Conv2d(64, 32, 1),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU(True),
                                  nn.Conv2d(32, 16, 1),
                                  nn.BatchNorm2d(16),
                                  nn.ReLU(True),
                                  nn.Conv2d(16, 1, 1),
                                  nn.Sigmoid())

    def forward(self,x):
        x = self.conv(x)

        return x

# Preprocess
def img_crop(in_img, crop_size=[512,768]):
    """
    Crop an image to a specific size.
    """

    crop_img = np.zeros([512,768],dtype=in_img.dtype)

    r_out = crop_size[0]
    c_out = crop_size[1]

    r_in, c_in = in_img.shape

    dr = int((r_in - r_out) / 2 + 0.5)
    dc = int((c_in - c_out) / 2 + 0.5)

    if dr > 0:
        rp_in = [dr,dr+r_out]
        rp_out = [0, r_out]
    else:
        dr = -dr
        rp_in = [0,r_in]
        rp_out = [dr,dr+r_in]

    if dc > 0:
        cp_in = [dc,dc+c_out]
        cp_out = [0, c_out]
    else:
        dc = -dc
        cp_in = [0,c_in]
        cp_out = [dc,dc+c_in]

    crop_img[rp_out[0]:rp_out[1],cp_out[0]:cp_out[1]] = in_img[rp_in[0]:rp_in[1],cp_in[0]:cp_in[1]]

    return crop_img


def ellip_fill(in_folder, out_folder):
    """
    Fill the original label to create segmentation mask.
    """

    dirs = os.listdir(in_folder)
    for i in range(len(dirs)):
        print('Ellip filling: %d / %d' % (i + 1, len(dirs)))
        img_name = dirs[i]
        img_path = in_folder + img_name

        img = cv2.imread(img_path, 0)

        img = 255 - img
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=4)
        ellipse = labels == 2
        ellipse = ellipse * 255
        ellipse = ellipse.astype('uint8')

        # Original label is 3 pixels wide so an expansion process is needed
        kernel = np.ones((3, 3), np.uint8)
        ellipse = cv2.dilate(ellipse, kernel, iterations=1)

        # Save
        save_path = out_folder + img_name
        cv2.imwrite(save_path, ellipse)


def img_augumentation(in_folder, out_folder, data):
    """
    Data augumentation, operations include rotation of 10, 20 and 30 degrees in clockwise and counterclockwise,
    magnification of 1.15 times, narrowing of 0.85 times, left and right flip, up and down flip.
    """

    dirs = os.listdir(in_folder)
    for i in range(len(dirs)):
        print('%s augumentation: %d / %d' % (data, i + 1, len(dirs)))
        img_name = dirs[i]
        img_path = in_folder + img_name
        img = cv2.imread(img_path, 0)
        h,w = img.shape

        # Image scale
        matRotate_085 = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), 0, 0.85)
        matRotate_115 = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), 0, 1.15)

        img_085 = cv2.warpAffine(img, matRotate_085, (w, h))
        img_115 = cv2.warpAffine(img, matRotate_115, (w, h))

        # Flip
        img_Flr = cv2.flip(img, 1)  # flip left-right
        img_Fud = cv2.flip(img, 0)  # flip up-down

        # Rotate clockwise and counterclockwise
        matRotate_p10 = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), 10, 1)
        matRotate_p20 = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), 20, 1)
        matRotate_p30 = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), 30, 1)
        matRotate_m10 = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), -10, 1)
        matRotate_m20 = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), -20, 1)
        matRotate_m30 = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), -30, 1)

        img_Rp10 = cv2.warpAffine(img, matRotate_p10, (w, h))
        img_Rp20 = cv2.warpAffine(img, matRotate_p20, (w, h))
        img_Rp30 = cv2.warpAffine(img, matRotate_p30, (w, h))
        img_Rm10 = cv2.warpAffine(img, matRotate_m10, (w, h))
        img_Rm20 = cv2.warpAffine(img, matRotate_m20, (w, h))
        img_Rm30 = cv2.warpAffine(img, matRotate_m30, (w, h))

        # Image name
        if data=="image":
            img_085_name = img_name[:-4] + '_S.png'
            img_115_name = img_name[:-4] + '_L.png'
            img_Flr_name = img_name[:-4] + '_Flr.png'
            img_Fud_name = img_name[:-4] + '_Fud.png'
            img_Rp10_name = img_name[:-4] + '_Rp10.png'
            img_Rp20_name = img_name[:-4] + '_Rp20.png'
            img_Rp30_name = img_name[:-4] + '_Rp30.png'
            img_Rm10_name = img_name[:-4] + '_Rm10.png'
            img_Rm20_name = img_name[:-4] + '_Rm20.png'
            img_Rm30_name = img_name[:-4] + '_Rm30.png'
        elif data == "label":
            img_085_name = img_name[:-15] + '_S_Annotation.png'
            img_115_name = img_name[:-15] + '_L_Annotation.png'
            img_Flr_name = img_name[:-15] + '_Flr_Annotation.png'
            img_Fud_name = img_name[:-15] + '_Fud_Annotation.png'
            img_Rp10_name = img_name[:-15] + '_Rp10_Annotation.png'
            img_Rp20_name = img_name[:-15] + '_Rp20_Annotation.png'
            img_Rp30_name = img_name[:-15] + '_Rp30_Annotation.png'
            img_Rm10_name = img_name[:-15] + '_Rm10_Annotation.png'
            img_Rm20_name = img_name[:-15] + '_Rm20_Annotation.png'
            img_Rm30_name = img_name[:-15] + '_Rm30_Annotation.png'

        else:
            print("Image augumentation falied! Please set the input parameter 'data' to 'image' or 'label'!")
            break

        # Save path
        img_path = out_folder + img_name
        img_085_path = out_folder + img_085_name
        img_115_path = out_folder + img_115_name
        img_Flr_path = out_folder + img_Flr_name
        img_Fud_path = out_folder + img_Fud_name
        img_Rp10_path = out_folder + img_Rp10_name
        img_Rp20_path = out_folder + img_Rp20_name
        img_Rp30_path = out_folder + img_Rp30_name
        img_Rm10_path = out_folder + img_Rm10_name
        img_Rm20_path = out_folder + img_Rm20_name
        img_Rm30_path = out_folder + img_Rm30_name

        # Save results
        cv2.imwrite(img_path, img)
        cv2.imwrite(img_085_path, img_085)
        cv2.imwrite(img_115_path, img_115)
        cv2.imwrite(img_Flr_path, img_Flr)
        cv2.imwrite(img_Fud_path, img_Fud)
        cv2.imwrite(img_Rp10_path, img_Rp10)
        cv2.imwrite(img_Rp20_path, img_Rp20)
        cv2.imwrite(img_Rp30_path, img_Rp30)
        cv2.imwrite(img_Rm10_path, img_Rm10)
        cv2.imwrite(img_Rm20_path, img_Rm20)
        cv2.imwrite(img_Rm30_path, img_Rm30)

    print("Images augumentation finished!")


# Train and predict
def train_model(model, dataloader, epoches=20, lr=0.001, device='cuda', save_model_name='models/trained_model.pth'):
    """
    Train the deep model and save result.
    """

    model.to(device)
    print("Using {} device".format(device))

    # Loss function and optimization methods
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train
    time_start = time.time()
    for epoch in range(epoches):
        running_loss = 0.0
        for data in dataloader:
            inputs, label1, label2 = data[0].to(device), data[1].to(device), data[2].to(device)
            optimizer.zero_grad()
            pre1, pre2, outputs = model(inputs)

            loss1 = criterion(pre1, label1)
            loss2 = criterion(pre2, label1)
            loss3 = criterion(outputs, label2)
            loss = loss1 + loss2 + loss3
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print("This is epoch %d / %d" % ((epoch + 1), epoches))
        print(" running loss : %f" % running_loss)
    print("Finished training !")

    # Save model
    if save_model_name[-4:] != '.pth':
        save_model_name = save_model_name + '.pth'
    torch.save(model.state_dict(), save_model_name)
    time_end = time.time()
    print("Time cost: %f s" % (time_end - time_start))


def predict(model, input_folder, predict_folder,device="cpu"):
    """
    Predict using the trained model and save results.
    """

    model.to(device)
    model.eval()

    # Resize factor
    k1 = 4
    w = int(768 / k1)
    h = int(512 / k1)

    # Predict
    dirs = os.listdir(input_folder)
    time_start = time.time()
    for i in range(len(dirs)):
        
        img_name = dirs[i]
        predict_name = img_name
        img_path = os.path.join(input_folder, img_name)
        if not img_name.endswith(".png"):  # 只处理 png 文件
            continue
        img = cv2.imread(img_path, 0)

        # Resize images
        input_img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        input_img = input_img.astype('float32')
        input_img = input_img / 255.0

        # To torch.tensor
        input_img = torch.from_numpy(input_img)
        input_img = input_img.unsqueeze(0)
        input_img = input_img.unsqueeze(0)

        input_img = input_img.to(device)

        # Predcit
        _, _, predict = model(input_img)
        predict = predict[0, 0, :, :]
        predict = predict.cpu().detach().numpy()
        predict = np.round(predict)

        # Save
        predict = predict * 255
        predict = predict.astype('uint8')

        predict_path = predict_folder + predict_name

        cv2.imwrite(predict_path, predict)

    # Time consumed
    time_end = time.time()



# Postprocess
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


# Visualization
def draw_ellip(xc,yc,a,b,theta,in_img,color='r'):
    """
    Image visualization by drawing predicted ellipse on original ultrasound image.
    """

    if len(in_img.shape) == 3:
        img = in_img
    else:
        w,h = in_img.shape
        img = np.zeros([w,h,3],in_img.dtype)
        img[:,:,0] = in_img
        img[:,:,1] = in_img
        img[:,:,2] = in_img

    if img.dtype != 'uint8':
        img = img*255
        img = img.astype('uint8')

    # Calculate the elliptic coordinate points
    rng = np.arange(0, 2 * np.pi, 0.001)
    ellipse_x = a * np.sin(rng)
    ellipse_y = b * np.cos(rng)

    # Rotate
    theta = theta
    ellipse_x1 = ellipse_x * np.cos(theta) - ellipse_y * np.sin(theta)
    ellipse_y1 = ellipse_x * np.sin(theta) + ellipse_y * np.cos(theta)

    # Translation
    ellipse_x1 = ellipse_x1 + xc
    ellipse_y1 = ellipse_y1 + yc

    # Round
    ellipse_x1 = ellipse_x1.astype(int)
    ellipse_y1 = ellipse_y1.astype(int)

    ellipse_x1 = ellipse_x1 % w
    ellipse_y1 = ellipse_y1 % h

    # Draw ellipse
    img[ellipse_x1, ellipse_y1, 0] = 0
    img[ellipse_x1, ellipse_y1, 1] = 0
    img[ellipse_x1, ellipse_y1, 2] = 0
    if color=='b':
        img[ellipse_x1,ellipse_y1,0]=255
    elif color=='g':
        img[ellipse_x1,ellipse_y1,1]=255
    elif color=='r':
        img[ellipse_x1,ellipse_y1,2]=255
    else:
        img[ellipse_x1,ellipse_y1,0] = 255
        img[ellipse_x1, ellipse_y1, 1] = 255
        img[ellipse_x1, ellipse_y1, 2] = 255

    return img


# Evaluation
def dice(img1, img2):
    """
    Calculate dice coefficient between two images.
    """
    img1 = np.round(img1 / 255).astype('uint8')
    img2 = np.round(img2 / 255).astype('uint8')
    s1 = np.sum(img1)
    s2 = np.sum(img2)
    s = np.sum(cv2.bitwise_and(img1, img2))
    d = 2 * s / (s1 + s2)
    return d

def dice_folder(label_folder, predict_folder):
    """
    Calculate dice coefficient in between tow folder.
    """
    dirs = os.listdir(predict_folder)
    D = []
    print('Calculating dice coefficient ...')
    for i in range(len(dirs)):
        predict_name = dirs[i]
        predict_path = predict_folder + predict_name

        label_name = predict_name[:-4] + '_Annotation.png'
        label_path = label_folder + label_name

        label = cv2.imread(label_path, 0)
        predict = cv2.imread(predict_path,0)

        r,c = label.shape
        resize_predict = cv2.resize(predict, (c, r), interpolation=cv2.INTER_CUBIC)
        resize_predict = (np.round(resize_predict / 255) * 255).astype('uint8')

        d = dice(label, resize_predict)
        D.append(d)

    dice_series = pd.Series(D, index=dirs)
    return dice_series

def hausdorff_d(img1,img2):
    """
    Calculate hausdorff distance between two images.
    """

    _, contours1, _ = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    _, contours2, _ = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    c1 = contours1[0]
    c2 = contours2[0]

    hausdorff_sd = cv2.createHausdorffDistanceExtractor()

    d1 = hausdorff_sd.computeDistance(c1, c2)
    d2 = hausdorff_sd.computeDistance(c2, c1)

    if d1 > d2:
        d = d1
    else:
        d = d2
    return d

def hausdorff_folder(label_folder, predict_folder,pixel_size):
    """
    Calculate hausdorff distance between images in two folder.
    """

    dirs = os.listdir(predict_folder)
    D = []
    print('Calculating hausdorff distance ...')
    for i in range(len(dirs)):
        predict_name = dirs[i]
        predict_path = predict_folder + predict_name

        label_name = predict_name[:-4] + '_Annotation.png'
        label_path = label_folder + label_name

        label = cv2.imread(label_path, 0)
        predict = cv2.imread(predict_path, 0)

        r, c = label.shape
        resize_predict = cv2.resize(predict, (c, r), interpolation=cv2.INTER_CUBIC)
        resize_predict = (np.round(resize_predict / 255) * 255).astype('uint8')

        d = hausdorff_d(label, resize_predict) * pixel_size[predict_name]
        D.append(d)

    hausdorff_d_seris = pd.Series(D,index=dirs)
    return hausdorff_d_seris




