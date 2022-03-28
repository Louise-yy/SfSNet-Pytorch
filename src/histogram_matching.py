import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cv2

from SfSNet_test import _decomposition


def arrayToHist(grayArray, nums):  # 将灰度数组映射为直方图字典,nums表示灰度的数量级
    if len(grayArray.shape) != 2:
        print("length error")
        return None
    w, h = grayArray.shape
    hist = {}
    for k in range(nums):
        hist[k] = 0
    for i in range(w):
        for j in range(h):
            if hist.get(grayArray[i][j]) is None:
                hist[grayArray[i][j]] = 0
            hist[grayArray[i][j]] += 1
    # normalize
    n = w * h
    for key in hist.keys():
        hist[key] = float(hist[key]) / n
    return hist


def drawHist(hist, name):  # 传入的直方图要求是个字典，每个灰度对应着概率
    keys = hist.keys()
    values = hist.values()
    x_size = len(hist) - 1  # x轴长度，也就是灰度级别
    axis_params = [0, x_size]

    # plt.figure()
    if name is not None:
        plt.title(name)
    plt.bar(tuple(keys), tuple(values))  # 绘制直方图
    # plt.show()


def histMatch(grayArray, h_d):  # 直方图匹配函数，接受原始图像和目标灰度直方图
    # 计算累计直方图
    tmp = 0.0
    h_acc = h_d.copy()
    for i in range(256):
        tmp += h_d[i]
        h_acc[i] = tmp

    h1 = arrayToHist(grayArray, 256)  # 原图的直方图
    tmp = 0.0
    h1_acc = h1.copy()
    for i in range(256):
        tmp += h1[i]
        h1_acc[i] = tmp
    # 计算映射
    M = np.zeros(256)
    for i in range(256):
        idx = 0
        minv = 1
        for j in h_acc:
            if np.fabs(h_acc[j] - h1_acc[i]) < minv:
                minv = np.fabs(h_acc[j] - h1_acc[i])
                idx = int(j)
        M[i] = idx
    des = M[grayArray]
    return des


if __name__ == '__main__':
    # img = cv2.imread("D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/Images/11.png_face.png")
    # img_match = cv2.imread("D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/Images/4.png_face.png")

    n_out2, al_out2, light_out, al_out3, n_out3 = _decomposition(
        "D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/Images/11.png_face.png")

    n_out2_m, al_out2_m, light_out_m, al_out3_m, n_out3_m = _decomposition(
        "D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/Images/4.png_face.png")



