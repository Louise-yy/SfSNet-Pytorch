import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cv2

from SfSNet_test import _decomposition
from src.utils import convert


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

    al_out3 = convert(al_out3)
    al_out3_m = convert(al_out3_m)
    b, g, r = cv2.split(al_out3)
    bm, gm, rm = cv2.split(al_out3_m)

    plt.figure(dpi=100, figsize=(25, 25))

    # 原始图和直方图 ---------------------------B------------------------------
    plt.subplot(6, 3, 1)
    plt.title("original img")
    plt.imshow(b, cmap="gray")

    plt.subplot(6, 3, 4)
    hist_s = arrayToHist(b, 256)
    drawHist(hist_s, "original histogram")

    # match图和其直方图
    plt.subplot(6, 3, 2)
    plt.title("match img")
    plt.imshow(bm, cmap="gray")

    plt.subplot(6, 3, 5)
    hist_m = arrayToHist(bm, 256)
    drawHist(hist_m, "match histogram")

    # match后的图片及其直方图
    im_d_b = histMatch(b, hist_m)  # 将目标图的直方图用于给原图做均衡，也就实现了match
    plt.subplot(7, 3, 3)
    plt.title("img after match")
    plt.imshow(im_d_b, cmap="gray")

    plt.subplot(7, 3, 6)
    hist_d = arrayToHist(im_d_b, 256)
    drawHist(hist_d, "histogram after match")

    # 原始图和直方图------------------------------G---------------------------
    plt.subplot(7, 3, 7)
    plt.title("original img")
    plt.imshow(g)

    plt.subplot(7, 3, 10)
    hist_s = arrayToHist(g, 256)
    drawHist(hist_s, "original histogram")

    # match图和其直方图
    plt.subplot(7, 3, 8)
    plt.title("match img")
    plt.imshow(gm)

    plt.subplot(7, 3, 11)
    hist_m = arrayToHist(gm, 256)
    drawHist(hist_m, "match histogram")

    # match后的图片及其直方图
    im_d_g = histMatch(g, hist_m)  # 将目标图的直方图用于给原图做均衡，也就实现了match
    plt.subplot(7, 3, 9)
    plt.title("img after match")
    plt.imshow(im_d_g)

    plt.subplot(7, 3, 12)
    hist_d = arrayToHist(im_d_g, 256)
    drawHist(hist_d, "histogram after match")

    # 原始图和直方图 -------------------------------------R----------------------
    plt.subplot(7, 3, 13)
    plt.title("original img")
    plt.imshow(r)

    plt.subplot(7, 3, 16)
    hist_s = arrayToHist(r, 256)
    drawHist(hist_s, "original histogram")

    # match图和其直方图
    plt.subplot(7, 3, 14)
    plt.title("match img")
    plt.imshow(rm)

    plt.subplot(7, 3, 17)
    hist_m = arrayToHist(rm, 256)
    drawHist(hist_m, "match histogram")

    # match后的图片及其直方图
    im_d_r = histMatch(r, hist_m)  # 将目标图的直方图用于给原图做均衡，也就实现了match
    plt.subplot(7, 3, 15)
    plt.title("img after match")
    plt.imshow(im_d_r)

    plt.subplot(7, 3, 18)
    hist_d = arrayToHist(im_d_r, 256)
    drawHist(hist_d, "histogram after match")

    # -----------------------------------------------------
    plt.subplot(7, 3, 19)
    plt.imshow(al_out3)

    im_d_b = im_d_b.astype(np.uint8)
    im_d_g = im_d_g.astype(np.uint8)
    im_d_r = im_d_g.astype(np.uint8)

    plt.subplot(7, 3, 20)
    img_after_match = cv2.merge([im_d_b, im_d_g, im_d_r])
    plt.imshow(img_after_match)

    plt.show()
    #
    img_after = cv2.merge([b, g, r])
    cv2.imshow("b", b)
    cv2.imshow("im_d_b", im_d_b)
    cv2.imshow("g", g)
    cv2.imshow("im_d_g", im_d_g)
    cv2.imshow("r", r)
    cv2.imshow("im_d_r", im_d_r)
    cv2.imshow("img", al_out3)
    cv2.imshow("match", al_out3_m)
    # cv2.imshow("img_after", img_after)
    cv2.imshow("img_after_match", img_after_match)
    # # cv2.imshow("bm", bm)
    cv2.waitKey(0)

