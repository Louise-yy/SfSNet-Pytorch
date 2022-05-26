import numpy as np
import cv2

from config import PROJECT_DIR
from src.functions import create_shading_recon
from src.utils import convert
from SfSNet_test import _decomposition
import matplotlib.pyplot as plt


def drawHis():
    _, _, _, img, _, _ = _decomposition(
        "D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/Images/11.png_face.png")

    ref = "D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/Images/4.png_face.png"
    n_out2, al_out2, light_out, al_out3, n_out3, mask2 = _decomposition(ref)

    al_out3 = convert(al_out3)
    al_out3 = cv2.cvtColor(al_out3, cv2.COLOR_BGR2RGB)
    img = convert(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = np.zeros_like(img)
    _, _, colorChannel = img.shape

    hist_img, _ = np.histogram(img[:, :, 2], 256)  # get the histogram  !!!!!!!!!!!!!!!!!!!!!!!!!
    hist_ref, _ = np.histogram(al_out3[:, :, 2], 256)  # !!!!!!!!!!!!!!!!!!!!!!!!!
    cdf_img = np.cumsum(hist_img)  # get the accumulative histogram
    cdf_ref = np.cumsum(hist_ref)

    for j in range(256):
        tmp = abs(cdf_img[j] - cdf_ref)
        tmp = tmp.tolist()
        idx = tmp.index(min(tmp))  # find the smallest number in tmp, get the index of this number
        out[:, :, 2][img[:, :, 2] == j] = idx  # !!!!!!!!!!!!!!!!!!!!!!!!!

    plt.figure(dpi=100, figsize=(25, 25))
    # 原始图像b通道的图片和直方图
    plt.subplot(2, 3, 1)
    plt.title("original reflectance img-B channel")
    plt.imshow(img[:, :, 2])  # !!!!!!!!!!!!!!!!!!!!!!!!!

    hist_img, bins = np.histogram(img[:, :, 2], 256)  # !!!!!!!!!!!!!!!!!!!!!!!!!
    plt.subplot(2, 3, 4, yticks=[])
    plt.title("original reflectance histogram-B channel")
    plt.bar(bins[:-1], hist_img)
    plt.xlabel("pixel value")
    plt.ylabel("number of pixels")

    # ref图像b通道的图片和直方图
    plt.subplot(2, 3, 2)
    plt.title("reference reflectance img-B channel")
    plt.imshow(al_out3[:, :, 2])  # !!!!!!!!!!!!!!!!!!!!!!!!!

    hist_ref, bins = np.histogram(al_out3[:, :, 2], 256)  # !!!!!!!!!!!!!!!!!!!!!!!!!
    plt.subplot(2, 3, 5, yticks=[])
    plt.title("reference reflectance histogram-B channel")
    plt.bar(bins[:-1], hist_ref)
    plt.xlabel("pixel value")
    plt.ylabel("number of pixels")

    # b通道匹配完之后的图片和直方图
    plt.subplot(2, 3, 3)
    plt.title("out img-B channel")
    plt.imshow(out[:, :, 2])  # !!!!!!!!!!!!!!!!!!!!!!!!!

    hist_out, bins = np.histogram(out[:, :, 2], 256)  # !!!!!!!!!!!!!!!!!!!!!!!!!
    plt.subplot(2, 3, 6, yticks=[])
    plt.title("matching output histogram-B channel")
    plt.bar(bins[:-1], hist_out)
    plt.xlabel("pixel value")
    plt.ylabel("number of pixels")

    plt.show()


def drawHis2():
    normal, _, lighting, img, _, mask = _decomposition(
        "D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/Images/11.png_face.png")

    ref = "D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/Images/4.png_face.png"
    n_out2, al_out2, light_out, al_out3, n_out3, mask2 = _decomposition(ref)

    al_out3 = convert(al_out3)
    img = convert(img)
    out = np.zeros_like(img)
    _, _, colorChannel = img.shape
    for i in range(colorChannel):  # RGB三个通道轮流来一遍
        hist_img, _ = np.histogram(img[:, :, i], 256)  # get the histogram
        hist_ref, _ = np.histogram(al_out3[:, :, i], 256)
        cdf_img = np.cumsum(hist_img)  # get the accumulative histogram
        # print(cdf_img)
        cdf_ref = np.cumsum(hist_ref)

        for j in range(256):
            tmp = abs(cdf_img[j] - cdf_ref)
            tmp = tmp.tolist()
            idx = tmp.index(min(tmp))  # find the smallest number in tmp, get the index of this number
            out[:, :, i][img[:, :, i] == j] = idx

    out = np.float32(out) / 255.0
    Irec, Ishd = create_shading_recon(normal, out, lighting)
    matching = convert(Irec)

    # img = cv2.imread("D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/Images/11.png_face.png")
    # ref = cv2.imread("D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/Images/4.png_face.png")

    plt.figure(dpi=100, figsize=(25, 25))
    # 原始图像和直方图
    plt.subplot(2, 3, 1)
    plt.title("original img")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    histImg, bins = np.histogram(img.flatten(), 256)
    plt.subplot(2, 3, 4, yticks=[])
    plt.title("original histogram")
    plt.bar(bins[:-1], histImg)
    plt.xlabel("pixel value")
    plt.ylabel("number of pixels")

    # ref图像和直方图
    plt.subplot(2, 3, 2)
    plt.title("reference img")
    plt.imshow(cv2.cvtColor(al_out3, cv2.COLOR_BGR2RGB))

    histRef, bins = np.histogram(al_out3.flatten(), 256)
    plt.subplot(2, 3, 5, yticks=[])
    plt.title("reference histogram")
    plt.bar(bins[:-1], histRef)
    plt.xlabel("pixel value")
    plt.ylabel("number of pixels")

    # 匹配完之后的图片和直方图
    plt.subplot(2, 3, 3)
    plt.title("matching output img")
    plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

    out = convert(out)
    histOut, bins = np.histogram(out.flatten(), 256)
    plt.subplot(2, 3, 6, yticks=[])
    plt.title("matching output histogram")
    plt.bar(bins[:-1], histOut)
    plt.xlabel("pixel value")
    plt.ylabel("number of pixels")

    plt.show()


if __name__ == '__main__':
    # drawHis()
    drawHis2()
