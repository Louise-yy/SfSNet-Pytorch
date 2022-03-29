import os

import cv2
import numpy as np
import copy
import random
from PIL import ImageStat
from skimage import data, exposure, img_as_float, filters

from config import PROJECT_DIR
from src.functions import lambertian_attenuation, normal_harmonics, create_shading_recon
from SfSNet_test import _decomposition
from src.utils import convert

if __name__ == '__main__':
    pass


def change_albedo():
    albedo = cv2.imread('data/Albedo.png', cv2.IMREAD_UNCHANGED)
    h, w = albedo.shape[0:2]
    neww = 300
    newh = int(neww * (h / w))
    al_out2 = cv2.resize(albedo, (neww, newh))
    cv2.imshow("Albedo", al_out2)

    rows, cols, channel = al_out2.shape
    dst = al_out2.copy()
    a = 1.25
    b = 5
    for i in range(rows):
        for j in range(cols):
            for c in range(3):
                # print(al_out2[i, j][c])
                color = al_out2[i, j][c] * a + b
                if color > 255:
                    dst[i, j][c] = 255
                elif color < 0:
                    dst[i, j][c] = 0
    cv2.imshow("Albedo change", dst)
    cv2.waitKey(0)
    return dst


def albedo_highlight(al_out3, n_out2, light_out, weight, gamma):  # 高光/对比度
    # n_out2, al_out2, light_out, al_out3, n_out3 = _decomposition(img_path)
    albedo = convert(al_out3)
    # cv2.imshow("alout3", al_out3)  # gai
    # cv2.imshow("Albedo", albedo)  # gai

    c = weight  # 1.25
    b = gamma  # 1
    h, w, ch = albedo.shape  # 初始化一张黑图
    blank = np.zeros([h, w, ch], albedo.dtype)
    # 图像混合，c, 1-c为这两张图片的权重
    dst = cv2.addWeighted(albedo, c, blank, 1 - c, b)

    # cv2.imshow("Albedo change", dst)  # gai
    dst = np.float32(dst) / 255.0
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    Irec, Ishd = create_shading_recon(n_out2, dst, light_out)
    Irec = cv2.cvtColor(Irec, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(PROJECT_DIR, 'data/highlight.png'), convert(Irec))
    # cv2.imshow("Irec", Irec)  # gai
    # cv2.waitKey(0)  # gai
    # return dst


def albedo_bilateral(al_out3, n_out2, light_out, sigmaColor):
    # sigmaColor：Sigma_color较大，则在邻域中的像素值相差较大的像素点也会用来平均。
    # sigmaSpace：Sigma_space较大，则虽然离得较远，但是，只要值相近，就会互相影响
    al_out3 = convert(al_out3)
    bilateral_filter_img = cv2.bilateralFilter(al_out3, 9, sigmaColor, 40)  # 9 75 75
    # cv2.imshow("bilateral_filter_img", bilateral_filter_img)

    bilateral_filter_img = np.float32(bilateral_filter_img) / 255.0
    bilateral_filter_img = cv2.cvtColor(bilateral_filter_img, cv2.COLOR_BGR2RGB)
    Irec, Ishd = create_shading_recon(n_out2, bilateral_filter_img, light_out)
    Irec = cv2.cvtColor(Irec, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(PROJECT_DIR, 'data/buffing.png'), convert(Irec))

    # cv2.imshow("Albedo", al_out3)
    # cv2.imshow("Recon", Irec)
    # cv2.waitKey(0)


def albedo_sharp(al_out3, n_out2, light_out):
    al_out3 = convert(al_out3)
    dst = cv2.Laplacian(al_out3, -2)
    # dst = cv2.addWeighted(al_out3, 1, blank, 1 - c, b)
    # dst2 = cv2.add(al_out3, dst)
    median = al_out3 - dst
    median = cv2.medianBlur(median, 3)
    median = np.float32(median) / 255.0
    median = cv2.cvtColor(median, cv2.COLOR_BGR2RGB)
    Irec, Ishd = create_shading_recon(n_out2, median, light_out)
    Irec = cv2.cvtColor(Irec, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(PROJECT_DIR, 'data/sharpening.png'), convert(Irec))

    # cv2.imshow("al_out3", al_out3)
    # cv2.imshow("dst", Irec)
    # cv2.waitKey(0)


def histogram_matching(img, ref):
    # img = cv2.imread("D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/Images/4.png_face.png")
    # ref = cv2.imread("D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/Images/11.png_face.png")

    out = np.zeros_like(img)
    _, _, colorChannel = img.shape
    for i in range(colorChannel):  # RGB三个通道轮流来一遍
        # print(i)
        hist_img, _ = np.histogram(img[:, :, i], 256)  # get the histogram
        hist_ref, _ = np.histogram(ref[:, :, i], 256)
        cdf_img = np.cumsum(hist_img)  # get the accumulative histogram
        cdf_ref = np.cumsum(hist_ref)

        for j in range(256):
            tmp = abs(cdf_img[j] - cdf_ref)
            tmp = tmp.tolist()
            idx = tmp.index(min(tmp))  # find the smallest number in tmp, get the index of this number
            out[:, :, i][img[:, :, i] == j] = idx

    cv2.imshow('img', img)
    cv2.imshow('ref', ref)
    cv2.imshow('out', out)
    cv2.waitKey(0)


def albedo_mean(img_path):
    img = cv2.imread(img_path)
    input_image_cp = np.copy(img)  # 输入图像的副本
    filter_template = np.ones((3, 3))  # 空间滤波器模板
    pad_num = int((3 - 1) / 2)  # 输入图像需要填充的尺寸
    input_image_cp = np.pad(input_image_cp, (pad_num, pad_num), mode="constant", constant_values=0)  # 填充输入图像
    m, n = input_image_cp.shape  # 获取填充后的输入图像的大小
    output_image = np.copy(input_image_cp)  # 输出图像
    # 空间滤波

    for i in range(pad_num, m - pad_num):
        for j in range(pad_num, n - pad_num):
            output_image[i, j] = np.sum(
                filter_template * input_image_cp[i - pad_num:i + pad_num + 1, j - pad_num:j + pad_num + 1]) / (3 ** 2)
    output_image = output_image[pad_num:m - pad_num, pad_num:n - pad_num]  # 裁剪
    cv2.imshow("m", output_image)
    cv2.waitKey(0)


def synthetic():
    shading = cv2.imread('data/shading.png', cv2.IMREAD_UNCHANGED)
    albedo = cv2.imread('data/Albedo.png', cv2.IMREAD_UNCHANGED)
    sh = cv2.cvtColor(shading, cv2.COLOR_GRAY2RGB)
    M = albedo.shape[0]
    albedo = np.reshape(albedo, (M, M, 3))
    sh = np.reshape(sh, (M, M, 3))

    cv2.imshow("albedo", albedo)
    cv2.imshow("sh", sh)

    Re = albedo * sh
    # h, w = Re.shape[0:2]
    # neww = 300
    # newh = int(neww * (h / w))
    # Re = cv2.resize(Re, (neww, newh))
    cv2.imshow("Re", Re)
    cv2.waitKey(0)


def synthetic2():
    # shading = cv2.imread('data/shading.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
    # M = shading.shape[0]
    # shading = np.reshape(shading, (M, M, 3))
    # shading = cv2.cvtColor(shading, cv2.COLOR_BGR2RGB)
    albedo = cv2.imread('data/Albedo.png', cv2.IMREAD_UNCHANGED)
    cv2.imshow("albedo", albedo)
    albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)
    n_out2, al_out2, light_out, al_out3, n_out3 = _decomposition()

    # 用al_out2
    Irec, Ishd = create_shading_recon(n_out2, al_out2, light_out)
    Irec = cv2.cvtColor(Irec, cv2.COLOR_RGB2BGR)

    # 用albedo
    Re, Red = create_shading_recon(n_out2, albedo, light_out)
    Re = cv2.cvtColor(Re, cv2.COLOR_RGB2BGR)
    # Re = cv2.multiply(shading, albedo)

    # M = shading.shape[0]
    # Re = np.reshape(shading, (M, M, 3)) * np.reshape(albedo, (M, M, 3))

    h, w = al_out2.shape[0:2]
    neww = 300
    newh = int(neww * (h / w))
    albedo = cv2.resize(albedo, (neww, newh))
    cv2.imshow("albedo", albedo)
    al_out2 = cv2.resize(al_out2, (neww, newh))
    cv2.imshow("al_out2", al_out2)
    al_out3 = cv2.resize(al_out3, (neww, newh))
    cv2.imshow("al_out3", al_out3)
    Irec = cv2.resize(Irec, (neww, newh))
    cv2.imshow("Irec", Irec)
    Re = cv2.resize(Re, (neww, newh))
    cv2.imshow("Re", Re)

    cv2.waitKey(0)


if __name__ == '__main__':
    # n_out2, al_out2, light_out, al_out3, n_out3 = _decomposition(
    #     "D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/Images/11.png_face.png")
    img = cv2.imread("D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/Images/11.png_face.png")
    ref = cv2.imread("D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/Images/4.png_face.png")
    # change_albedo()
    # albedo_highlight("D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/Images/4.png_face.png", 1.25, 1)
    # albedo_bilateral(al_out3, n_out2, light_out, 40)
    # albedo_bilateral(img, n_out2, light_out, 40)
    # albedo_sharp(al_out3, n_out2, light_out)
    # albedo_mean("D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/Images/4.png_face.png")
    # synthetic()
    # synthetic2()

    histogram_matching(ref, img)

