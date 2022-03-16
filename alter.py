import cv2
import numpy as np
import copy
import random
from PIL import ImageStat
from skimage import data, exposure, img_as_float

from src.functions import lambertian_attenuation, normal_harmonics, create_shading_recon
from SfSNet_test import _decomposition,_test
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


def albedo_highlight(img_path):  # 高光/对比度
    # albedo = cv2.imread('D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/data/Albedo.png', cv2.IMREAD_UNCHANGED)
    # img = cv2.imread(img_add, cv2.IMREAD_UNCHANGED)
    n_out2, al_out2, light_out, al_out3, n_out3 = _test(img_path)
    albedo = convert(al_out3)
    # h, w = albedo.shape[0:2]
    # neww = 300
    # newh = int(neww * (h / w))
    # al_out2 = cv2.resize(albedo, (neww, newh))
    # cv2.imshow("Albedo", albedo)

    c = 1.25  # 1.2
    b = 1  # 100
    h, w, ch = albedo.shape  # 初始化一张黑图
    blank = np.zeros([h, w, ch], albedo.dtype)
    # 图像混合，c, 1-c为这两张图片的权重
    dst = cv2.addWeighted(albedo, c, blank, 1 - c, b)
    # dst = cv2.resize(dst, (neww, newh))
    cv2.imwrite('data/highlight.png', dst)
    # cv2.imshow("Albedo change", dst)
    # cv2.waitKey(0)
    return dst


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
    shading = cv2.imread('data/shading.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
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
    # change_albedo()
    albedo_highlight("D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/Images/4.png_face.png")
    # synthetic()
    # synthetic2()
