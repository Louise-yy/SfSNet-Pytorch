import os

import cv2
import numpy as np

from config import PROJECT_DIR
from src.functions import create_shading_recon
from SfSNet_test import _decomposition
from src.utils import convert

# if __name__ == '__main__':
#     pass


def albedo_highlight(al_out3, n_out2, light_out, mask, weight, gamma):
    """
    @brief Adding highlight to the input image

    @param al_out3 albedo of the input image
    @param n_out2 normal of the input image
    @param light_out lighting of the input image
    @param mask mask of the input image
    @param weight Weighting of albedo, which is an argument to the addWeighted function
    @param gamma The value added to the image blend, which is also an argument to the addWeighted function

    @return dst Processed albedo
    """
    albedo = convert(al_out3)
    c = weight
    b = gamma
    h, w, ch = albedo.shape
    blank = np.zeros([h, w, ch], albedo.dtype)
    # Image blending, c, 1-c are the weights of the two images
    dst = cv2.addWeighted(albedo, c, blank, 1 - c, b)

    dst = np.float32(dst) / 255.0
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    Irec, Ishd = create_shading_recon(n_out2, dst, light_out)
    highlight = convert(cv2.cvtColor(Irec, cv2.COLOR_RGB2BGR))
    diff = (mask // 255)
    highlight = highlight * diff
    cv2.imwrite(os.path.join(PROJECT_DIR, 'data/highlight.png'), highlight)
    return cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)

    # cv2.imshow("Irec", highlight)
    # cv2.waitKey(0)


def albedo_bilateral(al_out3, n_out2, light_out, mask, sigma):
    """
    @brief Adding buffing to the input image

    @param al_out3 albedo of the input image
    @param n_out2 normal of the input image
    @param light_out lighting of the input image
    @param mask mask of the input image
    @param sigmaColor The sigma parameter in coordinate space, which is an argument to the bilateralFilter function

    @return bilateral_filter_img Processed albedo
    """
    al_out3 = convert(al_out3)
    bilateral_filter_img = cv2.bilateralFilter(al_out3, 7, sigma, sigma)  # 9 75 75

    bilateral_filter_img = np.float32(bilateral_filter_img) / 255.0
    bilateral_filter_img = cv2.cvtColor(bilateral_filter_img, cv2.COLOR_BGR2RGB)
    Irec, Ishd = create_shading_recon(n_out2, bilateral_filter_img, light_out)
    buffing = convert(cv2.cvtColor(Irec, cv2.COLOR_RGB2BGR))
    diff = (mask // 255)
    buffing = buffing * diff
    cv2.imwrite(os.path.join(PROJECT_DIR, 'data/buffing.png'), buffing)
    return cv2.cvtColor(bilateral_filter_img, cv2.COLOR_RGB2BGR)

    # cv2.imshow("Recon", buffing)
    # cv2.waitKey(0)


def unsharp_masking(al_out3, amount, n_out2, light_out, mask):
    """
    @brief Adding buffing to the input image

    @param al_out3 albedo of the input image
    @param n_out2 normal of the input image
    @param light_out lighting of the input image
    @param mask mask of the input image
    @param amount Scaling factor for high-frequency section

    @return dst Processed albedo
    """
    blur = cv2.GaussianBlur(al_out3, (5, 5), 5)
    diff = al_out3 - blur
    dst = al_out3 + amount * diff

    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    Irec, Ishd = create_shading_recon(n_out2, dst, light_out)
    sharpening = convert(cv2.cvtColor(Irec, cv2.COLOR_RGB2BGR))
    diff2 = (mask // 255)
    sharpening = sharpening * diff2
    cv2.imwrite(os.path.join(PROJECT_DIR, 'data/sharpening.png'), sharpening)
    return cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)

    # cv2.imshow("Irec", sharpening)
    # cv2.waitKey(0)


def histogram_matching(img, normal, lighting, ref, mask):
    """
    @brief Change the albedo of the original image according to the albedo of the reference image

    @param img albedo of the input image
    @param normal normal of the input image
    @param lighting lighting of the input image
    @param ref Address of the reference image
    @param mask mask of the input image

    @return out Processed albedo
    """
    n_out2, al_out2, light_out, al_out3, n_out3, mask2 = _decomposition(ref)

    al_out3 = convert(al_out3)
    img = convert(img)
    out = np.zeros_like(img)
    _, _, colorChannel = img.shape
    for i in range(colorChannel):
        hist_img, _ = np.histogram(img[:, :, i], 256)  # get the histogram
        hist_ref, _ = np.histogram(al_out3[:, :, i], 256)
        cdf_img = np.cumsum(hist_img)  # get the accumulative histogram
        cdf_ref = np.cumsum(hist_ref)
        for j in range(256):
            tmp = abs(cdf_img[j] - cdf_ref)
            tmp = tmp.tolist()
            idx = tmp.index(min(tmp))  # find the smallest number in tmp, get the index of this number
            out[:, :, i][img[:, :, i] == j] = idx

    out = np.float32(out) / 255.0
    Irec, Ishd = create_shading_recon(normal, out, lighting)
    matching = convert(Irec)
    diff = (mask // 255)
    matching = matching * diff
    cv2.imwrite(os.path.join(PROJECT_DIR, 'data/matching.png'), matching)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    # cv2.imshow('ref-albedo', al_out3)
    # cv2.imshow('albedo after matching', out)
    # cv2.imshow('Irec', matching)
    # cv2.waitKey(0)


def shading_alter(ref, normal, albedo, mask):
    """
    @brief Relighting

    @param ref Address of the reference imgage
    @param normal normal of the input image
    @param albedo albedo of the input image
    @param mask mask of the input image

    @return light_out Reference lighting
    """
    n_out2, al_out2, light_out, al_out3, n_out3, mask2 = _decomposition(ref)
    Irec, Ishd = create_shading_recon(normal, albedo, light_out)
    f2f = convert(Irec)
    diff = (mask // 255)
    f2f = f2f * diff
    cv2.imwrite(os.path.join(PROJECT_DIR, 'data/f2f.png'), f2f)
    return light_out

    # cv2.imshow('ref-light', light_out)
    # cv2.imshow('Irec', f2f)
    # cv2.waitKey(0)


# some previous function versions before improvement
# def change_albedo():  # equal to highlight function
#     albedo = cv2.imread('data/Albedo.png', cv2.IMREAD_UNCHANGED)
#     h, w = albedo.shape[0:2]
#     neww = 300
#     newh = int(neww * (h / w))
#     al_out2 = cv2.resize(albedo, (neww, newh))
#     cv2.imshow("Albedo", al_out2)
#
#     rows, cols, channel = al_out2.shape
#     dst = al_out2.copy()
#     a = 1.25
#     b = 5
#     for i in range(rows):
#         for j in range(cols):
#             for c in range(3):
#                 # print(al_out2[i, j][c])
#                 color = al_out2[i, j][c] * a + b
#                 if color > 255:
#                     dst[i, j][c] = 255
#                 elif color < 0:
#                     dst[i, j][c] = 0
#     cv2.imshow("Albedo change", dst)
#     cv2.waitKey(0)
#     return dst

# def albedo_sharp(al_out3, n_out2, light_out):  # no usages unsharp masking
#     al_out3 = convert(al_out3)
#     dst = cv2.Laplacian(al_out3, -2)
#     # dst = cv2.addWeighted(al_out3, 1, blank, 1 - c, b)
#     # dst2 = cv2.add(al_out3, dst)
#     median = al_out3 - dst
#     median = cv2.medianBlur(median, 3)
#     median = np.float32(median) / 255.0
#     # cv2.imshow("median", median)
#     median = cv2.cvtColor(median, cv2.COLOR_BGR2RGB)
#     Irec, Ishd = create_shading_recon(n_out2, median, light_out)
#     Irec = cv2.cvtColor(Irec, cv2.COLOR_RGB2BGR)
#     cv2.imwrite(os.path.join(PROJECT_DIR, 'data/sharpening.png'), convert(Irec))
#
#     # cv2.imshow("al_out3", al_out3)
#     # cv2.imshow("dst", dst)
#     cv2.imshow("re", Irec)
#     cv2.waitKey(0)


if __name__ == '__main__':
    n_out2, al_out2, light_out, al_out3, n_out3, mask = _decomposition(
        "D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/Images/11.png_face.png")
    img = cv2.imread("D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/Images/11.png_face.png")
    # img = "D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/Images/4.png_face.png"
    ref = "D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/Images/9.png_face.png"
    img_ref = cv2.imread("D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/Images/9.png_face.png")

    # albedo_highlight(al_out3, n_out2, light_out, mask, 1.3, 10)  # 1.25 1
    # albedo_bilateral(al_out3, n_out2, light_out, mask, 70)
    unsharp_masking(al_out3, 1, n_out2, light_out, mask)
    # histogram_matching(al_out3, n_out2, light_out, ref, mask)
    # shading_alter(ref, n_out2, al_out3, mask)
    cv2.imshow("img", img)
    # cv2.imshow("ref", img_ref)
    # cv2.imshow("albedo", al_out3)
    cv2.waitKey(0)

    # change_albedo()
    # albedo_sharp(al_out3, n_out2, light_out)
    # cv2.imshow("ori", img)
    # cv2.waitKey(0)
