import cv2
import numpy as np


def show_gray_img_hist(hist, window_title):
    """
    :param hist: 灰度图像的直方图，为一个256*1的 numpy.ndarray
    :return: none
    """
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(hist)
    hist_img = np.zeros([256, 256], np.uint8)
    for h in range(256):
        intensity = int(256 * hist[h] / max_val)
        cv2.line(hist_img, (h, 256), (h, 256 - intensity), [255, 0, 0])

    cv2.imshow(window_title, hist_img)


def get_acc_prob_hist(hist):
    acc_hist = np.zeros([256, 1], np.float32)
    pre_val = 0.
    for i in range(256):
        acc_hist[i, 0] = pre_val + hist[i, 0]
        pre_val = acc_hist[i, 0]

    acc_hist /= pre_val
    return acc_hist


def hist_specify(src_img, dst_img):
    """
    直方图规定化，把dst_img按照src_img的图像进行规定化
    :param src_img:
    :param dst_img:
    :return:
    """
    # 计算源图像和规定化之后图像的累计直方图
    src_hist = cv2.calcHist([src_img], [0], None, [256], [0.0, 255.])
    dst_hist = cv2.calcHist([dst_img], [0], None, [256], [0.0, 255.])
    src_acc_prob_hist = get_acc_prob_hist(src_hist)
    dst_acc_prob_hist = get_acc_prob_hist(dst_hist)

    # 计算源图像的各阶灰度到规定化之后图像各阶灰度的差值的绝对值，得到一个256*256的矩阵，第i行表示源图像的第i阶累计直方图到规定化后图像各
    # 阶灰度累计直方图的差值的绝对值，
    # diff_acc_prob = np.ndarray((256, 256), np.float32)
    # for i in range(256):
    #    for j in range(256):
    #        diff_acc_prob[i, j] = abs(src_acc_prob_hist[i] - dst_acc_prob_hist[j])
    diff_acc_prob = abs(np.tile(src_acc_prob_hist.reshape(256, 1), (1, 256)) - dst_acc_prob_hist.reshape(1, 256))

    # 求出各阶灰度对应的差值的最小值，该最小值对应的灰度阶即为映射之后的灰度阶
    table = np.argmin(diff_acc_prob, axis=0)
    table = table.astype(np.uint8)  # @注意 对于灰度图像cv2.LUT的table必须是uint8类型

    # 将源图像按照求出的映射关系做映射
    result = cv2.LUT(dst_img, table)

    # 显示各种图像
    show_gray_img_hist(src_hist, 'src_hist')
    show_gray_img_hist(dst_hist, 'dst_hist')
    cv2.imshow('src_img', src_img)
    cv2.imshow('dst_img', dst_img)
    cv2.imshow('result', result)

    result_hist = cv2.calcHist([result], [0], None, [256], [0.0, 255.])
    show_gray_img_hist(result_hist, 'result_hist')


if __name__ == '__main__':
    src_img = cv2.imread("D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/Images/1.png_face.png", 0)
    dst_img = cv2.imread("D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/Images/4.png_face.png", 0)

    hist_specify(src_img, dst_img)

    cv2.waitKey()