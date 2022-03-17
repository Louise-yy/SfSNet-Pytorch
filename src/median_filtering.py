import cv2 as cv
import numpy as np
import math
import copy


def spilt(a):
    if a / 2 == 0:
        x1 = x2 = a / 2
    else:
        x1 = math.floor(a / 2)
        x2 = a - x1
    return -x1, x2


def original(i, j, k, a, b, img):  # 取框内的值
    x1, x2 = spilt(a)
    y1, y2 = spilt(b)
    temp = np.zeros(a * b)
    count = 0
    for m in range(x1, x2):  # -1,0,1
        for n in range(y1, y2):  # img.shape[0]图片的高度 img.shape[1]图片的宽度
            if i + m < 0 or i + m > img.shape[0] - 1 or j + n < 0 or j + n > img.shape[1] - 1:  # 在图像的四条边界上 上i=0 下i=127 左j=0 右j=127
                t = img[i, j, k]
                temp[count] = img[i, j, k]
            else:
                t = img[i, j, k]
                temp[count] = img[i + m, j + n, k]
            count += 1

    return temp


def average_function(a, b, img):
    img0 = copy.copy(img)
    for i in range(0, img.shape[0]):
        for j in range(2, img.shape[1]):
            for k in range(img.shape[2]):
                temp = original(i, j, k, a, b, img0)
                max = np.max(temp)
                min = np.min(temp)
                if img[i, j, k] == max or img[i, j, k] == min:
                    img[i, j, k] = np.median(temp)
                else:
                    img[i, j, k] = img[i, j, k]
    return img


def main():
    img0 = cv.imread("D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/Images/11.png_face.png")
    ave_img = average_function(3, 3, copy.copy(img0))  # (3，3)滤波器大小
    cv.imshow("ave_img", ave_img)
    cv.imshow("original", img0)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
