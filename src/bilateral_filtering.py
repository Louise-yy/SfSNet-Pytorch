import cv2 as cv
import numpy as np
import math
import copy


def spilt(a):
    if a/2 == 0:
        x1 = x2 = a/2
    else:
        x1 = math.floor(a/2)  # 向下取整
        x2 = a - x1
    return -x1, x2


def d_value():
    value = np.zeros(256)
    var_temp = 30
    for i in range(0, 255):
        t = i*i
        value[i] = math.e ** (-t / (2 * var_temp * var_temp))
    return value


def gaussian_b0x(a, b):
    judge = 10
    box = []
    x1, x2 = spilt(a)
    y1, y2 = spilt(b)
    for i in range(x1, x2):
        for j in range(y1, y2):
            t = i*i + j*j
            re = math.e ** (-t/(2*judge*judge))
            box.append(re)
    # for x in box :
    #     print (x)
    return box


def original(i, j, k, a, b, img):
    x1, x2 = spilt(a)
    y1, y2 = spilt(b)
    temp = np.zeros(a * b)
    count = 0
    for m in range(x1, x2):
        for n in range(y1, y2):
            if i + m < 0 or i + m > img.shape[0] - 1 or j + n < 0 or j + n > img.shape[1] - 1:
                temp[count] = img[i, j, k]
            else:
                temp[count] = img[i + m, j + n, k]
            count += 1
    return temp


def bilateral_function(a, b, img, gauss_fun, d_value_e):
    x1, x2 = spilt(a)
    y1, y2 = spilt(b)
    re = np.zeros(a * b)
    img0 = copy.copy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(0, 2):
                temp = original(i, j, k, a, b, img0)
                # print("ave:",ave_temp)
                count = 0
                for m in range(x1, x2):
                    for n in range(y1, y2):
                        if i+m < 0 or i+m > img.shape[0]-1 or j+n < 0 or j+n > img.shape[1]-1:
                            x = img[i, j, k]
                        else:
                            x = img[i+m, j+n, k]
                        t = int(math.fabs(int(x) - int(img[i, j, k])))
                        re[count] = d_value_e[t]
                        count += 1
                evalue = np.multiply(re, gauss_fun)
                img[i, j, k] = int(np.average(temp, weights=evalue))
    return img


def main():
    gauss_new = gaussian_b0x(6, 6)  # 30
    # print(gauss_new)
    d_value_e = d_value()
    img0 = cv.imread("D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/Images/11.png_face.png")
    bilateral_img = bilateral_function(6, 6, copy.copy(img0), gauss_new, d_value_e)  # 30
    cv.imshow("bilateral", bilateral_img)
    cv.imshow("original_img", img0)
    # cv.imwrite("shuangbian.jpg", bilateral_img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
