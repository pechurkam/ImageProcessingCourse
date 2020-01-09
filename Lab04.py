import cv2
import numpy as np
import math

path = "images/Vespa.jpg"
img1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5)

def sobelX(img):
    Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    print(Gx)
    Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    res_img = np.copy(img)
    h = img.shape[0]
    w = img.shape[1]
    dif = 1
    for i in range(h):
        for j in range(w):
            sum = 0
            sum2 = 0
            for m in range(3):
                for n in range(3):
                    if i + m - dif < 0 or j + n - dif < 0 or i + m - dif >= img.shape[0] or j + n - dif >= img.shape[1]:
                        sum += img[i][j] * Gx[m][n]
                       # sum2 += img[i][j] * Gy[m][n]
                    else:
                        sum += img[i + m - dif][j + n - dif] * Gx[m][n]
                       # sum2 += img[i + m - dif][j + n - dif] * Gy[m][n]

            res_img[i][j] = math.sqrt(sum**2 + sum2**2) / 4
            #gr = np.atan2(sum2, sum) * 180 / math.pi
            #print(gr)
    return res_img

def sobelY(img):
    Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    print(Gx)
    Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    res_img = np.copy(img)
    h = img.shape[0]
    w = img.shape[1]
    dif = 1
    for i in range(h):
        for j in range(w):
            sum = 0
            sum2 = 0
            for m in range(3):
                for n in range(3):
                    if i + m - dif < 0 or j + n - dif < 0 or i + m - dif >= img.shape[0] or j + n - dif >= img.shape[1]:
                        #sum += img[i][j] * Gx[m][n]
                        sum2 += img[i][j] * Gy[m][n]
                    else:
                        #sum += img[i + m - dif][j + n - dif] * Gx[m][n]
                        sum2 += img[i + m - dif][j + n - dif] * Gy[m][n]

            res_img[i][j] = math.sqrt(sum**2 + sum2**2) / 4
            #gr = np.atan2(sum2, sum) * 180 / math.pi
            #print(gr)
    return res_img
def sobel(img):
    Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    print(Gx)
    Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    res_img = np.copy(img)
    h = img.shape[0]
    w = img.shape[1]
    dif = 1
    for i in range(h):
        for j in range(w):
            sum = 0
            sum2 = 0
            for m in range(3):
                for n in range(3):
                    if i + m - dif < 0 or j + n - dif < 0 or i + m - dif >= img.shape[0] or j + n - dif >= img.shape[1]:
                        sum += img[i][j] * Gx[m][n]
                        sum2 += img[i][j] * Gy[m][n]
                    else:
                        sum += img[i + m - dif][j + n - dif] * Gx[m][n]
                        sum2 += img[i + m - dif][j + n - dif] * Gy[m][n]

            res_img[i][j] = math.sqrt(sum**2 + sum2**2) / 4
            #gr = np.atan2(sum2, sum) * 180 / math.pi
            #print(gr)
    return res_img

cv2.imshow('SOBEL_X', sobelX(img))
cv2.imshow('SOBEL_Y', sobelY(img))
cv2.imshow('SOBEL_XY', sobel(img))

cv2.imshow('ORIGINAL', img)
cv2.waitKey(0)
cv2.destroyAllWindows()