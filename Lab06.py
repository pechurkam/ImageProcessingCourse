import cv2
import numpy as np
path = "images/contour.tif"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
def contour(img):
    res_img = np.copy(img)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i - 1, j] and img[i, j - 1] and img[i + 1, j] and img[i, j + 1]:
                res_img[i, j] = 255
            else:
                res_img[i, j] = 0

    res2_img = np.copy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            res2_img.itemset((i, j), (img.item(i, j) - res_img.item(i, j)))
    return res2_img


def contourE(img):
    res_img = np.copy(img)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            #якщо якесь число, то білий
            if img[i - 1, j] and img[i, j - 1] and img[i + 1, j] and img[i, j + 1]:
                res_img[i, j] = 255
            else:
                res_img[i, j] = 0
    return res_img

cv2.imshow('CONTOURe', contourE(img))
cv2.imshow('CONTOUR', contour(img))
cv2.imshow('ORIGINAL', img)
cv2.waitKey(0)
cv2.destroyAllWindows()