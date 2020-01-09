import cv2
import numpy as np

path = "images/kit.jpg"
#path = "animal.jpg"
#path = "IM15.tif"

im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
print(im[0, 0])
def bigger(img, num):
    h = img.shape[0]
    w = img.shape[1]
    h2 = h * num
    w2 = w * num
    # return a new array of given shape and type, filled with zeros
    res_img = np.zeros((h2, w2), np.uint8)
    for i in range(h2):
        for j in range(w2):
            res_img[i, j] = img[i // num, j // num]
    return res_img

def smaller(img, num):
    h = img.shape[0]
    w = img.shape[1]
    h2 = h // num
    w2 = w // num
    if h2 != w2:
        w2 = h2
    res_img = np.zeros((h2, w2, 3), np.uint8)
    for j in range(h2):
        for i in range(w2):
            res_img[i, j] = average(img, num, i * num, j * num)
    return res_img

def average(img, num, i, j):
    average_var = [0, 0, 0]
    for k in range(num):
        for g in range(num):
            if i + k >= img.shape[0] or j + g >= img.shape[1]:
                break
            average_var += img[i+k, j+g]
    return average_var // (num * num)

def color_threshold(img):
    h = img.shape[0]
    w = img.shape[1]
    res_im = np.zeros((h, w, 3), np.uint8)
    for i in range(h):
        for j in range(w):
            if img[i, j] > 150:
                res_im[i, j] = 255
            else:
                res_im[i, j] = 0
    return res_im


cv2.imshow("Origin", im)
cv2.imshow("Bigger", bigger(im, 2))
cv2.imshow("Smaller", smaller(im, 2))
cv2.imshow("Threshold", color_threshold(im))

#cv2.moveWindow("Origin", 100, 100)
#cv2.moveWindow("Bigger", 150, 150)

cv2.waitKey(0)
cv2.destroyAllWindows()

