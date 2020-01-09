import cv2
import numpy as np
#path = "vegetables.jpg"

#path = "route66.jpg"
path = "images/town.jpg"
#path = "pesok.jpg"
img1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5)
#ОПТИМАЛЬНИЙ ПОРІГ

#array of amounts of pixels with spec brightness

def _histogram(img):
    height, width = img.shape[:2]
    hist = [0 for _ in range(256)]

    for h in range(height):
        for w in range(width):
            hist[img[h, w]] += 1

    return hist

def thres(hist):
    TkNew = 128
    Tk = 0
    while TkNew != Tk:
        sum1 = 0
        sum2 = 0
        for i in range(TkNew):
            sum1 += i*hist[i]
        for j in range(TkNew):
            sum2 += hist[j]
        m0 = sum1 / sum2

        sum11 = 0
        sum12 = 0
        for i2 in range(TkNew + 1, 255):
            sum11 += i2 * hist[i2]
        for j2 in range(TkNew + 1, 255):
            sum12 += hist[j2]
        m1 = sum11 / sum12
        Tk = TkNew
        TkNew = int((m0 + m1)/2)
        print(TkNew)
    return TkNew

def transform_img(img):
    res_img = np.copy(img)
    h = res_img.shape[0]
    w = res_img.shape[1]
    threshold = thres(_histogram(img))
    for i in range(h):
        for j in range(w):
            if res_img.item(i, j) <= threshold:
                res_img.itemset((i, j), 0)
            if res_img.item(i, j) > threshold:
                res_img.itemset((i, j), 255)
    return res_img

cv2.imshow('Original', img)
cv2.imshow('Transformed', transform_img(img))
cv2.waitKey(0)
cv2.destroyAllWindows()