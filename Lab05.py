import cv2
import numpy as np
import math

path = "images/giraffe.jpg"
img1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5)
#img.item
def transform(img):
    sigma = float(input('Your sigma: '))
    gauss_size = int(input('Gauss matrix size: '))
    Xm = np.zeros((gauss_size, gauss_size), np.float_)
    a = - (gauss_size//2)
    print(a)
    ran = gauss_size
    for i in range(ran):
        for j in range(ran):
            Xm.itemset((i, j), i + a)
    print('Xm:')
    print(Xm)

    Ym = np.zeros((gauss_size, gauss_size), np.float_)
    for i in range(gauss_size):
        for j in range(gauss_size):
            Ym.itemset((i, j), j + a)
    print('Ym:')
    print(Ym)
    gausM = np.zeros((gauss_size, gauss_size), np.float_)
    norma = 0
    for i in range(gauss_size):
        for j in range(gauss_size):
            gausM.itemset((i, j), (1 / (2 * math.pi * sigma * sigma)) * math.e ** (
                    - ((Xm[i][j] ** 2) + (Ym[i][j]) ** 2) / (2 * (sigma ** 2))))
            norma += gausM.item(i, j)
    print(gausM)
    print('NORMA: ' + str(norma))
    dif = gauss_size // 2
    print('Dif:')
    print(dif)
    res_img = np.copy(img)
    #h, w = np.shape(img)
    h = img.shape[0]
    w = img.shape[1]
    for i in range(h):
        for j in range(w):
            sum = 0
            for m in range(gauss_size):
                for n in range(gauss_size):

                    if i + m - dif < 0 or j + n - dif < 0 or i + m - dif >= h or j + n - dif >= w:
                        sum += img.item(i, j)*gausM.item(m, n)
                    else:
                        sum += img.item(i + m - dif, j + n - dif) * gausM.item(m, n)
            res_img.itemset((i, j), sum / norma)
    return res_img

def sobel(img):
    gr = np.zeros((img.shape[0], img.shape[1]), np.int16)
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    print(Gx)
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
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
                    if i + m - dif < 0 or j + n - dif < 0 or i + m - dif >= h or j + n - dif >= w:
                        sum += img.item(i, j) * Gx.item(m, n)
                        sum2 += img.item(i, j) * Gy.item(m, n)
                    else:
                        sum += img.item(i + m - dif, j + n - dif) * Gx.item(m, n)
                        sum2 += img.item(i + m - dif, j + n - dif) * Gy.item(m, n)
            res_img.itemset((i, j), math.sqrt(sum**2 + sum2**2) / 4)
            gr.itemset((i, j), np.arctan2(sum2, sum) * 180 / math.pi)
            #print(gr)
    return res_img, gr

def non_maximum_suppression(sobel_res, angle):
    height, width = sobel_res.shape[:2]
    blank = np.zeros(sobel_res.shape, dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if angle.item(i, j) < 0:
                #angle[i][j] += 360
                angle.itemset((i, j), angle.item(i, j)+360)
            if ((j + 1) < width) and ((j - 1) >= 0) and ((i + 1) < height) and ((i - 1) >= 0):
                # 0 degrees
                if (angle.item(i, j) >= 337.5 or angle.item(i, j) < 22.5) or (157.5 <= angle.item(i, j) < 202.5):
                    if sobel_res.item(i, j) >= sobel_res.item(i, j+1) and sobel_res.item(i, j) >= sobel_res.item(i, j - 1):
                        blank.itemset((i, j), sobel_res.item(i, j))
                # 45 degrees
                elif (22.5 <= angle.item(i, j) < 67.5) or (202.5 <= angle.item(i, j) < 247.5):
                    if sobel_res.item(i, j) >= sobel_res.item(i - 1, j + 1) and sobel_res.item(i, j) >= sobel_res.item(i + 1, j - 1):
                        blank.itemset((i, j), sobel_res.item(i, j))
                # 90 degrees
                elif (67.5 <= angle.item(i, j) < 112.5) or (247.5 <= angle.item(i, j) < 292.5):
                    if sobel_res.item(i, j) >= sobel_res.item(i - 1, j) and sobel_res.item(i, j) >= sobel_res.item(i + 1, j):
                        blank.itemset((i, j), sobel_res.item(i, j))
                # 135 degrees
                elif (112.5 <= angle.item(i, j) < 157.5) or (292.5 <= angle.item(i, j) < 337.5):
                    if sobel_res.item(i, j) >= sobel_res.item(i - 1, j - 1) and sobel_res.item(i, j) >= sobel_res.item(i + 1, j + 1):
                        blank.itemset((i, j), sobel_res.item(i, j))
    return blank

def thresholding(im):
    thres = np.zeros(im.shape)
    strong = 1.0
    weak = 0.5
    mmax = np.max(im)
    lo, hi = 0.1 * mmax, 0.2 * mmax
    strongs = []
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            px = im.item(i, j)
            if px >= hi:
                thres.itemset((i, j), strong)
                strongs.append((i, j))
            elif px >= lo:
                thres.itemset((i, j), weak)

    res = np.zeros(im.shape, dtype=np.uint8)
    for st in strongs:
        res[st] = 255
        #print(st[0])
        for i in range(max(0, st[0] - 1), min(st[0] + 1, im.shape[0])):
            for j in range(max(0, st[1] - 1), min(st[1] + 1, im.shape[1])):
                if thres.item(i, j) == weak:
                    res.itemset((i, j), 255)
    return res


gausIm = transform(img)
sobel_im, gr = sobel(gausIm)
non_max_im = non_maximum_suppression(sobel_im, gr)
im_last = thresholding(non_max_im)
cv2.imshow('Image', im_last)
cv2.imshow('Original', img)
cv2.imshow('Gaus', gausIm)
cv2.imshow('Sobel', sobel_im)
cv2.imshow('NonMax', non_max_im)
cv2.waitKey(0)
cv2.destroyAllWindows()


