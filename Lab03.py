import cv2
import numpy as np
import math

#path = "Bikesgray.jpg"
path = "images/bird.jpg"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)



def transform(img):
    sigma = float(input('Your sigma: '))
    gauss_size = int(input('Gauss matrix size: '))
    Xm = np.zeros((gauss_size, gauss_size), np.float_)
    a = - (gauss_size//2)
    print(a)
    ran = gauss_size
    for i in range(ran):
        for j in range(ran):
            Xm[i][j] = i + a

    print('Xm:')
    print(Xm)

    Ym = np.zeros((gauss_size, gauss_size), np.float_)
    for i in range(gauss_size):
        for j in range(gauss_size):
            Ym[i][j] = j + a
    print('Ym:')
    print(Ym)


    gausM = np.zeros((gauss_size, gauss_size), np.float_)
    norma = 0
    for i in range(gauss_size):
        for j in range(gauss_size):
            gausM[i][j] = (1 / (2 * math.pi * sigma * sigma)) * math.e ** (
                    - ((Xm[i][j] ** 2) + (Ym[i][j]) ** 2) / (2 * (sigma ** 2)))
            norma += gausM[i][j]
    print(gausM)
    print('NORMA: ' + str(norma))
    dif = gauss_size // 2
    print('Dif:')
    print(dif)
    res_img = np.copy(img)
    h = img.shape[0]
    w = img.shape[1]
    for i in range(h):
        for j in range(w):
            sum = 0
            for m in range(gauss_size):
                for n in range(gauss_size):
                    #out of range
                    if i + m - dif < 0 or j + n - dif < 0 or i + m - dif >= img.shape[0] or j + n - dif >= img.shape[1]:
                        sum += img[i][j]*gausM[m][n]

                    else:
                        sum += img[i + m - dif][j + n - dif] * gausM[m][n]
            res_img[i][j] = sum / norma
    return res_img

def filter(img):
    v1 = [1, 2, 1]
    v2 = [1, 2, 1]
    matrix = np.zeros((3,3), np.float_)
    for i in range(3):
        for j in range(3):
            matrix[i][j] = v1[i]*v2[j]

    print(matrix)
    dif = 1
    res_img = np.copy(img)
    h = img.shape[0]
    w = img.shape[1]
    for i in range(h):
        for j in range(w):
            sum = 0
            for m in range(3):
                for n in range(3):
                    if i + m - dif < 0 or j + n - dif < 0 or i + m - dif >= img.shape[0] or j + n - dif >= img.shape[1]:
                        sum += img[i][j] * matrix[m][n]
                    else:
                        sum += img[i + m - dif][j + n - dif] * matrix[m][n]
            res_img[i][j] = sum / 16
    return res_img






#GAUSS
#cv2.imshow('GAUSS', transform(small))
cv2.imshow('FILTER', filter(small))

cv2.imshow('ORIGINAL', small)



#FILTER
#cv2.imshow('animal', img)
#cv2.imshow('FILTER', filter(img))

cv2.waitKey(0)
cv2.destroyAllWindows()
