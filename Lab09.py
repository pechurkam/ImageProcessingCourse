import cv2
import numpy as np

path = "images/love.jpg"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def dictior(dir, i, j):
    dict = {
        0: (i, j + 1),
        1: (i + 1, j + 1),
        2: (i + 1, j),
        3: (i + 1, j - 1),
        4: (i, j - 1),
        5: (i - 1, j - 1),
        6: (i - 1, j),
        7: (i - 1, j + 1)
    }
    #print(dict[dir])
    return dir, dict[dir]


def firstPixel(img):
    h = img.shape[0]
    w = img.shape[1]
    for i in range(h):
        for j in range(w):
            if img[i][j] == 255:
                first_x, first_y = i, j
                print(first_x, first_y)
                return first_x, first_y


def secondPixel(firstPixel):
    print(firstPixel[0] - 1, firstPixel[1] + 1)
    return firstPixel[0] - 1, firstPixel[1] + 1


def tracing(img):

    res_img = np.copy(img)
    h = res_img.shape[0]
    w = res_img.shape[1]
    for i in range(h):
        for j in range(w):
            res_img[i][j] = 0

    dir = 7

    first = firstPixel(img)
    second = secondPixel(firstPixel(img))
    curr = second
    prev = (0, 0)
    x = second[0]
    y = second[1]
    while not (prev == first and curr == second):

        if dir % 2 == 0:
            dir = (dir + 7) % 8
        elif dir % 2 == 1:
            dir = (dir + 6) % 8

        for c in range(8):

            if dir > 7:
                dir = 0

            dir, (x, y) = dictior(dir, x, y)

            #print((x, y))
            if img[x][y] == 255:
                res_img[x][y] = 255
                prev = curr
                curr = x, y
                break
            else:
                dir += 1

    return res_img


#secondPixel(firstPixel(img))
print(dictior(5, 2, 3)[0])
print(dictior(5, 2, 3)[1])
cv2.imshow('New', tracing(img))
cv2.waitKey(0)
cv2.destroyAllWindows()
