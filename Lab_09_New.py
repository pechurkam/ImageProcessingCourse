import cv2
import numpy as np

#path = "love.jpg"
path = "images/circle.jpg"

#path = "contour.tif"

#path = "triangle.png"

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#пряма трикутник прямокутник коло фантазійна фігура цифри або графік

def first_pixel(image):
    for x in range(image.shape[0] - 1):
        for y in range(image.shape[1] - 1):
            if image[x, y] == 255:
                first_pixel_p0 = (x, y)
                return first_pixel_p0


def second_pixel(image, x, y):
    dir, xx, yy = find_dir(5, image, x, y)
    return dir, xx, yy


def border_tracing(image):
    res_image = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    x, y = first_pixel(image)
    first_pixel_p0 = x, y
    dir, px, py = second_pixel(image, x, y)
    second_pixel_p1 = (px, py)
    pn_prev = (0, 0)
    pn = second_pixel_p1  # starting pixel of the region
    xx = px
    yy = py

    while not (first_pixel_p0 == pn_prev and second_pixel_p1 == pn):
        if dir % 2 == 0:
            dir = (dir + 7) % 8
        else:
            dir = (dir + 6) % 8
        dir, xx, yy = find_dir(dir, image, xx, yy)
        pn_prev = pn
        pn = xx, yy
        res_image[xx, yy] = 255

    return res_image


def find_dir(direction, image, x, y):
    cases = {
            0: (x + 1, y),
            1: (x + 1, y + 1),
            2: (x, y + 1),
            3: (x - 1, y + 1),
            4: (x - 1, y),
            5: (x - 1, y - 1),
            6: (x, y - 1),
            7: (x + 1, y - 1)
        }

    for i in range(8):
        if direction > 7:
            direction = 0
        (x1, y1) = cases[direction]
        if image[x1, y1] == 255:
            return direction, x1, y1
        else:
            direction += 1
    return direction, x1, y1


def border_list(image):
    list = []
    x, y = first_pixel(image)
    first_pixel_p0 = x, y
    dir, px, py = second_pixel(image, x, y)
    second_pixel_p1 = (px, py)
    pn_prev = (0, 0)
    pn = second_pixel_p1  # starting pixel of the region
    xx = px
    yy = py

    while not (first_pixel_p0 == pn_prev and second_pixel_p1 == pn):
        if dir % 2 == 0:
            dir = (dir + 7) % 8
        else:
            dir = (dir + 6) % 8
        dir, xx, yy = find_dir(dir, image, xx, yy)
        pn_prev = pn
        pn = xx, yy
        list.append((xx, yy))

    return list

cv2.imshow('Original', img)
cv2.imshow('Next', border_tracing(img))
cv2.waitKey(0)
cv2.destroyAllWindows()
