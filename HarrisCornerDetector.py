
import cv2 as cv
import numpy as np


def harris(img, threshold, window):
    height = len(img)
    width = len(img[0])
    window_size = 3

    corners = np.zeros((height, width))

    # Gradient intensity in x and y directions
    i_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
    i_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
    #print(i_x)
    ixx = i_x ** 2
    iyy = i_y ** 2

    # Calculated approximate change in intensity
    offset = window_size // 2
    for x in range(offset, height - offset):
        for y in range(offset, width - offset):
            sxx = np.sum(ixx[x - offset:x + 1 + offset, y - offset:y + 1 + offset])
            # print("RESULT:")
            # print(ixx[x - offset:x + 1 + offset, y - offset:y + 1 + offset])
            syy = np.sum(iyy[x - offset:x + 1 + offset, y - offset:y + 1 + offset])

            # R - score to which windows is likely to contain a corner
            r = min(sxx, syy)

            # if score is larger than threshold - it contains a corner
            if r > threshold:
                corners[x][y] = r
                #print(r)

    # Non local maxima suppression
    out_features = []

    d_x = window
    d_y = window

    x = 0
    while x < height - d_x:
        y = 0
        while y < width - d_y:
            window = corners[x:x + d_x, y:y + d_y]
            if window.size == 0:
                continue

            local_max = window.max()
            max_coord = np.unravel_index(np.argmax(window, axis=None), window.shape)

            # suppress everything
            corners[x:x + d_x, y:y + d_y] = 0

            # reset only the max
            if local_max > 0:
                #print(local_max)
                max_x = max_coord[0] + x
                max_y = max_coord[1] + y
                corners[max_x, max_y] = local_max
                out_features.append((max_x, max_y))
                print("max coord is ", (max_x, max_y))

            y += d_y
        x += d_x

    return out_features


def draw_img(color_img, features):
    out = color_img.copy()
    for corners in features:
        y, x = corners
        cv.circle(out, (x, y), 4, [0, 43, 255], -1)
    return out


def draw_circles(color_img, features):
    out = color_img.copy()
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i,j] = [0,0,0]
    #out = np.zeros(shape, np.uint8)
    for corners in features:
        y, x = corners
        # Circling the corners in red
        cv.circle(out, (x, y), 4, [0, 43, 255], -1)
    return out


if __name__ == '__main__':
    img = "images/Squares.jpg"
    img = "images/doska.jpg"
    #img = "lion.jpg"

    threshold = 300000  # chess

    window = 10

    g_im = cv.imread(img, cv.IMREAD_GRAYSCALE)
    c_im = cv.imread(img, cv.IMREAD_COLOR)

    list_features = harris(g_im, threshold, window)
    out_im = draw_img(c_im, list_features)
    circles = draw_circles(c_im, list_features)

    cv.imshow("Original", c_im)
    cv.imshow("Features on img", out_im)
    cv.imshow("Features", circles)

    cv.waitKey(0)
    cv.destroyAllWindows()
