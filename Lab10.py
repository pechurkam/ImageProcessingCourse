import numpy as np
import cv2
import Lab_09_New as lab09
from math import atan2
from matplotlib import pyplot as plt



def curvature(contour, k=1):
    n = len(contour)
    res = np.zeros(n)

    def p_i(i):
        cur = contour[i]
        next = contour[(i+k) % n]
        prev = contour[(i-k) % n]
        forward_v = cur[0] - next[0], cur[1] - next[1]
        backward_v = cur[0] - prev[0], cur[1] - prev[1]
        distance_f = vector_length(forward_v)
        distance_b = vector_length(backward_v)
        angle_f = atan2(abs(forward_v[0]), abs(forward_v[1]))
        angle_b = atan2(abs(backward_v[0]), abs(backward_v[1]))
        angle_i = (angle_f + angle_b)/2
        angle_dif = abs(angle_f - angle_i)
        return angle_dif*(distance_b+distance_f) / (2*distance_b*distance_f)

    for i in range(n):
        res[i] = p_i(i)
    plt.plot(res)
    plt.show()


def vector_length(vec):
    return (vec[0] **2 + vec[1]**2) ** 0.5

#path = "love.jpg"
path = "images/circle.jpg"


#path = "triangle.png"
#path = "line.jpg"
img1 = cv2.imread(path, 0)
cv2.imshow('original', img1)

bounds6 = lab09.border_list(img1)

cont_vec = bounds6
#cont_vec6 = bounds6.contour


curvature(cont_vec, 1)
# 2 або 3 - трикутник
#1 - лінія
#120 - серце
#120 - коло

#curvature(cont_vec6, 5)