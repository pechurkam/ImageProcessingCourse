import cv2
import numpy as np

# МАРКУВАННЯ
# перший піксель
path = "images/OneMoreTry.jpg"

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def mark_func(img, poroh):
    res_img = np.copy(img)
    h = res_img.shape[0]
    w = res_img.shape[1]
    labels = []
    bool_arr = np.zeros((h, w), np.uint8)
    label = 20

    for i in range(h):
        for j in range(w):
            if bool_arr.item(i, j) == 1:
                continue
            if label > 250:
                print('First' + str(label))
                label = label % 255
                print(label)
            stack = [(i, j)]
            dif = 1
            while stack:
                (i1, j1) = stack.pop()

                for m in range(3):
                    for n in range(3):
                        ran1 = i1 + m - dif
                        ran2 = j1 + n - dif
                        if ran1 < 0 or ran2 < 0 or ran1 >= h or ran2 >= w:
                            continue
                        if bool_arr.item(ran1, ran2) == 1:
                            continue
                        if abs(img.item(i, j) - img.item(ran1, ran2)) <= poroh:
                            # res_img.itemset((ran1, ran2), label)
                            res_img[ran1, ran2] = label
                            stack.append((ran1, ran2))
                            # bool_arr.itemset((ran1, ran2), 1)
                            bool_arr[ran1, ran2] = 1

            label += 50
            labels.append(label)
    print(labels)

    labels_res = []
    labels.sort()
    #print(labels)
    l_size = len(labels)

    for i in range(l_size - 1):
        if labels[i] != labels[i + 1]:
            labels_res.append(labels[i])

    #print(labels_res)
    return res_img


def onclick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('chosen coords: ' + str(x) + ',' + str(y))
        new_img = np.copy(mark_func(img, 30))

        h = new_img.shape[0]

        w = new_img.shape[1]

        for i in range(h):
            for j in range(w):
                if new_img[i, j] != new_img[y, x]:
                    new_img[i, j] = 255
        cv2.imshow('Region' + str(x) + str(y), new_img)


cv2.imshow('Marking', mark_func(img, 100))
cv2.imshow('Original', img)
cv2.setMouseCallback("Marking", onclick)


cv2.waitKey(0)
cv2.destroyAllWindows()
