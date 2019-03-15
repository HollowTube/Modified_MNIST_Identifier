import numpy as np  # linear algebra
import cv2

import pandas as pd

train_images = pd.read_pickle('../data/train_images.pkl')
train_labels = pd.read_csv('../data/train_labels.csv')

img_rows = train_images.shape[1]
img_cols = train_images.shape[2]


import matplotlib.pyplot as plt

# Let's show image with id 16
img_idx = 1
print(train_labels.iloc[img_idx])
img = train_images[img_idx]
img_bin = cv2.threshold(img, 200, 255, 0)[1]


def getBodyContours(bw_img):
    intImg = np.uint8(bw_img)
    contours = cv2.findContours(intImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    contours = np.array(contours)
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 20:
            x, y, w, h = cv2.boundingRect(cnt)
            box = [x, y, x+w, h+y,h,w]
            # rect = cv2.minAreaRect(cnt)
            # box = cv2.boxPoints(rect)
            box = np.int0(box)
            boxes.append(box)
    boxes = np.array(boxes)
    return boxes


boxes = getBodyContours(img_bin)
topbox = sorted(boxes,key=lambda x: max(x[4],x[5]))[-1]

# for box in boxes:
#     cv2.drawContours(img, [box], 0, (0,255,0), 1)

x, y, x2, y2,w,h = map(int, topbox)
cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)

cv2.imshow('img', img / 255)
cv2.imshow('bw', img_bin)
cv2.waitKey(0)
