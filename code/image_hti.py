import cv2 as cv
import os
import numpy as np

image_path = '../../../data/islabdata/ori/train/fake/'
image = cv.imread(image_path+'adylbeequz0132.jpg')
new_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

array = np.full(new_image.shape, (0,255,0), dtype=np.uint8)
val_add_image = cv.add(new_image, array)
val_add_image = cv.cvtColor(val_add_image, cv.COLOR_HSV2BGR)

filename = 'adylbeequz0132.jpg'
cv.imwrite(filename,val_add_image)
