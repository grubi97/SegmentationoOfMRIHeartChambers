import os
import cv2
import numpy as np

for img in os.listdir('zdim/imagesbayes'):
    image = cv2.imread('zdim/imagesbayes/' + img)
    if np.sum(image) == 0:
        os.remove('zdim/imagesbayes/' + img)

dir_a = []
dir_b = []

for fileA in os.listdir('zdim/imagesbayes'):
    dir_a.append(fileA)
for fileB in os.listdir('zdim/images'):
    dir_b.append(fileB)

for fileA in dir_a:
    if not fileA in dir_b:
        os.remove(os.path.join('zdim/imagesbayes', (fileA)))
