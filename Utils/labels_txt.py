import pandas as pd
import cv2
import numpy as np

path1 = '/home/amitjans/Desktop/Yolo/data/custom/'
path2 = 'data/custom/images/image'

train = open(path1 + 'train.txt', 'w')
val = open(path1 + 'valid.txt', 'w')

for j in range(0, 403):
    if j != 181 and j != 7 and j != 185:
        a = np.random.randint(0, 100, dtype=int)
        if a <= 20:
            val.write(path2 + str(j) + '.tif' + '\n')
        else:
            train.write(path2 + str(j) + '.tif' + '\n')

train.close()
val.close()
