import cv2
import time
import pandas as pd
import numpy as np
import csv


def read_image(path):
    file_type = '.png'

    depth = cv2.imread(path + file_type, -1)

    depth = depth[152:456, 92:588]

    cv2.imwrite(path + file_type, depth)

    if file_type == '.tif':
        f = pd.read_csv(path + '.csv', header=None)
        c = f.iloc[:, :].as_matrix()
        for i, (color, x, y) in enumerate(c):
            if x == '-' and y == '-':
                f[1][i] = 0
                f[2][i] = 0
            elif x is 0 and y is 0:
                continue
            else:
                f[1][i] = int(x) - 152
                f[2][i] = int(y) - 92

        f.to_csv(path + '.csv', header=False, index=False)


path = '/home/amitjans/Desktop/My_network/data/full_dataset/'

for j in range(0, 403):
    try:
        read_image(path + 'image' + str(j))
    except TypeError:
        print("No image found")
        continue
    print(j)
