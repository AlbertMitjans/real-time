import pandas as pd
import cv2
import numpy as np


def read_image(path_depth, path_csv, path2):
    depth = cv2.imread(path_depth + '.png', -1)
    f = pd.read_csv(path_csv + '.csv', header=None)
    c = f.iloc[:, :].as_matrix()
    width = 10.0
    height = 10.0
    array = []
    for i, (color, x, y) in enumerate(c):
        if x != 0 and y != 0:
            array.append([0, float(y)/depth.shape[1], float(x)/depth.shape[0], width/depth.shape[1], height/depth.shape[0]])

    np.savetxt(path2 + '.txt', array, delimiter=' ', fmt='%.15f')


path_depth = '/home/amitjans/Desktop/Hourglass/data/rgb/image'
path_csv = '/home/amitjans/Desktop/Hourglass/data/csv/image'
path2 = '/home/amitjans/Desktop/Yolo/data/custom/labels/image'

for j in range(0, 403):
    print(j)
    try:
        read_image(path_depth + str(j), path_csv + str(j), path2 + str(j))
    except IOError:
        print("No image found")
        continue
