import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_image(path):
    file_type = '.tif'

    depth = cv2.imread(path + file_type, -1)

    indices = np.where(depth == 0)

    for idx in range(indices[0].size):
        av = 0
        num = 0
        for a in range(indices[0][idx]-2, indices[0][idx]+3):
            for b in range(indices[1][idx]-2, indices[1][idx]+3):
                if depth[a, b] != 0:
                    av += depth[a, b]
                    num += 1

        av = av / num
        depth[indices[0][idx], indices[1][idx]] = av

    cv2.imwrite(path + file_type, depth)


path = '/home/amitjans/Desktop/My_network/data/full_dataset/'

i = 0

for j in range(0, 403):
    print(j)
    if j != 7 and j != 181 and j != 185:
        read_image(path + 'image' + str(j))


