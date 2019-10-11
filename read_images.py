import cv2
import matplotlib.pyplot as plt
import time


def read_image(path):
    for i in range(100):
        img = cv2.imread(path + '.tif', -1)
        if img is None:
            time.sleep(0.5)
            continue
        break

    plt.imshow(img, cmap='gray', vmin=500, vmax=1000)
    plt.show()
