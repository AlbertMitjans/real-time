import numpy as np
import time
import matplotlib.pyplot as plt
import cv2


def hsv_segmentation(path):
    for i in range(100):
        rgb_image = cv2.imread(path + '_rgb.png', -1)
        if rgb_image is None:
            time.sleep(0.5)  # we wait until the image is saved (so we can open it)
            continue
        break

    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)  # we convert to hsv
    centers = []

    for [i, j, color] in [[[0, 170], [5, 180], 'red'], [45, 75, 'green'], [24, 32, 'yellow'], [100, 120, 'blue']]:
        if color == 'red':
            mask = cv2.inRange(hsv, np.array([i[0], 200, 50]), np.array([j[0], 255, 255]))
            mask += cv2.inRange(hsv, np.array([i[1], 200, 50]), np.array([j[1], 255, 255]))  # RED
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))  # filter to kill noise

        elif color == 'blue':
            mask = cv2.inRange(hsv, np.array([i, 50, 30]), np.array([j, 255, 255]))  # BLUE
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

        elif color == 'yellow':
            mask = cv2.inRange(hsv, np.array([i, 200, 110]), np.array([j, 255, 255]))  # YELLOW
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))  # filter to kill noise
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

        elif color == 'green':
            mask = cv2.inRange(hsv, np.array([i, 150, 60]), np.array([j, 255, 255]))  # GREEN
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))  # filter to kill noise
            #cv2.imshow('im', hsv)
            #cv2.imshow('rgb', rgb_image)
            #cv2.imshow('blue', mask)
            #cv2.waitKey(0)

        indexes = np.where(mask == 255)  # we get the coordinates of the pixels segmented

        # cv2.imwrite(path + "_mask_" + color + ".png", mask)

        try:  # assuming we just have one red object, we detect the center
            center_x = int((indexes[0][0] + indexes[0][-1]) / 2)
            center_line = np.where(indexes[0] == center_x)[0]
            center_y = indexes[1][center_line[int(np.size(center_line) / 2)]]
            centers.append([color, center_x, center_y])
            # we draw a circle in the rgb image to see if it was done correctly
            color_bgr = {'red': (0, 0, 255), 'green': (0, 255, 0), 'yellow': (0, 255, 255), 'blue': (255, 0, 0)}
            cv2.circle(rgb_image, (center_y, center_x), 15, color_bgr[color], thickness=1)
            # print(color, center_x, center_y)
            #cv2.imwrite(path + '_mask_' + color + ".png", mask)

        except IndexError:
            print('There is none or more than one ' + color + ' object in the image')
            centers.append([color, '-1', '-1'])
            #cv2.imwrite(path + "_mask_" + color + ".png", mask)

    np.savetxt(path + '.csv', centers, delimiter=",", fmt='%s')
    # we show the image with the drawn circles
    plt.figure()
    plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
    plt.show(block=False)
    plt.pause(3)
    plt.close('all')