from utils.msg_to_pixels import Msg2Pixels
import cv2
import os
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ros_master_uri", type=str, default=None, help="connection to raspberry pi")
parser.add_argument("--path", type=str, default=None, help="path to dataset folder")
opt = parser.parse_args()


def read_image(path):
    img = cv2.imread(path + '.png')
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.show()
    plt.waitforbuttonpress()
    plt.close('all')


os.environ['ROS_MASTER_URI'] = opt.ros_master_uri  # connection to raspberry pi

i = 0

path = opt.path
a = Msg2Pixels(save_pcl=True)

while True:
    text = raw_input("-----------------------------------------\nc --> Capture \nb --> Break \nr --> "
                 "Repeat\nint i --> start at point i\n-----------------------------------------\n\n")

    if text == 'c':
        save = a.save_images(path + '/image' + str(i))
    elif text == 'b':
        print('\nBreaking...\n')
        break
    elif text == "r":
        print('\nRepeating...')
        i = i - 1
        save = a.save_images(path + '/image' + str(i))
    else:
        try:
            i = int(text)
            print('\nChanging value...')
            save = a.save_images(path + '/image' + str(i))
        except ValueError:
            print('\nTry again :(\n')
    i += 1

a.unsubscribe()
