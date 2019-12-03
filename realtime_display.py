import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from Utils.msg_to_pixels import Msg2Pixels

import os

os.environ['ROS_MASTER_URI'] = 'http://192.168.102.10:11311'  # connection to raspberry pi
os.environ['ROS_IP'] = '192.168.102.10'

a = Msg2Pixels()

# create two subplots
fig, ax = plt.subplots(1, 2)
ax[0].axis('off')
ax[1].axis('off')
# create two image plots
im_rgb = ax[0].imshow(a.return_images()[0])
im_depth = ax[1].imshow(a.return_images()[1], cmap='gray')


def update(i):
    im_rgb.set_data(a.return_images()[0])
    im_depth.set_data(a.return_images()[1])


ani = FuncAnimation(fig, update, interval=20)


def close(event):
    if event.key == 'q':
        plt.close(event.canvas.figure)


cid = plt.gcf().canvas.mpl_connect("key_press_event", close)

plt.show()
