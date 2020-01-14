from utils.msg_to_pixels import Msg2Pixels
import os
import argparse

parser = argparse.ArgumentParser('hello')
parser.add_argument("--ros_master_uri", type=str, default='http://192.168.102.15:11311', help="connection to "
                                                                                              "raspberry pi")
parser.add_argument("--path", type=str, default=None, help="path to dataset folder")
parser.add_argument("--display_time", type=int, default=3, help="time of the display of the RGB image with the "
                                                                "colored circles")
opt = parser.parse_args()

os.environ['ROS_MASTER_URI'] = opt.ros_master_uri  # connection to raspberry pi

i = 0

path = opt.path
a = Msg2Pixels(save_pcl=True)

while True:
    text = raw_input("-----------------------------------------\nc --> Capture \ne --> Exit \nr --> "
                     "Repeat\nint i --> start at point i\n-----------------------------------------\n\n")

    if text == 'c':
        save = a.save_images(path + '/image' + str(i))
    elif text == 'e':
        print('\nExiting...\n')
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
