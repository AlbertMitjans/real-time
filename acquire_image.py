from Utils.msg_to_pixels import Msg2Pixels
from read_images import read_image
import os

os.environ['ROS_MASTER_URI'] = 'http://192.168.102.10:11311'  # connection to raspberry pi
os.environ['ROS_IP'] = '192.168.102.10'

i = 0

path = '/media/amitjans/DATA/'
a = Msg2Pixels()

while True:
    text = raw_input("-----------------------------------------\nEnter --> Take picture \nb --> Break \nr --> "
                 "Repeat\nint i --> start at point i\n-----------------------------------------\n")
    if text == 'b':
        print('Breaking...')
        break
    if text == "r":
        print('Repeating...')
        i = i - 1
    try:
        i = int(text)
        print('Changing value...')
    except ValueError:
        pass

    save = a.save_images(path + 'image' + str(i))
    read_image(path + 'image' + str(i))
    i += 1

a.unsubscribe()


