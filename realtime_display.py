import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Utils.msg_to_pixels import Msg2Pixels
from Hourglass.Stacked_Hourglass import HourglassNet, Bottleneck
import torch.nn as nn
import torch
import os
import time
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from Utils.img_utils import compute_gradient, cut_image, corner_mask, depth_layers


def update(i):
    rgb, output, gradient, corners, grad_values = get_output()
    rgb, corners = transforms.ToPILImage()(rgb), transforms.ToPILImage()(corners)
    image = Image.blend(rgb, corners, 0.5)
    gradient = Image.blend(transforms.ToPILImage()((gradient*20).expand(3, -1, -1)), corners, 0.5)
    im_rgb.set_data(image)
    im_output.set_data(output)
    im_gradient.set_data(gradient)


def get_output():
    # we read the image and get rid of the borders (they have 0 values)
    rgb, depth = a.return_images()
    rgb, depth = cut_image(rgb), cut_image(depth)
    rgb, depth = transforms.ToTensor()(rgb), transforms.ToTensor()(depth.astype(np.float32))
    depth = depth / depth.max()

    gradient = compute_gradient(depth[0])

    # we compute the edges and contour for the neural network
    depth = depth_layers(depth)
    output = model(depth).cpu().detach().numpy().clip(0)[0][0]
    corners, grad_values = corner_mask(output, gradient)
    return rgb, output, gradient, corners, grad_values


os.environ['ROS_MASTER_URI'] = 'http://192.168.102.10:11311'  # connection to raspberry pi

a = Msg2Pixels()

# initiate model
model = HourglassNet(Bottleneck)
model = nn.DataParallel(model).cuda()
model = nn.Sequential(model, nn.Conv2d(16, 1, kernel_size=1).cuda())
checkpoint = torch.load('best_checkpoints/ckpt_1.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# create two subplots
fig, ax = plt.subplots(1, 3)
ax[2].axis('off')
ax[2].set_title('RGB image')
ax[0].axis('off')
ax[0].set_title('Network\'s output')
ax[1].axis('off')
ax[1].set_title('Gradient')

# create two image plots
im_rgb = ax[2].imshow(transforms.ToPILImage()(get_output()[0]))
im_output = ax[0].imshow(get_output()[1], cmap='afmhot')
im_gradient = ax[1].imshow(get_output()[2])

ani = FuncAnimation(fig, update, interval=1000/5)


def action(event):
    if event.key == 'q':
        plt.close(event.canvas.figure)

    if event.key == 's':
        plt.savefig('Images/image' + str(time.time()) + '.png')


cid = plt.gcf().canvas.mpl_connect("key_press_event", action)

plt.show()
