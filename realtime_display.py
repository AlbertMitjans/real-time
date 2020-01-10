import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch.nn as nn
import torch
import os
import time
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from Hourglass.Stacked_Hourglass import HourglassNet, Bottleneck
from utils.msg_to_pixels import Msg2Pixels
from utils.img_utils import cut_image, corner_mask, depth_layers, corner_mask_color


models = False
ckpt1 = 'ckpt_11'
layers1 = 'all'
ckpt2 = 'ckpt_1_1'
layers2 = 'all'


def update(i):
    if not models:
        rgb, output, corners = get_output(model, layers1)
        out = torch.stack((transforms.ToTensor()(output)[0], torch.zeros(output.shape), torch.zeros(output.shape)))
        out = transforms.ToPILImage()(out)
        rgb, corners = transforms.ToPILImage()(rgb), transforms.ToPILImage()(corners)
        image = Image.blend(rgb, out, 0.5)
        im_rgb.set_data(image)
        im_output.set_data(output)
    if models:
        rgb, output, corners = get_output(model, layers1, color='red')
        _, output2, corners2 = get_output(model2, layers2, color='blue')
        rgb, corners, corners2 = transforms.ToPILImage()(rgb), transforms.ToPILImage()(corners), transforms.ToPILImage()(corners2)
        image = Image.blend(rgb, corners, 0.5)
        #image = Image.blend(image, corners2, 0.5)
        im_rgb.set_data(image)
        im_output.set_data(output)
        im_output2.set_data(output2)


def get_output(model, layers=layers1, color='red'):
    depth, rgb = get_images()
    # we compute the edges and contour for the neural network
    depth = depth_layers(depth, layers).cuda()
    output = model(depth).cpu().detach().numpy().clip(0)[0][0]

    if not models:
        corners = corner_mask(output)
    if models:
        corners = corner_mask_color(output, color)

    return rgb, output, corners


def get_images():
    # we read the image and get rid of the borders (they have 0 values)
    rgb, depth = a.return_images()
    depth = depth.copy()
    depth[np.isnan(depth)] = 0
    rgb, depth = cut_image(rgb), cut_image(depth)
    rgb, depth = transforms.ToTensor()(rgb), transforms.ToTensor()(depth.astype(np.float32))
    depth = depth / depth.max()

    return depth, rgb


os.environ['ROS_MASTER_URI'] = 'http://192.168.102.15:11311'  # connection to raspberry pi

a = Msg2Pixels()

# initiate model
model = HourglassNet(Bottleneck)
model = nn.DataParallel(model).cuda()
model = nn.Sequential(model, nn.Conv2d(16, 1, kernel_size=1).cuda())
checkpoint = torch.load('best_checkpoints/{model}.pth'.format(model=ckpt1))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
if models:
    model2 = HourglassNet(Bottleneck)
    model2 = nn.DataParallel(model2).cuda()
    model2 = nn.Sequential(model2, nn.Conv2d(16, 1, kernel_size=1).cuda())
    checkpoint2 = torch.load('best_checkpoints/{model}.pth'.format(model=ckpt2))
    model2.load_state_dict(checkpoint2['model_state_dict'])
    model2.eval()

# create two subplots
if not models:
    fig, ax = plt.subplots(1, 2)
    ax[1].axis('off')
    ax[1].set_title('RGB image')
    ax[0].axis('off')
    ax[0].set_title('Network\'s output')

if models:
    fig, ax = plt.subplots(1, 3)
    ax[2].axis('off')
    ax[2].set_title('RGB image')
    ax[0].axis('off')
    ax[0].set_title('{model} network\'s output'.format(model=ckpt1))
    ax[1].axis('off')
    ax[1].set_title('{model} network\'s output'.format(model=ckpt2))

# create two image plots
if not models:
    im_rgb = ax[1].imshow(transforms.ToPILImage()(get_images()[1]))
    im_output = ax[0].imshow(get_output(model, layers1)[1], cmap='afmhot', vmin=0, vmax=1)

if models:
    im_rgb = ax[2].imshow(transforms.ToPILImage()(get_images()[1]))
    im_output = ax[0].imshow(get_output(model, layers1)[1], cmap='afmhot', vmin=0, vmax=1)
    im_output2 = ax[1].imshow(get_output(model2, layers2)[1], cmap='afmhot', vmin=0, vmax=1)

ani = FuncAnimation(fig, update, interval=1000/20)


def action(event):
    if event.key == 'q':
        plt.close(event.canvas.figure)

    if event.key == 's':
        plt.savefig('Images/image' + str(time.time()) + '.png')


cid = plt.gcf().canvas.mpl_connect("key_press_event", action)

plt.show()
