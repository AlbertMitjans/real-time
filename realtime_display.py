import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch.nn as nn
import torch
import os
import time
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import argparse

from model.Stacked_Hourglass import HourglassNet, Bottleneck
from utils.msg_to_pixels import Msg2Pixels
from utils.img_utils import cut_image, depth_layers


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt1", type=str, default="checkpoints/ckpt_11.pth", help="path to ckpt file")
parser.add_argument("--ckpt2", type=str, default="checkpoints/ckpt_1_1.pth", help="path to ckpt file")
parser.add_argument("--compare", type=str2bool, default=False, help="if True, the outputs of ckpt1 and ckpt2 will "
                                                                    "be displayed in order to compare them")
opt = parser.parse_args()

models = opt.compare
ckpt1 = opt.ckpt1
ckpt2 = opt.ckpt2


def update(i):
    rgb, output = get_output(model)
    out = torch.stack((transforms.ToTensor()(output)[0], torch.zeros(output.shape), torch.zeros(output.shape)))
    out = transforms.ToPILImage()(out)
    rgb = transforms.ToPILImage()(rgb)
    im_output.set_data(output)
    if not models:
        image = Image.blend(rgb, out, 0.5)
        im_rgb.set_data(image)
    if models:
        _, output2 = get_output(model2)
        out2 = torch.stack((transforms.ToTensor()(output)[0], transforms.ToTensor()(output2)[0], torch.zeros(output.shape)))
        out2 = transforms.ToPILImage()(out2)
        image = Image.blend(rgb, out2, 0.5)
        im_rgb.set_data(image)
        im_output2.set_data(output2)


def get_output(model, only_depth=False):
    depth, rgb = get_images()
    # we compute the edges and contour for the neural network
    depth = depth_layers(depth, only_depth).cuda()
    output = model(depth).cpu().detach().numpy().clip(0)[0][0]

    return rgb, output


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
checkpoint = torch.load('{model}'.format(model=ckpt1))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
if models:
    model2 = HourglassNet(Bottleneck)
    model2 = nn.DataParallel(model2).cuda()
    model2 = nn.Sequential(model2, nn.Conv2d(16, 1, kernel_size=1).cuda())
    checkpoint2 = torch.load('{model}'.format(model=ckpt2))
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
    ax[0].set_title('Model 1 (red)')
    ax[1].axis('off')
    ax[1].set_title('Model 2 (green)')

# create two image plots
if not models:
    im_rgb = ax[1].imshow(transforms.ToPILImage()(get_images()[1]))
    im_output = ax[0].imshow(get_output(model)[1], cmap='afmhot', vmin=0, vmax=1)

if models:
    im_rgb = ax[2].imshow(transforms.ToPILImage()(get_images()[1]))
    im_output = ax[0].imshow(get_output(model)[1], cmap='afmhot', vmin=0, vmax=1)
    im_output2 = ax[1].imshow(get_output(model2)[1], cmap='afmhot', vmin=0, vmax=1)

ani = FuncAnimation(fig, update, interval=1000/20)


def action(event):
    if event.key == 'q':
        plt.close(event.canvas.figure)

    if event.key == 's':
        plt.savefig('images/image' + str(time.time()) + '.png')


cid = plt.gcf().canvas.mpl_connect("key_press_event", action)

plt.show()
