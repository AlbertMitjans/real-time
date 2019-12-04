import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Utils.msg_to_pixels import Msg2Pixels
from Hourglass.Stacked_Hourglass import HourglassNet, Bottleneck
from Hourglass.my_classes import pad_to_square, gaussian
from Utils.img_utils import fix_depth, cut_image
import torch.nn as nn
import torch
import os
import time
from skimage.feature import peak_local_max
from scipy.ndimage.measurements import center_of_mass, label
import numpy as np
import torchvision.transforms as transforms
from PIL import ImageFilter
from PIL import Image
import skimage.draw as draw


def update(i):
    rgb, output, corners = get_output()
    corners = torch.stack((corners, corners, corners))
    rgb, corners = transforms.ToPILImage()(rgb), transforms.ToPILImage()(corners)
    image = Image.blend(rgb, corners, 0.5)
    im_rgb.set_data(image)
    im_output.set_data(output[0][0])


def get_output():
    rgb, depth = a.return_images()
    rgb, depth = cut_image(rgb), cut_image(depth)
    rgb, depth = transforms.ToTensor()(rgb), transforms.ToTensor()(depth.astype(np.float32))
    depth = depth / depth.max()
    edges = transforms.ToTensor()(transforms.ToPILImage()(depth[0]).convert('L').filter(ImageFilter.FIND_EDGES))
    contours = transforms.ToTensor()(transforms.ToPILImage()(depth[0]).convert('L').filter(ImageFilter.CONTOUR))
    depth = torch.stack((depth[0], edges[0], contours[0]))
    rgb, depth = pad_to_square(rgb), pad_to_square(depth).unsqueeze(0)
    output = model(depth).cpu().detach().numpy()
    max_out = peak_local_max(output[0][0], min_distance=19, threshold_rel=0.3, exclude_border=False, indices=False)
    labels_out = label(max_out)[0]
    max_out = np.array(center_of_mass(max_out, labels_out, range(1, np.max(labels_out) + 1))).astype(np.int)
    corners = torch.zeros(output[0][0].shape)
    for (i, j) in max_out:
        cx, cy = draw.circle_perimeter(i, j, 5, shape=output[0][0].shape)
        corners[cx, cy] = 1
    return rgb, output, corners


os.environ['ROS_MASTER_URI'] = 'http://192.168.102.10:11311'  # connection to raspberry pi
os.environ['ROS_IP'] = '192.168.102.10'

a = Msg2Pixels()

# initiate model
model = HourglassNet(Bottleneck)
model = nn.DataParallel(model).cuda()
model = nn.Sequential(model, nn.Conv2d(16, 1, kernel_size=1).cuda())
checkpoint = torch.load('best_checkpoints/ckpt_1.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# create two subplots
fig, ax = plt.subplots(1, 2)
ax[0].axis('off')
ax[1].axis('off')

# create two image plots
im_rgb = ax[1].imshow(transforms.ToPILImage()(get_output()[0]))
im_output = ax[0].imshow(get_output()[1][0][0], cmap='gray')
plt.tight_layout()

ani = FuncAnimation(fig, update, interval=1000/5)


def close(event):
    if event.key == 'q':
        plt.close(event.canvas.figure)


cid = plt.gcf().canvas.mpl_connect("key_press_event", close)

plt.show()