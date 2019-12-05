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

from scipy import ndimage


def update(i):
    rgb, output, corners, gradient, grad_values = get_output()
    rgb, corners = transforms.ToPILImage()(rgb), transforms.ToPILImage()(corners)
    image = Image.blend(rgb, corners, 0.5)
    gradient = Image.blend(transforms.ToPILImage()((gradient*20).expand(3, -1, -1)), corners, 0.5)
    im_rgb.set_data(image)
    im_output.set_data(output)
    im_gradient.set_data(gradient)
    txt = ''
    max_colors = ['1st (red)', '2nd (green)', '3rd (blue)', '4th (yellow)']
    for idx, val in enumerate(grad_values):
        txt += max_colors[idx] + '---> grad = ' + str(val) + '\n'

    text.set_text(txt)


def get_output():
    # we read the image and get rid of the borders (they have 0 values)
    rgb, depth = a.return_images()
    rgb, depth = cut_image(rgb), cut_image(depth)
    rgb, depth = transforms.ToTensor()(rgb), transforms.ToTensor()(depth.astype(np.float32))
    depth = depth / depth.max()

    gradient = compute_gradient(depth[0])

    # we compute the edges and contour for the neural network
    edges = transforms.ToTensor()(transforms.ToPILImage()(depth[0]).convert('L').filter(ImageFilter.FIND_EDGES))
    contours = transforms.ToTensor()(transforms.ToPILImage()(depth[0]).convert('L').filter(ImageFilter.CONTOUR))
    depth = torch.stack((depth[0], edges[0], contours[0])).unsqueeze(0)
    output = model(depth).cpu().detach().numpy().clip(0)[0][0]
    max_out = max_gaussian(output)
    corners = torch.zeros(3, output.shape[0], output.shape[1])
    grad_values = []
    for idx, (i, j) in enumerate(max_out):
        cx, cy = draw.circle_perimeter(i, j, 5, shape=output.shape)
        if idx < 4:
            grad_values.append(gradient[cx.min():cx.max(), cy.min():cy.max()].sum())
            if idx == 3:
                corners[0, cx, cy] = 1.
                corners[1, cx, cy] = 1.
            else:
                corners[idx, cx, cy] = 1.

        else:
            corners[:, cx, cy] = 1.
    return rgb, output, corners, gradient, grad_values


def compute_gradient(image):
    # we compute the gradient of the image
    '''kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sx = ndimage.convolve(depth[0][0], kx)
        sy = ndimage.convolve(depth[0][0], ky)'''
    sx = ndimage.sobel(image, axis=0, mode='nearest')
    sy = ndimage.sobel(image, axis=1, mode='nearest')
    gradient = transforms.ToTensor()(np.hypot(sx, sy))

    return gradient[0]


def max_gaussian(image):
    max_out = peak_local_max(image, min_distance=19, threshold_rel=0.5, exclude_border=False, indices=False)
    labels_out = label(max_out)[0]
    max_out = np.array(center_of_mass(max_out, labels_out, range(1, np.max(labels_out) + 1))).astype(np.int)
    max_values = []

    for index in max_out:
        max_values.append(image[index[0]][index[1]])

    max_out = np.array([x for _, x in sorted(zip(max_values, max_out), reverse=True, key=lambda x: x[0])])

    return max_out


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
im_gradient = ax[1].imshow(get_output()[3])
text = plt.figtext(0.44, 0.25, '')

ani = FuncAnimation(fig, update, interval=1000/5)


def action(event):
    if event.key == 'q':
        plt.close(event.canvas.figure)

    if event.key == 's':
        plt.savefig('Images/image' + str(time.time()) + '.png')


cid = plt.gcf().canvas.mpl_connect("key_press_event", action)

plt.show()