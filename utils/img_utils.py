import numpy as np
import torchvision.transforms as transforms
from scipy import ndimage
from skimage.feature import peak_local_max
from scipy.ndimage.measurements import center_of_mass, label
import matplotlib.pyplot as plt
import torch
import skimage.draw as draw
from PIL import ImageFilter


def fill_zero_values(img):
    image = img.copy()
    indices = np.where(image == 0)

    for idx in range(indices[0].size):
        av = 0
        num = 0
        for a in range(indices[0][idx]-2, indices[0][idx]+3):
            for b in range(indices[1][idx]-2, indices[1][idx]+3):
                try:
                    if image[a, b] != 0:
                        av += image[a, b]
                        num += 1
                except IndexError:
                    continue

        av = av / num
        image[indices[0][idx], indices[1][idx]] = av

    return image


def cut_image(img):
    img = img[80:-30, :-60]
    return img


def local_max(image):
    max_out = peak_local_max(image, min_distance=19, threshold_rel=0.5, exclude_border=False, indices=False)
    labels_out = label(max_out)[0]
    max_out = np.array(center_of_mass(max_out, labels_out, range(1, np.max(labels_out) + 1))).astype(np.int)
    max_values = []

    for index in max_out:
        max_values.append(image[index[0]][index[1]])

    max_out = np.array([x for _, x in sorted(zip(max_values, max_out), reverse=True, key=lambda x: x[0])])

    return max_out


def corner_mask(output):
    max_coord = local_max(output)
    corners = torch.zeros(3, output.shape[0], output.shape[1])
    for idx, (i, j) in enumerate(max_coord):
        cx, cy = draw.circle_perimeter(i, j, 5, shape=output.shape)
        if idx < 4:
            corners[0, cx, cy] = 1.

    return corners


def corner_mask_color(output, color):
    max_coord = local_max(output)
    corners = torch.zeros(3, output.shape[0], output.shape[1])
    for idx, (i, j) in enumerate(max_coord):
        cx, cy = draw.circle_perimeter(i, j, 5, shape=output.shape)
        if color == 'red':
            corners[0, cx, cy] = 1.
        if color == 'green':
            corners[1, cx, cy] = 1.
        if color == 'blue':
            corners[2, cx, cy] = 1.

    return corners


def depth_layers(depth, layers):
    edges = transforms.ToTensor()(transforms.ToPILImage()(depth[0]).convert('L').filter(ImageFilter.FIND_EDGES))
    contours = transforms.ToTensor()(transforms.ToPILImage()(depth[0]).convert('L').filter(ImageFilter.CONTOUR))
    if layers == 'all':
        depth = torch.stack((depth[0], edges[0], contours[0])).unsqueeze(0)
    elif layers == 'edges':
        depth = torch.stack((depth[0], edges[0], depth[0])).unsqueeze(0)
    elif layers == 'contours':
        depth = torch.stack((depth[0], depth[0], contours[0])).unsqueeze(0)
    elif layers == 'depth':
        depth = torch.stack((depth[0], depth[0], depth[0])).unsqueeze(0)
    elif layers == 'one':
        depth = depth.unsqueeze(0)

    return depth
