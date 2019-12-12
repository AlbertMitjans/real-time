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
    img = img[152:456, 92:585]
    return img


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


def local_max(image):
    max_out = peak_local_max(image, min_distance=19, threshold_rel=0.5, exclude_border=False, indices=False)
    labels_out = label(max_out)[0]
    max_out = np.array(center_of_mass(max_out, labels_out, range(1, np.max(labels_out) + 1))).astype(np.int)
    max_values = []

    for index in max_out:
        max_values.append(image[index[0]][index[1]])

    max_out = np.array([x for _, x in sorted(zip(max_values, max_out), reverse=True, key=lambda x: x[0])])

    return max_out


def corner_mask(output, gradient):
    max_coord = local_max(output)
    corners = torch.zeros(3, output.shape[0], output.shape[1])
    grad_values = []
    for idx, (i, j) in enumerate(max_coord):
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
    return corners, grad_values


def depth_layers(depth):
    edges = transforms.ToTensor()(transforms.ToPILImage()(depth[0]).convert('L').filter(ImageFilter.FIND_EDGES))
    contours = transforms.ToTensor()(transforms.ToPILImage()(depth[0]).convert('L').filter(ImageFilter.CONTOUR))
    depth = torch.stack((depth[0], depth[0], depth[0])).unsqueeze(0)
    return depth
