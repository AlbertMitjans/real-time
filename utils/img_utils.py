import numpy as np
import torchvision.transforms as transforms
from skimage.feature import peak_local_max
from scipy.ndimage.measurements import center_of_mass, label
import torch
import skimage.draw as draw
from PIL import ImageFilter


def cut_image(img):
    img = img[80:-30, :-60]
    return img


def depth_layers(depth, only_depth=False):
    if not only_depth:
        edges = transforms.ToTensor()(transforms.ToPILImage()(depth[0]).convert('L').filter(ImageFilter.FIND_EDGES))
        contours = transforms.ToTensor()(transforms.ToPILImage()(depth[0]).convert('L').filter(ImageFilter.CONTOUR))
        depth = torch.stack((depth[0], edges[0], contours[0])).unsqueeze(0)
    if only_depth:
        depth = depth.unsqueeze(0)

    return depth
