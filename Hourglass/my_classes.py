from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.nn.parallel
import torch.optim
import torch.utils.data
import re
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
import scipy.stats as st
from skimage.feature import peak_local_max
from scipy.ndimage.measurements import center_of_mass, label

from PIL import Image, ImageFilter

from Hourglass.Stacked_Hourglass import HourglassNet, Bottleneck
from losses import JointsMSELoss


class CornersDataset(Dataset):
    def __init__(self, root_dir, end_file, depth, target_shape=(8, 12, 12), transform=None):
        self.img_names = []
        self.corners = []
        self.colors = []
        self.root_dir = root_dir
        self.transform = transform
        self.end_file = end_file
        self.target_shape = target_shape
        self.depth = depth
        self.validation = False
        self.read_csv()

    def read_csv(self):
        for root, dirs, files in os.walk(self.root_dir):
            files.sort(key=natural_keys)
            for file in files:
                if file.endswith(self.end_file):
                    self.img_names.append(file)
                if file.endswith(".csv"):
                    f = pd.read_csv(os.path.join(self.root_dir, file), header=None)
                    c = f.iloc[:, :].values
                    for color, x, y in c:
                        self.colors.append(color)
                        if x == '-' and y == '-':
                            c[np.where(c == x)] = 0
                            c[np.where(c == y)] = 0
                        elif x is 0 and y is 0:
                            continue
                        else:
                            c[np.where(c == x)] = int(x)
                            c[np.where(c == y)] = int(y)

                    self.corners.append(c[:, 1:])

    def evaluate(self):
        self.validation = True

    def __len__(self):
        return len(self.corners)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.img_names[idx])
        img_number = os.path.basename(img_name)[:-4]

        corners = np.array([self.corners[idx]])
        corners = corners.astype(int).reshape(-1, 2)

        if not self.depth:
            image = Image.open(img_name)
            image = transforms.ToTensor()(image).type(torch.float32)

        if self.depth:
            image = Image.open(img_name)
            image = transforms.ToTensor()(image).type(torch.float32)
            image = image / image.max()
            edges = transforms.ToTensor()(transforms.ToPILImage()(image[0]).convert('L').filter(ImageFilter.FIND_EDGES))
            contours = transforms.ToTensor()(transforms.ToPILImage()(image[0]).convert('L').filter(ImageFilter.CONTOUR))
            image = torch.stack((image[0], edges[0], contours[0]))

        grid = transforms.ToTensor()(gaussian(image[0], corners, target_size=image[0].size())[0]).type(torch.float32)

        sample = {'image': image, 'grid': grid, 'img_name': img_number, 'corners': corners}

        if not self.validation:
            if self.transform:
                sample = self.transform(sample)

        sample['image'] = pad_to_square(sample['image'])
        sample['grid'] = pad_to_square(sample['grid'])

        '''fig, ax = plt.subplots(1, 2)
        ax[0].imshow(sample['grid'][0], cmap='gray')
        ax[1].imshow(sample['image'][0], cmap='gray')
        plt.show()
        plt.waitforbuttonpress()
        plt.close('all')'''

        return sample


class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        return normalize(tensor, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ToTensor(object):
    def __init__(self, depth):
        self.depth = depth

    def __call__(self, sample):
        image, grid = sample['image'], sample['grid']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        '''if self.depth:
            image = torch.from_numpy(image).expand(1, image.shape[0], image.shape[1])'''
        if not self.depth:
            image = image.transpose((2, 0, 1))
        sample['image'] = torch.from_numpy(image)
        sample['grid'] = torch.from_numpy(grid)
        return sample


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        if np.random.random() < 0.5:
            image, grid = sample['image'], sample['grid']
            rc = transforms.RandomCrop((int(image.shape[1] * self.size), int(image.shape[2] * self.size)))
            im_grid = torch.stack((image[0], grid[0]))
            crop_im_grid = rc(transforms.ToPILImage()(im_grid))
            crop_image, crop_grid = Image.Image.split(crop_im_grid)
            res_image = crop_image.resize((image.shape[2], image.shape[1]))
            grid = transforms.ToTensor()(crop_grid.resize((image.shape[2], image.shape[1])))
            edges = transforms.ToTensor()(res_image.convert('L').filter(ImageFilter.FIND_EDGES))
            contours = transforms.ToTensor()(res_image.convert('L').filter(ImageFilter.CONTOUR))
            image = torch.stack((transforms.ToTensor()(res_image)[0], edges[0], contours[0]))

            sample['image'] = image
            sample['grid'] = grid

        return sample


class HorizontalFlip(object):
    def __call__(self, sample):
        if np.random.random() < 0.5:
            image = sample['image']
            grid = sample['grid']
            sample['image'] = torch.flip(image, [-1])
            sample['grid'] = torch.flip(grid, [-1])
        return sample


def pad_to_square(img):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = nn.functional.pad(img, pad, "constant", value=0)

    return img


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 2


def show_corners(image, corners):
    """Show image with landmarks"""
    plt.imshow(image, cmap='gray')
    plt.scatter(corners[:, 1], corners[:, 0], s=10, marker='.', c='r')
    plt.pause(0.005)  # pause a bit so that plots are updated


def normalize(tensor, mean, std, inplace=False):
    image = tensor['image']

    if not _is_tensor_image(image):
        raise TypeError('tensor is not a torch image.')

    if not inplace:
        image = image.clone()

    dtype = image.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=image.device)
    std = torch.as_tensor(std, dtype=dtype, device=image.device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])

    tensor['image'] = image

    return tensor


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def show_output(i, input, output, target, depth=True):
    img = input.cpu().detach().numpy()

    f, axarray = plt.subplots(1, 2)
    axarray[0].imshow(target[i, 0], cmap='gray')
    axarray[1].imshow(output[i, 0], cmap='gray')
    plt.figure()
    if depth:
        plt.imshow(img[i][0] - img[i][0].min(), cmap='gray', vmin=0, vmax=img[i].max())
    if not depth:
        plt.imshow(np.moveaxis(img[i] - img[i].min(), 0, -1), vmax=img[i].max())
    plt.show()
    plt.waitforbuttonpress()
    plt.close('all')


###############################################
# INITIATE MODEL
###############################################


def init_model_and_dataset(depth, directory, normalize_data, lr=5e-6, weight_decay=0, momentum=0):
    # define the model
    model = HourglassNet(Bottleneck)
    model = nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss().cuda()
    optimizer = torch.optim.RMSprop(model.parameters(), lr, weight_decay=weight_decay, momentum=momentum)

    checkpoint = torch.load("checkpoint/hg_s2_b1/model_best.pth.tar")

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    model = nn.Sequential(model, nn.Conv2d(16, 1, kernel_size=1).cuda())

    if depth:
        end_file = '.tif'
    if not depth:
        end_file = '.png'

    cudnn.benchmark = True

    random_crop = RandomCrop(size=0.8)
    horizontal_flip = HorizontalFlip()

    if normalize_data:
        normalize_image = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_dataset = CornersDataset(root_dir=directory + 'train_dataset', end_file=end_file, depth=depth,
                                       transform=transforms.Compose([normalize_image, random_crop, horizontal_flip]))
        val_dataset = CornersDataset(root_dir=directory + 'val_dataset', end_file=end_file, depth=depth,
                                     transform=transforms.Compose([normalize_image, random_crop, horizontal_flip]))
    if not normalize_data:
        train_dataset = CornersDataset(root_dir=directory + 'train_dataset', end_file=end_file, depth=depth,
                                       transform=transforms.Compose([random_crop, horizontal_flip]))
        val_dataset = CornersDataset(root_dir=directory + 'val_dataset', end_file=end_file, depth=depth,
                                     transform=transforms.Compose([random_crop, horizontal_flip]))

    return model, train_dataset, val_dataset, criterion, optimizer


##############################################
# TYPES OF TARGETS
##############################################


def distance_from_corner(target_shape, image, corners):
    target = np.zeros(target_shape, dtype=int)
    n = float(image.shape[0]) / float(target.shape[1])
    m = float(image.shape[1]) / float(target.shape[2])
    for i, (x, y) in enumerate(corners):
        a = int(x / n)
        b = int(y / m)
        ax = n * (a + 1 / 2) - x
        ay = (m * (b + 1 / 2)) - y

        if x == 0 and y == 0:
            target[3 * i, a, b] = 0
        else:
            target[3 * i, a, b] = 1
            target[3 * i + 1, a, b] = ax
            target[3 * i + 2, a, b] = ay

    return target


def corner_colors(target_shape, image, corners, colors):
    target = np.zeros(target_shape[1:], dtype=int)
    n = float(image.shape[0]) / float(target.shape[0])
    m = float(image.shape[1]) / float(target.shape[1])
    color_idx = {'red': 1, 'green': 2, 'yellow': 3, 'blue': 4}
    for i, (x, y) in enumerate(corners):
        a = int(x / n)
        b = int(y / m)
        target[a, b] = color_idx[colors[i]]

    return target


def red_color(corners, image):
    target = np.array([0, 0])
    if corners[0][0] and corners[0][1] != 0:
        target = np.array([corners[0][0], corners[0][1]])

    return target


def number_red_images(corners):
    red_images = 0
    for (red, green, yellow, blue) in corners:
        if np.all(red != 0):
            red_images += 1

    return 100 * float(red_images) / float(len(corners))


def gaussian_red(image, corners, kernel=20, nsig=5, target_size=(1, 76, 124)):
    target = np.zeros(target_size)
    n = float(image.shape[0]) / float(target.shape[1])
    m = float(image.shape[1]) / float(target.shape[2])
    if np.all(corners[0] != 0):
        a = int(corners[0, 0] / n)
        b = int(corners[0, 1] / m)
        x = np.linspace(-nsig, nsig, kernel + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kern2d = np.outer(kern1d, kern1d)
        ax = a - kern2d.shape[0] // 2
        ay = b - kern2d.shape[1] // 2
        paste(target[0], kern2d / kern2d.max(), (ax, ay))

    return target


def gaussian(image, corners, kernel=39, nsig=5):
    target = np.zeros((4, image.shape[0], image.shape[1]))
    n = float(image.shape[0]) / float(target.shape[1])
    m = float(image.shape[1]) / float(target.shape[2])
    for i, (x, y) in enumerate(corners):
        if x != 0 and y != 0:
            a = int(x / n)
            b = int(y / m)
            x = np.linspace(-nsig, nsig, kernel + 1)
            kern1d = np.diff(st.norm.cdf(x))
            kern2d = np.outer(kern1d, kern1d)
            ax = a - kern2d.shape[0] // 2
            ay = b - kern2d.shape[1] // 2
            paste(target[i], kern2d / kern2d.max(), (ax, ay))

    target = np.resize(target.sum(0), (1, image.shape[0], image.shape[1]))

    return target


def nearest_corner(image, corners, kernel=40, nsig=5, target_size=(1, 76, 124)):
    target = np.zeros(target_size)
    n = float(image.shape[0]) / float(target.shape[1])
    m = float(image.shape[1]) / float(target.shape[2])
    for i, (x, y) in enumerate(corners):
        if x == corners[:, 0].max() and (x != 0 and y != 0):
            a = int(x / n)
            b = int(y / m)
            x = np.linspace(-nsig, nsig, kernel + 1)
            kern1d = np.diff(st.norm.cdf(x))
            kern2d = np.outer(kern1d, kern1d)
            ax = a - kern2d.shape[0] // 2
            ay = b - kern2d.shape[1] // 2
            paste(target[0], kern2d / kern2d.max(), (ax, ay))

    return target


def paste_slices(tup):
    pos, w, max_w = tup
    wall_min = max(pos, 0)
    wall_max = min(pos + w, max_w)
    block_min = -min(pos, 0)
    block_max = max_w - max(pos + w, max_w)
    block_max = block_max if block_max != 0 else None
    return slice(wall_min, wall_max), slice(block_min, block_max)


def paste(wall, block, loc):
    loc_zip = zip(loc, block.shape, wall.shape)
    wall_slices, block_slices = zip(*map(paste_slices, loc_zip))
    wall[wall_slices] = block[block_slices]


def distance_from_max(output):
    max = np.where(output == output.max())
    max_middle = [max[0][int(max[0].size / 2)], max[1][int(max[1].size / 2)]]
    distance = np.zeros(output.shape)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            len = ((i - max_middle[0]) ** 2 + (j - max_middle[1]) ** 2) ** (1 / 2)
            if len >= 30:
                distance[i][j] = len * output[i, j]

    return distance.sum()


###############################################
# TYPES OF ACCURACY
###############################################


def accuracy(corners, output, target, input, end_epoch, epoch, global_recall, global_precision, depth=True):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)

    # we send the data to CPU
    output = output.cpu().detach().numpy().clip(0)
    target = target.cpu().detach().numpy()

    for batch_unit in range(batch_size):  # for each batch element
        recall, precision, max_out = multiple_gaussians(output[batch_unit], target[batch_unit])

        global_recall.update(recall)
        for i, (a, b) in enumerate(sorted(corners[batch_unit], key=lambda x: x[0], reverse=True)):
            if a != 0 and b != 0:
                global_precision[i].update(precision[i])

        # we check the result
        if epoch >= end_epoch:
            show_output(batch_unit, input, output, target, depth)

    return max_out


def one_gaussian(output, target, i, j):
    # we calculate the positions of the max value in output and target
    correct = 0

    if output[i, j].max() == 0:
        max_out = np.array([[-10, -10]])
    else:
        max_out = np.array(np.where(output[i, j] == output[i, j].max()))
        max_out = np.squeeze(np.split(max_out, max_out[0].size, 1))
        if max_out.size <= 2:
            max_out = [max_out]

    if target[i, j].max() == 0:
        max_target = np.array([[-10, -10]])
    else:
        max_target = np.array(np.where(target[i, j] == target[i, j].max()))
        max_target = np.squeeze(np.split(max_target, max_target[0].size, 1))
        if max_target.size <= 2:
            max_target = [max_target]

    # we check if the maximum values are in the same position (+-3 pixels)
    for (a, b) in max_target:
        try:
            for (c, d) in max_out:
                l = np.absolute((a - c, b - d))
                if l[0] <= 3 and l[1] <= 3:
                    correct += 1
                    break
        except TypeError:
            print(max_out)

    return correct, max_target, max_out


def multiple_gaussians(output, target):
    # we calculate the positions of the max value in output and target
    max_target = peak_local_max(target[0].clip(0.99), min_distance=19, exclude_border=False, indices=False) #num_peaks=4)
    labels_target = label(max_target)[0]
    max_target = np.array(center_of_mass(max_target, labels_target, range(1, np.max(labels_target) + 1))).astype(np.int)

    true_p = np.array([0, 0, 0, 0]).astype(np.float)
    all_p = np.array([0, 0, 0, 0]).astype(np.float)

    max_out = peak_local_max(output[0], min_distance=19, threshold_rel=0.1, exclude_border=False, indices=False)
    labels_out = label(max_out)[0]
    max_out = np.array(center_of_mass(max_out, labels_out, range(1, np.max(labels_out) + 1))).astype(np.int)

    max_values = []

    for index in max_out:
        max_values.append(output[0][index[0]][index[1]])

    max_out = np.array([x for _,x in sorted(zip(max_values, max_out), reverse=True, key=lambda x: x[0])])

    for n in range(min(4, max_out.shape[0])):
        max_out2 = max_out[:n+1]
        for i, (c, d) in enumerate(max_out2):
            if i < max_out2.shape[0] - 1:
                dist = np.absolute((max_out2[i+1][0] - c, max_out2[i+1][1] - d))
                if dist[0] <= 8 and dist[1] <= 8:
                    continue
            all_p[n] += 1
            count = 0
            for (a, b) in max_target:
                l = np.absolute((a - c, b - d))
                if l[0] <= 8 and l[1] <= 8:
                    true_p[n] += 1
                    count += 1
                    if count > 1:
                        all_p[n] += 1

    '''print(max_out)
    print(max_target)
    print(true_p)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(output[0], cmap='gray')
    ax[1].imshow(target[0], cmap='gray')
    plt.plot(max_out[:, 1], max_out[:, 0], 'r.')
    plt.show()
    plt.waitforbuttonpress(0)
    plt.close('all')'''

    num_targets = max_target.shape[0]
    if num_targets != 0:
        recall = true_p[min(4, max_out.shape[0]) - 1]/num_targets
        precision = true_p/all_p
        precision[np.isnan(precision)] = 0
    if num_targets == 0:
        recall = 0
        precision = np.array([0, 0, 0, 0]).astype(np.float)

    return recall, precision, max_out