import numpy as np


def fix_depth(img):
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

