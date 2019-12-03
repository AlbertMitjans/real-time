from __future__ import print_function, division

import time
import torch.utils.data
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import transforms
from my_classes import pad_to_square
import numpy as np

from my_classes import AverageMeter, accuracy, init_model_and_dataset


def validate(val_loader, model, end_epoch, epoch=0, save_imgs=False):
    batch_time = AverageMeter()
    eval_recall = AverageMeter()
    eval_precision = np.array([AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()])

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for data_idx, data in enumerate(val_loader):
        input = data['image'].float().cuda()
        target = data['grid'].float().cuda()
        corners = data['corners']

        # compute output
        output = model(input)

        # measure accuracy
        max_out = accuracy(corners, output.data, target, input, end_epoch, epoch, eval_recall, eval_precision)

        if save_imgs:
            image = Image.open('/home/amitjans/Desktop/Hourglass/data/rgb/' + data['img_name'][0] + '.png')
            image = pad_to_square(transforms.ToTensor()(image))
            fig, ax = plt.subplots(1, 2)
            plt.ion()
            ax[1].imshow(transforms.ToPILImage()(image))
            plt.scatter(max_out[:, 1], max_out[:, 0], marker='o', c='r', s=2)
            ax[0].imshow(output.cpu().detach().numpy()[0][0], cmap='gray')
            plt.savefig('/home/amitjans/Desktop/Hourglass/output/' + data['img_name'][0] + '.png')
            plt.close('all')

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(' * Recall(%): {top1:.3f}\t' ' * Precision(%):  ({top2:.3f}, {top3:.3f}, {top4:.3f}, {top5:.3f})\t'
          .format(top1=eval_recall.avg * 100, top2=eval_precision[0].avg * 100, top3=eval_precision[1].avg * 100,
                  top4=eval_precision[2].avg * 100, top5=eval_precision[3].avg * 100))

    global_precision = np.array([eval_precision[0].avg, eval_precision[1].avg, eval_precision[2].avg, eval_precision[3].avg])

    return eval_recall.avg, global_precision
