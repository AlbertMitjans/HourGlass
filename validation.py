from __future__ import print_function, division

import time
import torch.utils.data
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import transforms
from my_classes import pad_to_square
import numpy as np
import skimage.draw as draw

from my_classes import AverageMeter, accuracy, init_model_and_dataset
from utils.img_utils import compute_gradient


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
            gradient = compute_gradient(input)
            grad_values = []

            for idx, (i, j) in enumerate(max_out):
                cx, cy = draw.circle_perimeter(i, j, 5, shape=output.shape)
                if idx < 4:
                    grad_values.append(gradient[i - 5:i + 5, j - 5:j + 5].sum())
                    image[:, cx, cy] = 0
                    if idx == 3:
                        image[0, cx, cy] = 1.
                        image[1, cx, cy] = 1.
                    else:
                        corners[idx, cx, cy] = 1.

                else:
                    corners[:, cx, cy] = 1.

            txt = ''
            max_colors = ['1st (red)', '2nd (green)', '3rd (blue)', '4th (yellow)']
            for idx, val in enumerate(grad_values):
                txt += max_colors[idx] + '---> grad = ' + str(val) + '\n'
            fig, ax = plt.subplots(1, 2)
            plt.ioff()
            ax[1].imshow(transforms.ToPILImage()(image))
            plt.scatter(max_out[:, 1], max_out[:, 0], marker='o', c='r', s=2)
            ax[0].imshow(output.cpu().detach().numpy()[0][0], cmap='gray')
            plt.figtext(0.44, 0.25, txt)
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
