from __future__ import print_function, division

import time
from PIL import Image
from torchvision.transforms import transforms
from transforms.pad_to_square import pad_to_square
import numpy as np

from utils.utils import AverageMeter, accuracy
from utils.img_utils import compute_gradient, save_img, plot_gradient


def test(val_loader, model, save_imgs=False, plt_gradient=False):
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
        output = model(input).split(input.shape[0], dim=0)

        # measure accuracy
        accuracy(corners=corners, output=output[-1].data, target=target, global_recall=eval_recall,
                 global_precision=eval_precision)

        if save_imgs:
            # rgb image
            rgb = Image.open('/home/amitjans/Desktop/Hourglass/data/rgb/' + data['img_name'][0] + '.png')
            rgb = transforms.ToTensor()(rgb)[:, :-1, :-1]
            depth = input[0][0][int((input.shape[2] - rgb.shape[1]) / 2): int((input.shape[2] + rgb.shape[1]) / 2)][
                    int((input.shape[3] - rgb.shape[2]) / 2): int((input.shape[3] + rgb.shape[2]) / 2)]
            rgb = pad_to_square(rgb)
            # gradient plot
            gradient = compute_gradient(depth.cpu().detach().numpy())
            gradient = pad_to_square(gradient.expand(3, -1, -1))[0]
            if plt_gradient:
                plot_gradient(gradient, output[-1].cpu().detach().numpy()[0][0], data['img_name'][0])
            save_img(rgb, output[-1].cpu().detach().numpy()[0][0], gradient, data['img_name'][0])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(' * Recall(%): {top1:.3f}\t' ' * Precision(%):  ({top2:.3f}, {top3:.3f}, {top4:.3f}, {top5:.3f})\t'
          .format(top1=eval_recall.avg * 100, top2=eval_precision[0].avg * 100, top3=eval_precision[1].avg * 100,
                  top4=eval_precision[2].avg * 100, top5=eval_precision[3].avg * 100))

    global_precision = np.array(
        [eval_precision[0].avg, eval_precision[1].avg, eval_precision[2].avg, eval_precision[3].avg])

    return eval_recall.avg, global_precision
