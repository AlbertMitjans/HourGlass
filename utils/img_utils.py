import matplotlib.pyplot as plt
import numpy as np
import skimage.draw as draw
import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy import ndimage
from scipy.ndimage.measurements import center_of_mass, label
from skimage.feature import peak_local_max
from pathlib import Path


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
        cx, cy = draw.circle_perimeter(i, j, 9, shape=output.shape)
        if idx < 4:
            grad_values.append(gradient[cx.min():cx.max(), cy.min():cy.max()].sum())
            if idx == 3:
                corners[0, cx, cy] = 1.
                corners[1, cx, cy] = 1.
            else:
                corners[idx, cx, cy] = 1.
        else:
            corners[:, cx, cy] = 1.

    return corners, grad_values, max_coord


def save_img(rgb, output, gradient, name):
    corners, grad_values, max_coord = corner_mask(output, gradient)
    rgb, corners = transforms.ToPILImage()(rgb), transforms.ToPILImage()(corners)
    image = Image.blend(rgb, corners, 0.5)
    plt.ioff()
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches((15, 6))
    ax[2].axis('on')
    ax[2].set_title('RGB image')
    ax[0].axis('off')
    ax[0].set_title('Network\'s output')
    ax[1].axis('off')
    ax[1].set_title('Gradient')
    ax[1].scatter(max_coord[:, 1], max_coord[:, 0], marker='o', c='r', s=0.1)
    ax[0].imshow(output, cmap='afmhot', vmin=0, vmax=1)
    ax[1].imshow(gradient, cmap='gray')
    ax[2].imshow(image)
    txt = ''
    max_colors = ['Red', 'Green', 'Blue', 'Yellow']
    for idx, val in enumerate(grad_values):
        txt += '{color} ---> grad = {top1:.2f}\n'.format(top1=val.item(), color=max_colors[idx])
    plt.figtext(0.46, 0.07, txt, fontsize=10)
    Path('output/').mkdir(parents=True, exist_ok=True)
    plt.savefig('output/{image}.png'.format(image=name))
    plt.close('all')


def show_corners(image, corners):
    """Show image with landmarks"""
    plt.imshow(image, cmap='gray')
    plt.scatter(corners[:, 1], corners[:, 0], s=10, marker='.', c='r')
    plt.pause(0.005)  # pause a bit so that plots are updated
