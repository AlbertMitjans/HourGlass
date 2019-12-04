from torchvision.transforms import RandomCrop, ToPILImage, ToTensor
import matplotlib.pyplot as plt
import numpy as np
from my_classes import gaussian
from PIL import Image
import torch

def see(images, corners):
    path = "/home/amitjans/Desktop/Hourglass/data/depth/"
    for i, corner in enumerate(corners):
        try: im = Image.open(path + images[i])
        except FileNotFoundError: continue
        corner = np.array(corner)
        corner = corner.astype(int).reshape(-1, 2)
        grid_0 = ToTensor()(gaussian(im, corner)[0])
        grid = ToPILImage()(grid_0.type(torch.float32)/grid_0.max())
        rc = RandomCrop((int(im.size[1]*0.8), int(im.size[0]*0.8)))
        im_cropped = rc(grid)
        im_resize = im_cropped.resize((im.size[0], im.size[1]))
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(grid_0[0], cmap='gray')
        ax[1].imshow(im_cropped, cmap='gray')
        ax[2].imshow(im_resize, cmap='gray')
        plt.show()
        plt.waitforbuttonpress()
        plt.close('all')
