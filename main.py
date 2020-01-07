import torch
import torch.utils.data
from utils.utils import init_model_and_dataset

from train import train
from test import test


def main(train_flag, evaluate_val, save_imgs):
    depth = True
    ckpt = None #'checkpoint/hg_ckpt_82.pth'
    end_epoch = 200

    if train_flag:
        freeze = False
        train(ckpt, freeze, depth, end_epoch)

    if not train_flag:
        batch_size = 1
        num_workers = 0
        directory = '/home/amitjans/Desktop/Hourglass/data/'

        model, train_dataset, val_dataset, _, _ = init_model_and_dataset(depth, directory)

        train_dataset.evaluate()
        val_dataset.evaluate()

        if evaluate_val:
            transformed_dataset = val_dataset
        if not evaluate_val:
            transformed_dataset = torch.utils.data.ConcatDataset((train_dataset, val_dataset))

        # load the pretrained network
        if ckpt is not None:
            checkpoint = torch.load(ckpt)

            model.load_state_dict(checkpoint['model_state_dict'])

        val_loader = torch.utils.data.DataLoader(transformed_dataset, batch_size=batch_size, shuffle=False,
                                                 num_workers=num_workers, pin_memory=True)

        test(val_loader, model, end_epoch, save_imgs=save_imgs, plt_gradient=plt_gradient)


if __name__ == "__main__":
    train_flag = True
    evaluate_val = True
    save_imgs = True
    plt_gradient = True

    main(train_flag, evaluate_val, save_imgs)
