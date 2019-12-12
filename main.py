import torch
import torch.utils.data
from my_classes import init_model_and_dataset

from train import train
from validation import validate

from scipy import ndimage


def main(train_flag, evaluate_val, save_imgs):
    normalize_data = False
    depth = True
    ckpt = None#'checkpoint/hg_ckpt_97.pth'
    end_epoch = 200

    if train_flag:
        freeze = False
        train(ckpt, freeze, depth, normalize_data, end_epoch)

    if not train_flag:
        batch_size = 1
        num_workers = 0
        directory = '/home/amitjans/Desktop/Hourglass/data/'

        model, train_dataset, val_dataset, _, _ = init_model_and_dataset(depth, directory, normalize_data)

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

        validate(val_loader, model, end_epoch, save_imgs=save_imgs, plt_gradient=plt_gradient)


if __name__ == "__main__":
    train_flag = True
    evaluate_val = True
    save_imgs = True
    plt_gradient = False

    main(train_flag, evaluate_val, save_imgs)
