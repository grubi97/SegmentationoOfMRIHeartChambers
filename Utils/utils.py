import torch
import torchvision
import os
import wandb


def save_checkpoint(state, filename='tmp/checkpoint.pth.tar'):
    print('[INFO] Saving checkpoint')
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print('[INFO] Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])


def check_accuracy(loader, model, device):
    num_correct = 0
    num_pixels = 0
    dice_score = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            preds = (preds > 0.5).float()

            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-9)
            print('Got {}/{} with acc {:2f}'.format(num_correct, num_pixels, num_correct / num_pixels * 100))
            print('Dice score {}'.format(dice_score / len(loader)))
            wandb.log({"dice": dice_score})
            wandb.log({"acc": (num_correct, num_pixels, num_correct / num_pixels * 100)})

            model.train()


def save_predictions_as_imgs(loader, model, device, folder='tmp/'):
    model.eval()

    for idx, (x, y) in enumerate(loader):
        x = x.to(device)

        with torch.no_grad():
            preds = model(x)
            preds = (preds > 0.5).float()

        torchvision.utils.save_image(preds, os.path.join(folder, 'pred_{}.png'.format(idx)))
        # torchvision.utils.save_image(y, os.path.join(folder, '{}.png'.format(idx)))

    model.train()
