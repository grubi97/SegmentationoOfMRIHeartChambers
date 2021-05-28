import torch
from tqdm import tqdm
import torch.nn  as nn
import torch.optim as optim
from Unet.model import UNET
from imutils import paths
from Dataset.datasetloader import MRIDataset
from Utils import utils
from torch.utils.data import DataLoader
from Preprocessing.preprocessing import Preprocessor
import argparse
import wandb


# Hyperparam etc.
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
NUM_EPOCHS = 100
LOAD_MODEL = False
NUM_WORKERS = 0


wandb.init(project='Unet', entity='grubi')
config = wandb.config
config.learning_rate = LEARNING_RATE


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)


    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)


        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        wandb.log({"loss": loss})




ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to images directory")
ap.add_argument("-l", "--labels", required=True, help="path to labels directory")
ap.add_argument("-vi", "--val_images", required=True, help="path to images directory")
ap.add_argument("-vl", "--val_labels", required=True, help="path to labels directory")
args = vars(ap.parse_args())

print("[INFO] loading images and labels...")
imagePath = list(paths.list_images(args["images"]))
labelPath = list(paths.list_images(args["labels"]))
val_imagePath = list(paths.list_images(args["val_images"]))
val_labelPath = list(paths.list_images(args["val_labels"]))

prep = Preprocessor(128, 256)
dl = DataLoader(MRIDataset(imgpath=imagePath, labelpath=labelPath, preprocessors=[prep], verbose=200),
                batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
dl_val = DataLoader(MRIDataset(imgpath=val_imagePath, labelpath=val_labelPath, preprocessors=[prep], verbose=200),
                    batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

model = UNET(in_channels=1, out_channels=1).to(DEVICE).float()
loss_fnc = nn.BCEWithLogitsLoss()
optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)

if LOAD_MODEL:
    utils.load_checkpoint(torch.load('tmp/checkpoint.pth.tar'), model)

scaler = torch.cuda.amp.GradScaler()

wandb.watch(model)
for epoch in range(NUM_EPOCHS):
    print(epoch+1)

    train_fn(dl, model, optimizer, loss_fnc, scaler)

    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    utils.save_checkpoint(checkpoint,filename='tmp/checkpoint.pth.tar')
    utils.check_accuracy(dl_val, model, DEVICE)
    utils.save_predictions_as_imgs(dl_val, model, DEVICE)
