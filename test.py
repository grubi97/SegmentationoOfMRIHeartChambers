import torch
from Unet.model import UNET
from imutils import paths
from Dataset.datasetloader import MRIDataset
from Utils import utils
from torch.utils.data import DataLoader
from Preprocessing.preprocessing import Preprocessor
import argparse

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ap = argparse.ArgumentParser()
ap.add_argument("-vi", "--val_images", required=True, help="path to images directory")
ap.add_argument("-vl", "--val_labels", required=True, help="path to labels directory")
args = vars(ap.parse_args())
val_imagePath = list(paths.list_images(args["val_images"]))
val_labelPath = list(paths.list_images(args["val_labels"]))
prep = Preprocessor(64, 64)

dl_val = DataLoader(MRIDataset(imgpath=val_imagePath, labelpath=val_labelPath, preprocessors=[prep], verbose=5),
                    batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

model = UNET(in_channels=1, out_channels=1).to(DEVICE).float()

utils.load_checkpoint(torch.load('tmp/checkpoint.pth.tar'), model)
utils.save_predictions_as_imgs(dl_val, model, DEVICE)

