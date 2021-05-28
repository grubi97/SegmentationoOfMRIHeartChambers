import torch
from Unet.model import UNET
from Utils import utils
import torchvision
from Preprocessing.preprocessing import Preprocessor
import cv2
import os

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = UNET(in_channels=1, out_channels=1).to(DEVICE).float()
utils.load_checkpoint(torch.load('tmp/checkpoint.pth.tar', map_location=torch.device('cpu')), model)
model.eval()

prep = Preprocessor(128, 128)

for fileA in os.listdir('Dataset/zdim/valimagesbayes'):
    with torch.no_grad():
        x = cv2.imread('Dataset/zdim/valimagesbayes/'+fileA, 0)
        x = prep.preprocess(x)
        x = torch.from_numpy(x).float()
        x = x.unsqueeze(0).unsqueeze(0)
        x = x.to(DEVICE)
        preds = model(x)
        # preds[preds<0.5]=0
        # preds = torch.sigmoid(preds)
        torchvision.utils.save_image(preds, './resultsz/'+fileA)
