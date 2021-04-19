import numpy as np
import cv2
from torch.utils.data import Dataset
import torch
import random
IGNORED = ['.DS_Store']


class MRIDataset(Dataset):
    def __init__(self, imgpath, labelpath, preprocessors=None, verbose=-1):
        super(MRIDataset, self).__init__()
        # store the image preprocessor
        self.preprocessors = preprocessors
        self.imgpath = imgpath
        self.labelpath = labelpath

        # if the preprocessors are None, initialize them as an
        # empty list
        if self.preprocessors is None:
            self.preprocessors = []

        self.images = []
        self.masks = []

        for (i, path) in enumerate(self.imgpath):
            image = cv2.imread(path,0)

            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)

            image = torch.from_numpy(image)
            image = image.unsqueeze(0)

            self.images.append(image)

            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(path)))

        for (i, path) in enumerate(self.labelpath):
            label = cv2.imread(path)

            if self.preprocessors is not None:
                for p in self.preprocessors:
                    label = p.preprocess(label)
            label = np.sum(label, axis=2)
            label = label > 0.5
            label = torch.from_numpy(label)
            label = label.unsqueeze(0)

            self.masks.append(label)

            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(path)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = self.images[idx]
        mask = self.masks[idx]
        #
        # Flip image for data augmentation
        # if random.random() > 0.5:
        #     image = torch.flip(image, [0])
        #     mask = torch.flip(mask, [0])

        return image.float(), mask.float()

