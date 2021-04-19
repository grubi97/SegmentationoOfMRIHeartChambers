import cv2
import torch


class Preprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        image = image / 255
        image = cv2.resize(image, (self.width, self.height), interpolation=self.inter)
        # image = torch.from_numpy(image)
        return image
