import numpy as np


class BayesPreprocessor:
    def __init__(self, image):
        # store target image width, height and interpolation
        # inter is an optional argument
        self.image = image
        # self.height = height
        # self.kernel = kernel
        # self.inter = inter

    @staticmethod
    def bayes_noise_removal(self, o):

        # implement image processing
        S1 = self.image
        S2 = 255 - S1
        S3 = S1 + S2
        dv = self.calculate_params(S1, S2, S3, o)

        dv[dv < 0] = 0

        return dv

    def calculate_params(self, S1, S2, S3, o):
        try:
            w = S2 / S3
        except ZeroDivisionError:
            w = 0
        w = np.divide(S2, S3)
        p = np.divide(S3, w)
        try:
            p = S3 / w
        except ZeroDivisionError:
            p = 0

        dv = np.log(np.i0((p * np.sinc(w) * S1) / o ** 2)) + np.log(np.i0((p * np.sinc(1 - w) * S2) / o ** 2)) - p * (
                ((np.sinc(w)) ** 2 + (np.sinc(1 - w)) ** 2) / (2 * o ** 2))
        return dv
