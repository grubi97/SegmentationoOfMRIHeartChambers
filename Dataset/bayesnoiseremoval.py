import numpy as np


class BayesPreprocessor:

    @staticmethod
    def bayes_noise_removal(image, o):

        # implement image processing
        S1 = image
        S2 = 255 - S1
        S3 = S1 + S2
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

        dv[dv < 0] = 0

        return dv
