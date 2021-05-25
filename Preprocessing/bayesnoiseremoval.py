import cv2
import numpy as np
from nilearn import plotting
import nibabel as nib
import matplotlib.pyplot as plt

class BayesPreprocessor:
    def __init__(self, image):
        # store target image width, height and interpolation
        # inter is an optional argument
        self.image = image
        # self.height = height
        # self.kernel = kernel
        # self.inter = inter
    @staticmethod
    def bayes_noise_removal(self,o):

        # implement image processing
        S1=self.image
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

# plt.imsave('slice.png',slice_2l)


# cv2.imshow('dw.jpg',slice_2l)
# cv2.waitKey()

img = nib.load('../../mr_train/mr_train_1020_image.nii.gz')
label=nib.load('../../mr_train/mr_train_1001_label.nii.gz')

# display = plotting.plot_anat(img)
# plotting.show()
# plotting.plot_img(img)
# plotting.show()
#
img_data=img.get_fdata()
shape_0=int((img_data.shape[0]-1)/2)
shape_1=int((img_data.shape[1]-1)/2)
shape_2=int((img_data.shape[2]-1)/2)


slice_1=img_data[shape_0,:,:]
image=cv2.imread('../tmp/pred_0.png')
image = cv2.resize(image, (128,512), interpolation=cv2.INTER_AREA)
cv2.imshow('daw',image)
cv2.waitKey(0)
# bayes=BayesPreprocessor(image)

# # image = image / 255
# cv2.imshow('dfdw',bayes.bayes_noise_removal(90))
# # plt.imshow(slice_1)
# # plt.show()
# cv2.waitKey(0)


##img=cv2.imread("mri.png",0);
#
# # plt.figure(1)
# S1=slice_1
# # plt.figure(2)
# S2=255-S1
# S3=S1+S2
#
#
# # plt.figure(3)
# try:
#     w = S2/S3
# except ZeroDivisionError:
#     w = 0
# w=np.divide(S2,S3)
# p=np.divide(S3,w)
# try:
#     p = S3/w
# except ZeroDivisionError:
#     p = 0
#
#
#
# # cv2.imshow("pikseli",np.uint8(p))
# # cv2.imwrite("pikseli.png",np.uint8(p))
#
# o=40
#
# dv=np.log(np.i0((p*np.sinc(w)*S1)/o**2))+np.log(np.i0((p*np.sinc(1-w)*S2)/o**2))-p*(((np.sinc(w))**2+(np.sinc(1-w))**2)/(2*o**2))
# # plt.figure(4)
# dv[dv<0]=0
# cv2.imshow("popravljena.png",np.uint8(dv))
# cv2.waitKey(0)
