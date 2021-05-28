import os, glob
import nibabel as nib
import numpy as np
import cv2
from bayesnoiseremoval import BayesPreprocessor as bp

# STEP 1 - Load and visualize data
imgPath = '../../mr_train/'
maskPath = '../../train_labels/'

imagePathInput = os.path.join(imgPath)
maskPathInput = os.path.join(maskPath)

imgOutput = './zdim/imagesbayes/'
maskOutput = './zdim/masks/'
imageSliceOutput = os.path.join(imgOutput)
maskSliceOutput = os.path.join(maskOutput)

# STEP 2 - Image normalization
HOUNSFIELD_MIN = -1000
HOUNSFIELD_MAX = 2000
HOUNSFIELD_RANGE = HOUNSFIELD_MAX - HOUNSFIELD_MIN

# STEP 3 - Slicing and saving
SLICE_X = False
SLICE_Y = False
SLICE_Z = True

SLICE_DECIMATE_IDENTIFIER = 3


# Normalize image
def normalizeImageIntensityRange(img):
    img[img < HOUNSFIELD_MIN] = HOUNSFIELD_MIN
    img[img > HOUNSFIELD_MAX] = HOUNSFIELD_MAX
    return (img - HOUNSFIELD_MIN) / HOUNSFIELD_RANGE


def readImageVolume(imgPath, normalize=False):
    img = nib.load(imgPath).get_fdata()
    if normalize:
        return normalizeImageIntensityRange(img)
    else:
        return img


# Save volume slice to file

def saveSlice(img, fname, path):
    # img = np.uint8(img * 255)
    fout = os.path.join(path, f'{fname}.png')
    cv2.imwrite(fout, img)
    print(f'[+] Slice saved: {fout}', end='\r')


def sliceAndSaveVolumeImage(vol, fname, path):
    (dimx, dimy, dimz) = vol.shape
    print(dimx, dimy, dimz)
    cnt = 0
    if SLICE_X:
        cnt += dimx
        print('Slicing X: ')
        for i in range(dimx):
            saveSlice(bp.bayes_noise_removal(image=vol[i, :, :], o=140),
                      fname + f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_x', path)

    if SLICE_Y:
        cnt += dimy
        print('Slicing Y: ')
        for i in range(dimy):
            saveSlice(bp.bayes_noise_removal(image=vol[:, i, :], o=140),
                      fname + f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_y', path)

    if SLICE_Z:
        cnt += dimz
        print('Slicing Z: ')
        for i in range(dimz):
            saveSlice(bp.bayes_noise_removal(image=vol[:, :, i], o=140),
                      fname + f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_z', path)
    return cnt


for index, filename in enumerate(sorted(glob.iglob(imagePathInput + '*.nii.gz'))):
    img = readImageVolume(filename)
    print(filename, img.shape, np.sum(img.shape), np.min(img), np.max(img))
    numOfSlices = sliceAndSaveVolumeImage(img, 'heart' + str(index), imageSliceOutput)
    print(f'\n{filename}, {numOfSlices} slices created \n')

# # Read and process image mask volumes
# for index, filename in enumerate(sorted(glob.iglob(maskPathInput + '*.nii'))):
#     img = readImageVolume(filename, False)
#     print(filename, img.shape, np.sum(img.shape), np.min(img), np.max(img))
#     numOfSlices = sliceAndSaveVolumeImage(img, 'tooth' + str(index), maskSliceOutput)
#     print(f'\n{filename}, {numOfSlices} slices created \n')
