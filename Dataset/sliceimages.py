##Script to get all the slices form labels and images, seperate them and save in them seperate folders
import nibabel as nb
import argparse
import os
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True, help="path to dataset directory")
args = vars(ap.parse_args())


def get_slices(img_data):
    n_i, n_j, n_k = img_data.shape
    center_i = (n_i - 1) // 2
    center_j = (n_j - 1) // 2
    center_k = (n_k - 1) // 2

    slice_0 = img_data[center_i, :, :]
    slice_1 = img_data[:, center_j, :]
    slice_2 = img_data[:, :, center_k]

    slices = [slice_0, slice_1, slice_2]

    return slices


for subdir, dirs, files in os.walk(args["dataset"]):
    for file in files:
        img = nb.load(os.path.join(subdir, file)).get_fdata()
        for ind, x in enumerate(get_slices(img)):
            if 'label' in file:
                plt.imsave('label_train_img/' + file[:19] + str(ind) + '.png', x)
            else:
                plt.imsave('train_img/' + file[:19] + str(ind) + '.png', x)
