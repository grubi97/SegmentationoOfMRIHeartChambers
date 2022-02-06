# MRI-Segmentation #

This project was created to train and MRI heart images on UNET.

## Project Structure ##
+ Dataset
+ Preprocessing
+ Unet
+ Utils


The dataset directory contains the dataset and labels split in training and validation sets(currently only the original dataset) and two scripts. The datasetloader.py script loads the images and labels and then uses any preprocessing methods if they are available. It prepares the labels and images for training. The sliceimages.py loads the dataset with nii.gz format and gets the images and saves them.

The preprocessing directory contains preprocessing scripts: preprocessing.py and bayesnoiseremoval.py(not yet implemented)

tmp directory saves the model checkpoints and output test images.

Unet directory contains the model.py script with the implemented unet architecture.

Utils contains a script with help functions during trainong.

train.py contains the training alghoritm.


## Setup ##

```
python3 -m venv unet
source unet/bin/activate
pip install -r requirements.txt
```

## Usage ##

To run the train.py script:
```
 python train.py --images Dataset/train_img --labels Dataset/label_train_img --val_images Dataset/val_img --val_labels Dataset/val_label
```

To run the test.py script:
Download the checkpoint.pth.tar on:https://drive.google.com/file/d/12for4qAjDropcVG4AryMvb5Jss2w_DMa/view?usp=sharing

Place it in the tmp folder.

```
python test.py --val_images Dataset/val_img --val_labels Dataset/val_label
```


## Thesis ##

+ For all information about the dataset and the nature of this work check out the thesis on this repository: Segmentation of heart chambers from 2D MRI images using the U-NET convolutional neural network
