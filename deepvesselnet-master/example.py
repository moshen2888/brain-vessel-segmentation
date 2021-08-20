from dvn import FCN, UNET, VNET
import numpy as np
import dvn.misc as ms
import dvn.losses as ls
import dvn.metrics as mts
import pandas as pd
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import data
import tensorflow as tf

from sklearn.model_selection import train_test_split

from dvn.utils import *

from tensorflow import keras

from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau

import SimpleITK as sitk
from scipy import ndimage
import os
# img_directory = '/content/drive/MyDrive/Project/trainraw/'

# labels_path = '/content/drive/MyDrive/Project/trainseg/'


img_select = []
resize_img = []
label_select = []
not_included = []

# image_files = []
# labels = []

def read_nifti_file(filepath):
    """Read and load volume"""
    # Get raw data
    image_select = get_itk_array(filepath)
    return image_select


def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 64
    desired_height = 64
    # Get current depth
    current_depth = img.shape[0]
    current_width = img.shape[2]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    # img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (depth_factor, height_factor, width_factor), order=1)
    return img


def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    # volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume


def train_preprocessing(train_data):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    train_data = train_data.tolist()
    for i in range(len(train_data)):
        train_data[i] = np.array(train_data[i], ndmin = 4)
    return np.array(train_data)


class My_Custom_Generator(keras.utils.Sequence) :
  
    def __init__(self, image_filenames, labels_filenames, batch_size) :
      self.image_filenames = image_filenames
      self.labels_filenames = labels_filenames
      self.batch_size = batch_size
      
      
    def __len__(self) :
      return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(int)
    
    
    def __getitem__(self, idx) :
      batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
      batch_y = self.labels_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
      image_scans = np.array([process_scan('/nobackup/ml20t2w/Data/trainraw/' + str(file_name))
                 for file_name in batch_x])
      image_scans = normalize(image_scans)
      label_scans = np.array([process_scan('/nobackup/ml20t2w/Data/trainseg/' + str(file_name))
                 for file_name in batch_y])
      x_train = train_preprocessing(image_scans)
      y_train = train_preprocessing(label_scans)
      x_train = normalize(x_train)
      Y = np.array(y_train)
      Y = np.squeeze(Y)
      Y = ms.to_one_hot(Y)
      Y = np.transpose(Y, axes=[0,dim+1] + list(range(1,dim+1)))
      return x_train, Y

file_names = os.listdir("/nobackup/ml20t2w/Data/trainraw")
file_names_train = file_names[:116]
label_names = os.listdir("/nobackup/ml20t2w/Data/trainseg")
label_names_train = label_names[:116]

model_path = '/home/home02/ml20t2w/Project/deepvesselnet-master/fcn_model.pb'

epochs = 5

batch_size = 2

dim = 3
net = FCN(cross_hair=True, dim=dim)
if os.path.exists(model_path):
    net = net.load(model_path)
    print("checkpoint_loaded")
opt = Adam(lr=0.01)
#opt = SGD(lr=0.01, momentum=0.9, decay=1e-2/epochs)
net.compile(loss=ls.weighted_categorical_crossentropy_with_fpr(), optimizer=opt)
# model_checkpoint = ModelCheckpoint(model_path, monitor='loss',verbose=1, save_best_only=True,save_weights_only=False,mode='min',period=epochs)


# reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.00001)
# print(X.shape)
# print(Y.shape)
print('Testing FCN Network')
#print('Data Information => ', 'volume size:', X.shape, ' labels:',np.unique(Y))
#net.fit(x=X, y=Y, epochs=epochs, batch_size=10, shuffle=True)
net.fit_generator(generator=My_Custom_Generator(file_names_train, label_names_train, batch_size), epochs=epochs, steps_per_epoch=int(116 // batch_size))

net.save(model_path)





