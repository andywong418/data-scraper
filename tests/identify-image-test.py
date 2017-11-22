import numpy as np
import keras
from keras.datasets import mnist
# from basis_vectors import resize_data_images,plot_gallery, estimator, form_NMF_matrices, form_dictionary, KMeans, fastICA, sparsePCA, computeFeatureArray, calculateVarianceArray
from keras.models import Sequential
import sys
sys.path.append('/Users/androswong/.virtualenvs/cv/lib/python2.7/site-packages/')
import cv2
import imutils
from imutils import paths
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
K.set_image_dim_ordering('th')
img_width, img_height = 150, 150
model = load_model('open_set_cnn.h5')
neither_images = list(paths.list_images('image_sets/neither'))
results_array = []
for index, images in enumerate(neither_images):
    img = load_img(images,target_size=(img_width, img_height))
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape( (1,) + x.shape )  # this is a Numpy array with shape (1, 3, 150, 150)

    result = model.predict(x)
    result = list(result[0])
    results_array.append(result.index(max(result)))
print results_array
img = load_img('Donald-Trump.jpg',target_size=(img_width, img_height))
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape( (1,) + x.shape )  # this is a Numpy array with shape (1, 3, 150, 150)

result = model.predict(x)
print result
