from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import os
import keras
from keras.datasets import mnist
# from basis_vectors import resize_data_images,plot_gallery, estimator, form_NMF_matrices, form_dictionary, KMeans, fastICA, sparsePCA, computeFeatureArray, calculateVarianceArray
from keras.models import Sequential
import sys
sys.path.append('/Users/androswong/.virtualenvs/cv/lib/python2.7/site-packages/')
import cv2
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras import backend as K
from keras import optimizers
import imutils
from imutils import paths
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import h5py
import random
import shutil
import math
import theano
from scipy.misc import imsave
import time
from sklearn.cluster import KMeans
batch_size = 32
nb_classes = 3
nb_epoch = 50
import matplotlib.pyplot as plt

# input image dimensions
img_rows, img_cols = 150, 150
img_width, img_height = 150, 150
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (10, 10)

# the data, shuffled and split between train and test sets
cat_image_paths = list(paths.list_images('image_sets/cat'))
dog_image_paths = list(paths.list_images('image_sets/dog'))
cat_validation_image_paths = list(paths.list_images('validation/cat'))
dog_validation_image_paths = list(paths.list_images('validation/dog'))

# patches = KMeans(dog_image_paths)
# patches = normalize(patches, axis=1, norm='l1')
# print("patches shape before", patches.shape)
# patches = patches.reshape(32,3,3,3)
# print("HELLO")
# print("patches shape", patches.shape)
# test_image = cv2.imread('./image_sets/cat/image_432_unlabelled.jpg')
# print("TEST IMAGE", test_image)

import numpy.ma as ma

# utility functions
from mpl_toolkits.axes_grid1 import make_axes_locatable

def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    pl.colorbar(im, cax=cax)

def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]

    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                            dtype=np.float32)

    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in xrange(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols

        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def checkImages(image_directory):
    for(i, imagePath) in enumerate(image_directory):
        image = cv2.imread(imagePath)
        if not image is None and len(image) > 0:
            print("okay")
        else:
            print("IMAG EPATH", imagePath)
            os.remove(imagePath)

# checkImages(cat_image_paths)
# checkImages(dog_image_paths)
weights_path = './vgg16_weights.h5'
top_model_weights_path = 'top_model_weights.h5'
nb_validation_samples = 402
nb_epoch = 50
nb_train_samples = len(list(paths.list_images('image_sets/cat'))) + len(list(paths.list_images('image_sets/dog'))) + len(list(paths.list_images('image_sets/neither')))

def save_pre_trained_features():
    datagen = ImageDataGenerator(rescale=1./255)
    #build VGG network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_rows, img_cols)))

    model.add(Convolution2D(64,3,3, activation = 'relu', name = 'conv1_v1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64,3,3,activation= 'relu',name='conv1_v2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # load the weights of the VGG16 networks
    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path)
    print("F", f)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')
    train_generator = datagen.flow_from_directory(
            './image_sets',  # this is the target directory
            target_size=(150, 150),  # all images will be resized to 150x150
            batch_size=32,
            class_mode=None,
            shuffle = False)
    bottleneck_features_train = model.predict_generator(train_generator, nb_train_samples)
    np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)
    validation_generator = datagen.flow_from_directory(
            './validation',
            target_size=(150, 150),
            batch_size=32,
            class_mode=None,
            shuffle = False)
    bottleneck_features_validation = model.predict_generator(validation_generator, 402)
    np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.array([0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array([0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))
    print("training features input")
    print(train_data.shape)
    TwoDim_dataset = train_data.reshape(nb_train_samples,-1)
    reduced_data = PCA(n_components=4).fit_transform(TwoDim_dataset)
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(6, 4))
        for lab, col in zip((0, 1),
                            ('blue', 'red')):
            plt.scatter(reduced_data[train_labels==lab, 0],
                        reduced_data[train_labels==lab, 1],
                        label=lab,
                        c=col)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(loc='lower center')
        plt.tight_layout()
        plt.show()
    print("DATA U SHAPE", reduced_data)
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              nb_epoch=nb_epoch, batch_size=32,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)

def initialiseModel(img_rows,img_cols):
    weights_path = './vgg16_weights.h5'
    nb_validation_samples = 402
    nb_epoch = 20
    model = Sequential()
    conv_layers = []
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_rows, img_cols)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    # convout1_f = theano.function([model.get_input(train=False)], conv1_1.get_output(train=False))
    # conv_layers.append(convout1_f)
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    # convout2_f = theano.function([model.get_input(train=False)], conv1_2.get_output(train=False))
    # conv_layers.append(convout2_f)
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    # convout3_f = theano.function([model.get_input(train=False)], conv2_1.get_output(train=False))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    # convout4_f = theano.function([model.get_input(train=False)], conv1_2.get_output(train=False))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, name='conv5_3'))
    convout_final = Activation('relu')
    model.add(convout_final)

    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')
    return model, convout_final
def initialise_top_model():

    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    # conv_out_dense = theano.function([model.get_input(train=False)], dense_1.get_output(train=False))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(nb_classes, activation='softmax'))
    return top_model
def initialise_one_class_detector(X_train):
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(X_train)
def initialise_isolation_forest_detector(X_train)

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

# add the model on top of the convolutional base
def run_cnn(model, top_model, train_data_dir, validation_data_dir):
    model.add(top_model)
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer_name = 'conv5_1'
    filter_index = 0  # can be any integer from 0 to 511, as there are 512 filters in that layer

    # build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    kept_filters = []
    for filter_index in range(0, 20):
        # we only scan through the first 20 filters,
        # but there are actually 512 of them
        print('Processing filter %d' % filter_index)
        start_time = time.time()

        # we build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        layer_output = layer_dict[layer_name].output
        if K.image_dim_ordering() == 'th':
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_index])

        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # step size for gradient ascent
        step = 1.

        # we start from a gray image with some random noise
        if K.image_dim_ordering() == 'th':
            input_img_data = np.random.random((1, 3, img_width, img_height))
        else:
            input_img_data = np.random.random((1, img_width, img_height, 3))
        input_img_data = (input_img_data - 0.5) * 20 + 150

        # we run gradient ascent for 20 steps
        for i in range(20):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

            print('Current loss value:', loss_value)
            if loss_value <= 0.:
                # some filters get stuck to 0, we can skip them
                break

        # decode the resulting input image
        if loss_value > 0:
            img = deprocess_image(input_img_data[0])
            kept_filters.append((img, loss_value))
        end_time = time.time()
        print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

    # we will stich the best 64 filters on a 8 x 8 grid.
    n = 3

    # the filters that have the highest loss are assumed to be better-looking.
    # we will only keep the top 64 filters.
    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[:n * n]

    # build a black picture with enough space for
    # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    margin = 5
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            img, loss = kept_filters[i * n + j]
            stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                             (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

    # save the result to disk
    imsave('stitched_filters_%dx%d_%s.png' % (n, n, layer_name), stitched_filters)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:25]:
        layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.

    # model.compile(loss='categorical_crossentropy',
    #               optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
    #               metrics=['accuracy'])
    #
    # # prepare data augmentation configuration
    # train_datagen = ImageDataGenerator(
    #         rescale=1./255,
    #         shear_range=0.2,
    #         zoom_range=0.2,
    #         horizontal_flip=True)
    #
    # test_datagen = ImageDataGenerator(rescale=1./255)
    #
    # train_generator = train_datagen.flow_from_directory(
    #         train_data_dir,
    #         target_size=(img_rows, img_cols),
    #         batch_size=32)
    #
    # validation_generator = test_datagen.flow_from_directory(
    #         validation_data_dir,
    #         target_size=(img_rows, img_cols),
    #         batch_size=32)
    #
    # # fine-tune the model
    # model.fit_generator(
    #         train_generator,
    #         samples_per_epoch=nb_train_samples,
    #         nb_epoch=nb_epoch,
    #         validation_data=validation_generator,
    #         nb_val_samples=nb_validation_samples)

model, convout_final = initialiseModel(150,150)
top_model = initialise_top_model()
run_cnn(model, top_model, './image_sets', './validation')


#
#
X_train = []
y_train = []
# X_test = []
# y_test = []
# filePaths = []
def appendImagePixels(image_directory, label,featureArray, labelArray, removeBool):
    for (i, imagePath) in enumerate(image_directory):
        image = cv2.imread(imagePath)
        label = label
        if not image is None:
            if removeBool[0]:
                removeBool[1].append(imagePath)
            print(image.shape)
            pixels = cv2.resize(image, (128, 128))
            print("pixels shape", pixels.shape)
            featureArray.append(pixels)
            labelArray.append(label)
# #cat is 0 , dog is 1
# appendImagePixels(cat_validation_image_paths, 0, X_test, y_test, [False])
# appendImagePixels(dog_validation_image_paths, 1, X_test, y_test, [False])

# appendImagePixels(dog_image_paths, 1, X_train, y_train, [True,filePaths])
#
# X_train= np.array(X_train)
# y_train = np.array(y_train)
# X_test = np.array(X_test)
# y_test = np.array(y_test)
#
# # (X_train, y_train), (X_test, y_test) = mnist.load_data()
# if K.image_dim_ordering() == 'th':
#     print("X TRAIN SHAPE", X_train.shape)
#     X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
#     X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
#     input_shape = (3, img_rows, img_cols)
# else:
#     X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
#     X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
#     input_shape = (img_rows, img_cols, 3)
#
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255
# print('X_train shape:', X_train.shape)
# print(X_train.shape[0], 'train samples')
# print(X_test.shape[0], 'test samples')
#
# # convert class vectors to binary class matrices
# Y_train = y_train.reshape((-1, 1))
# Y_test = y_test


# bias = [0] * nb_filters
# bias= np.array(bias)
# model = Sequential()
# # without ,
# weights = (patches, bias)
# model.add(Convolution2D(32, 3, 3, input_shape=(3, 150, 150), weights = weights))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Convolution2D(32, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Convolution2D(64, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# # the model so far outputs 3D feature maps (height, width, features)
#
# model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1))
# model.add(Activation('sigmoid'))
#
# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])
# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)
#
# # this is the augmentation configuration we will use for testing:
# # only rescaling
# test_datagen = ImageDataGenerator(rescale=1./255)
# # this is a generator that will read pictures found in
# # subfolers of 'data/train', and indefinitely generate
# # batches of augmented image data
# train_generator = train_datagen.flow_from_directory(
#         './image_sets',  # this is the target directory
#         target_size=(150, 150),  # all images will be resized to 150x150
#         batch_size=32,
#         class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels
# validation_generator = test_datagen.flow_from_directory(
#         './validation',
#         target_size=(150, 150),
#         batch_size=32,
#         class_mode='binary')
#
# model.fit_generator(
#         train_generator,
#         samples_per_epoch=1896,
#         nb_epoch=50,
#         validation_data=validation_generator,
#         nb_val_samples=402)
# model.save_weights('first_try.h5')  # always save your weights after training or during training

# save_pre_trained_features()
# train_top_model()
