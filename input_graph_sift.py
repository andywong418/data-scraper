from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import os
import json
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
from sklearn.cluster import KMeans
from knn import measure_knn_acc, computeSparseWithLabel
from basis_vectors import KMeans
from sklearn.neighbors import KNeighborsClassifier
K.set_image_dim_ordering('th')
batch_size = 32
nb_classes = 2
nb_epoch = 50
import matplotlib.pyplot as plt
import test-sift
def randomShuffle(src_dir, target_dir):
    src_dir_list = list(paths.list_images(src_dir))
    random.shuffle(src_dir_list)
    print("src dir pre length", len(src_dir_list))
    counter = 0
    #if os.path.exists(target_dir):
     #target_dir_list = list(paths.list_images(target_dir))
     #src_dir_list = list(set(src_dir_list) - set(target_dir_list))
     #print("Src dir post length", len(src_dir_list))
    #else:
    for (i, imagePath) in enumerate(src_dir_list):
        counter+= 1
        if counter > 50:
            break
        dest_file = os.path.join(target_dir, os.path.basename(imagePath))
        shutil.copyfile(imagePath, dest_file)
        src_dir_list.remove(imagePath)
        print("src dir postfor real length", len(src_dir_list))
    return src_dir_list

def initialiseModel(img_rows,img_cols):
    weights_path = './vgg16_weights.h5'
    nb_validation_samples = 402
    nb_epoch = 20
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_rows, img_cols)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
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
    return model
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('val_acc'))
def siftSelection(currentIndex, imagePathArray1, imagePathArray2, label):
    #select remaining array
    setArray = list(paths.list_images('./graph_train_sift_directory/' + label))
    chooseArray = imagePathArray1
    # Add 10 of the best to train_array using a genetic algorithm
    params = [50, 0.05, 1, 10, 10] #These are just some parameters for the GA, defined below in order:
    # [Init pop (pop=25), mut rate (=5%), num generations (5), chromosome/solution length (10), # winners/per gen]
    setArray = np.array(setArray)
    chooseArray = np.array(chooseArray)
    curPop = np.random.choice(chooseArray, size = (params[0], params[3]), replace=False)
    nextPop = np.empty((curPop.shape[0], curPop.shape[1]), dtype=object) #100 by 250
    fitVec = np.zeros((params[0], 2))
    if label == 'cat':
      labelArray = ['cat', 'dog']
    else:
      labelArray = ['dog', 'cat']
    patches = KMeans(list(paths.list_images('./graph_train_sift_directory/cat/')))
    patches = normalize(patches, axis=1, norm='l1')
    cat_validation_image_paths = list(paths.list_images('validation/cat'))
    dog_validation_image_paths = list(paths.list_images('validation/dog'))
    sparseValidationFeatures = [cat_validation_image_paths, dog_validation_image_paths]
    labelValidationArray = ['cat', 'dog']
    # computeSparseWithLabel(cat_validation_image_paths, sparseValidationLabels, 'cat', sparseValidationLabels, patches)
    # computeSparseWithLabel(dog_validation_image_paths, sparseValidationLabels, 'dog', sparseValidationLabels, patches)
    # sparseValidationFeatures = np.array(cat_validation_image_paths + dog_validation_image_paths)
    # sparseValidationLabels = np.array(sparseValidationLabels)

    for i in range(params[2]):
    	fitVec = np.array([np.array([x, measure_sift_acc([setArray.tolist() + curPop[x].tolist(), imagePathArray2], sparseValidationFeatures )]) for x in range(params[0])])
        winners = np.empty((params[4], params[3]), dtype=object)
        for n in range(len(winners)):
            selected = np.random.choice(range(len(fitVec)), params[4]/2, replace=False)
            wnr = np.argmax(fitVec[selected,1])
            winners[n] = curPop[int(fitVec[selected[wnr]][0])].tolist()
            winners[n] = np.array(winners[n])
	nextPop[:len(winners)] = winners
        nextPop[len(winners):] = np.array([np.array(np.random.permutation(np.repeat(winners[:, x], ((params[0] - len(winners))/len(winners)), axis=0))) for x in range(winners.shape[1])]).T #Populate the rest of the generation with offspring of mating pairs
        curPop = nextPop
    best_soln = curPop[np.argmax(fitVec[:,1])]
    return best_soln
def calculateImagesGraph(loops):
    #Initialise plot here first!
    #initialise pre trained vgg model
    # input image dimensions
    img_rows, img_cols = 150, 150
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (10, 10)
    nb_validation_samples = 402
    nb_epoch = 20
    wrapperArray = []
    for j in range(0,loops):
        if os.path.exists('./graph_train_sift_directory'):
           shutil.rmtree('./graph_train_sift_directory')
           os.makedirs('./graph_train_sift_directory/cat/')
           random_list_cat = randomShuffle('./image_sets/cat', './graph_train_sift_directory/cat')
           os.makedirs('./graph_train_sift_directory/dog/')
           random_list_dog = randomShuffle('./image_sets/dog', './graph_train_sift_directory/dog')
        else:
            #initialise 50 datapoints - 25 random dog images and 25 random cat images.
            os.makedirs('./graph_train_sift_directory/cat/')
            random_list_cat = randomShuffle('./image_sets/cat', './graph_train_sift_directory/cat')
            os.makedirs('./graph_train_sift_directory/dog/')
            random_list_dog = randomShuffle('./image_sets/dog', './graph_train_sift_directory/dog')
        samples_axis = []
        accuracy_axis = []
        nb_train_samples = int(math.floor((len(list(paths.list_images('./graph_train_sift_directory/cat'))) + len(list(paths.list_images('./graph_train_sift_directory/dog')))) *0.5))
        total_train_samples = int(math.floor((len(list(paths.list_images('image_sets/cat'))) + len(list(paths.list_images('image_sets/dog')))) *0.5))
        if j==0:
          for i in range(nb_train_samples, 2000, 10):
            training_data_length = len(list(paths.list_images('./graph_train_sift_directory/cat'))) + len(list(paths.list_images('./graph_train_sift_directory/dog')))
            samples_axis.append(training_data_length)
            model = initialiseModel(img_rows, img_cols)
            top_model = Sequential()
            top_model.add(Flatten(input_shape=model.output_shape[1:]))
            top_model.add(Dense(256, activation='relu'))
            top_model.add(Dropout(0.5))
            top_model.add(Dense(1, activation='sigmoid'))

            # note that it is necessary to start with a fully-trained
            # classifier, including the top classifier,
            # in order to successfully do fine-tuning

            #top_model.load_weights(top_model_weights_path)

            # add the model on top of the convolutional base
            model.add(top_model)

            # set the first 25 layers (up to the last conv block)
            # to non-trainable (weights will not be updated)
            for layer in model.layers[:25]:
                layer.trainable = False

            model.compile(loss='binary_crossentropy',
                          optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                          metrics=['accuracy'])

            # prepare data augmentation configuration
            train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)

            test_datagen = ImageDataGenerator(rescale=1./255)
            #
            train_generator = train_datagen.flow_from_directory(
                     './graph_train_sift_directory',
                     target_size=(img_rows, img_cols),
                     batch_size=32,
                     class_mode='binary')
            #
            validation_generator = test_datagen.flow_from_directory(
                     './validation',
                     target_size=(img_rows, img_cols),
                     batch_size=32,
                     class_mode='binary')

            # # fine-tune the model
            loss_history = LossHistory()
            hist = model.fit_generator(
                     train_generator,
                     samples_per_epoch=training_data_length,
                     nb_epoch=nb_epoch,
                     validation_data=validation_generator,
                    nb_val_samples=nb_validation_samples)
            print(hist.history.get('val_acc')[-1])
            accuracy_axis.append(hist.history.get('val_acc')[-1])
            current_acc = hist.history.get('val_acc')[-1]
            with open("input_graph_progress_sift.json", "w") as outfile:
                 json.dump({'accuracy_axis': accuracy_axis, 'samples_axis': samples_axis}, outfile, indent=4)
	    print("CURRENT ACCURACY", current_acc)
            if i < len(random_list_cat):
                #select 10 best KNN
                best_soln = knnSelection(i, random_list_cat, list(paths.list_images('./graph_train_sift_directory/dog/')), 'cat')
                print ("BEST SOLUTIONS", best_soln)
                #select 10 best knn
                for path in best_soln:
                    #if path is not None:
                      dest_file = os.path.join('./graph_train_sift_directory/cat', os.path.basename(path))
                      shutil.copyfile(path, dest_file)
                      pathArr = []
	              with open('cat_train_sift.json', "r") as f:
                        if os.stat("cat_train_sift.json").st_size != 0:
                          sift = json.load(f)
                          pathArr = sift['sift_paths']
                          pathArr.append(path)
                      with open('cat_train_sift.json', "wb") as f:
                       json.dump({'sift_paths': pathArr}, f, indent=4)
                      f.close()
                      random_list_cat.remove(path)
            if i < len(random_list_dog):
                best_soln = knnSelection(i, random_list_dog, list(paths.list_images('./graph_train_sift_directory/cat/')), 'dog')
		print("BEST SOlutions dog", best_soln)
                for path in best_soln:
                   # if path is not None:
                      dest_file = os.path.join('./graph_train_sift_directory/dog', os.path.basename(path))
                      shutil.copyfile(path, dest_file)
                      pathArr = []
                      with open('dog_train_sift.json', "r") as f:
                        if os.stat("dog_train_sift.json").st_size != 0:
                          sift = json.load(f)
                          pathArr  = sift['sift_paths']
                          pathArr.append(path)
                      with open('dog_train_sift.json', "wb") as f:
                       json.dump({'sift_paths': pathArr}, f, indent=4)
                      f.close()
                      random_list_dog.remove(path)
          print("SAMPLES", samples_axis)
          print("ACCURACY", accuracy_axis)
     	  new_array= [samples_axis,accuracy_axis]
    	  wrapperArray.append(new_array)
    for double in wrapperArray:
      plt.plot(double[0], double[1])
    plt.savefig('sift_selection.png')
calculateImagesGraph(2)
