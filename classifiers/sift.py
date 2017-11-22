import numpy as np
import argparse
import sys
sys.path.append('/data/greyostrich/not-backed-up/aims/aimsre/awong/OpenCV-Install/lib')
import cv2
import glob, os
import imutils
from imutils import paths
import shutil
from scipy.cluster.vq import *
# Importing the library which classifies set of observations into clusters
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
from sklearn.svm import SVC

def populate_data(path, image_paths, image_labels):
    names = os.listdir(path)
    class_id = 0
    for name in names:  # Iterate over the training_names list
        class_dir = os.path.join(path, name)
        class_path = list(paths.list_images(class_dir))
        image_paths+=class_path
        image_labels+=[class_id]*len(class_path)
        class_id+=1

# Reading the image and calculating the features and corresponding descriptors
def sift_features_list(image_paths, image_labels, BOW):
    for i, image_path in enumerate(image_paths):
        im = cv2.imread(image_path, 0)
        if not im is None:
            kpts, des = sift.detectAndCompute(im, None) # Computing the keypoints
            if not des is None:
              BOW.add(des)
            else:
              del image_paths[i]
              del image_labels[i]
        else:
            os.remove(image_path)
            del image_paths[i]
            del image_labels[i]

def sift_features_list_train(image_paths,buffer_image_labels, BOW,image_labels):
    for i, image_path in enumerate(image_paths):
        im = cv2.imread(image_path, 0)
        label = buffer_image_labels[i]
        if not im is None:
            kpts, des = sift.detectAndCompute(im, None) # Computing the keypoints
            if not des is None:
              BOW.add(des)
              image_labels.append(label)
            else:
              del image_paths[i]
        else:
            os.remove(image_path)
            del image_paths[i]

def feature_extract(pth, bowDiction):
    im = cv2.imread(pth, 0)
    return bowDiction.compute(im, sift.detect(im))


def computeHistogram(trial_im_features,bowDiction, bow, voc):
    for i in xrange(len(des_list)):
        trial_im_features.extend(feature_extract(des_list[i][0], bowDiction))


def measure_sift_acc(train_path, validation_path):
    train_image_paths = []  # Inilialising the list
    train_image_labels = []  # Inilialising the list
    buffer_train_image_labels = []
    validation_image_paths = []
    validation_image_labels = []
    trial_buffer = []
    trial_buffer_labels=[]
    populate_data(train_path, trial_buffer, trial_buffer_labels)
    populate_data(train_path,train_image_paths, train_image_labels)
    populate_data(validation_path, validation_image_paths, validation_image_labels)

    sift = cv2.xfeatures2d.SURF_create(300)
    train_descriptor_list = []
    validation_descriptor_list = []
    dictionarySize = 5
    train_BOW = cv2.BOWKMeansTrainer(dictionarySize)
    validation_BOW = cv2.BOWKMeansTrainer(dictionarySize)
    train_labels =[]
    sift_features_list_train(trial_buffer, trial_buffer_labels,train_BOW, train_labels)
    sift_features_list(validation_image_paths, validation_image_labels, validation_BOW)
    print("train images length check", len(train_image_paths))
    print('train labels check', len(train_image_labels))
    train_dictionary = train_BOW.cluster()
    validation_dictionary = validation_BOW.cluster()
    train_dictionary = train_BOW.cluster()
    validation_dictionary = validation_BOW.cluster()
    # Stack all the descriptors vertically in a numpy array
    train_bow_dict = cv2.BOWImgDescriptorExtractor(sift, cv2.BFMatcher(cv2.NORM_L2))
    train_bow_dict.setVocabulary(train_dictionary)
    val_bow_dict = cv2.BOWImgDescriptorExtractor(sift, cv2.BFMatcher(cv2.NORM_L2))
    val_bow_dict.setVocabulary(validation_dictionary)


    train_image_features= []
    validation_image_features= []
    computeHistogram(train_image_features, train_bow_dict, train_BOW,  voc)
    computeHistogram(validation_image_features, val_bow_dict, validation_BOW, voc)

    model = KNeighborsClassifier(n_neighbors= 5,
        n_jobs= 1)
    model.fit(np.array(train_image_features), np.array(rain_labels))
    acc = model.score(np.array(validation_image_features), np.array(validation_image_labels))
    clf = SVC()
    clf.fit(np.array(train_image_features), np.array(trial_train_labels))
    svm_acc = clf.score(np.array(validation_image_features), np.array(validation_image_labels))
    return max(acc, svm_acc)
print("final result", measure_sift_acc('./image_sets', './validation'))
