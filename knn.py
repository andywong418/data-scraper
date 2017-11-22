from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
# from imutils import paths
import numpy as np
import argparse
import sys
sys.path.append('/Users/androswong/.virtualenvs/cv/lib/python2.7/site-packages/')
import cv2
import glob, os
import imutils
from imutils import paths
import shutil
import ntpath
from numpy import genfromtxt
from scipy import signal
from basis_vectors import resize_data_images,plot_gallery, estimator, form_NMF_matrices, form_dictionary, KMeans, fastICA, sparsePCA, computeFeatureArray, calculateVarianceArray
my_data = genfromtxt('W_matrices.csv', delimiter=',')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from scipy import misc
import scipy.misc
from sklearn.preprocessing import normalize

def convolveSignal(image_signal, patch):
    return signal.convolve2d(image_signal, patch, mode='full', boundary='fill', fillvalue=0)
def convert_img_to_feature_vector(image, size =(32, 32)):
    #flatten image to become a 1-d vector
    return cv2.resize(image, size).flatten()

#Find power of our filtered images
def extract_color_histogram(image, bins=(8, 8, 8)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,[0, 180, 0, 256, 0, 256])
	# handle normalizing the histogram if we are using OpenCV 2.4.X
	if imutils.is_cv2():
		hist = cv2.normalize(hist)

	# otherwise, perform "in place" normalization in OpenCV 3 (I
	# personally hate the way this is done
	else:
		cv2.normalize(hist, hist)

	# return the flattened histogram as the feature vector
	return hist.flatten()
def appendImagePixels(image_directory, label,featureArray, labelArray, removeBool):
    for (i, imagePath) in enumerate(image_directory):
        image = cv2.imread(imagePath)
        label = label
        if not image is None:
            if removeBool[0]:
                removeBool[1].append(imagePath)
            pixels = convert_img_to_feature_vector(image)
            hist = extract_color_histogram(image)
            # if i ==1:
            #     print imagePath
                # plt.subplot(1,3,1)
                # img = cv2.imread(imagePath)
                # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            	# hist = cv2.calcHist([hsv], [0, 1, 2], None, (8,8,8),[0, 180, 0, 256, 0, 256])
                # plt.imshow(hist,interpolation = 'nearest')
                # plt.show()
                # img = scipy.misc.imread(imagePath)
                # array=np.asarray(img)
                # arr=(array.astype(float))/255.0
                # img_hsv = colors.rgb_to_hsv(arr[...,:3])
                #
                # lu1=img_hsv[...,0].flatten()
                # plt.subplot(1,3,1)
                # plt.hist(lu1*360,bins=360,range=(0.0,360.0),histtype='stepfilled', color='r', label='Hue')
                # plt.title("Hue")
                # plt.xlabel("Value")
                # plt.ylabel("Frequency")
                # plt.legend()
                #
                # lu2=img_hsv[...,1].flatten()
                # plt.subplot(1,3,2)
                # plt.hist(lu2,bins=100,range=(0.0,1.0),histtype='stepfilled', color='g', label='Saturation')
                # plt.title("Saturation")
                # plt.legend()
                #
                # lu3=img_hsv[...,2].flatten()
                # plt.subplot(1,3,3)
                # plt.hist(lu3*255,bins=256,range=(0.0,255.0),histtype='stepfilled', color='b', label='Intensity')
                # plt.title("Intensity")
                # plt.legend()
                # plt.show()
                # plt.hist(hist)
                # plt.title("color Histogram")
                # plt.xlabel("Value")
                # plt.ylabel("Frequency")
                #
                # fig = plt.gcf()
                # plt.show()
            featureArray.append(hist)
            labelArray.append(label)
        else:
            os.remove(imagePath)

def computeSparseFeatures(image_directory, sparseFeatureArray, patches):
    for (i, imagePath) in enumerate(image_directory):
        oneImageArray = []
        image = cv2.imread(imagePath)
        if not image is None:
            pixels = cv2.resize(image, (32, 32))
            individualColourArray = computeFeatureArray(patches, pixels)
            individualColourArray = calculateVarianceArray(individualColourArray)
            oneImageArray = individualColourArray
            total = np.linalg.norm(oneImageArray)
            if total != 0:
              oneImageArray /= total
              print oneImageArray
            # if i == 0:
            #     print oneImageArray
            #     # print oneImageArray.shape
            #     plt.hist(np.array(oneImageArray))
            #     plt.title("KMeans Histogram")
            #     plt.xlabel("Value")
            #     plt.ylabel("Frequency")
            #
            #     fig = plt.gcf()
            #     plt.show()
        oneImageArray = np.array(oneImageArray).flatten()
        if len(oneImageArray) > 0:
            sparseFeatureArray.append(oneImageArray)
def computeSparseWithLabel(image_directory, sparseFeatureArray, label, labelArray, patches):
        for (i, imagePath) in enumerate(image_directory):
            oneImageArray = []
            image = cv2.imread(imagePath)
            if not image is None:
                pixels = cv2.resize(image, (32, 32))
                individualColourArray = computeFeatureArray(patches, pixels)
                individualColourArray = calculateVarianceArray(individualColourArray)
                oneImageArray = individualColourArray
            oneImageArray = np.array(oneImageArray).flatten()
            if len(oneImageArray) > 0:
                sparseFeatureArray.append(oneImageArray)
                labelArray.append(label)

def computeSparseLabelFilePath(image_directory, sparseFeatureArray, label, labelArray, patches, filePaths):
        for (i, imagePath) in enumerate(image_directory):
            oneImageArray = []
            image = cv2.imread(imagePath)
            if not image is None:
                filePaths.append(imagePath)
                pixels = cv2.resize(image, (32, 32))
                individualColourArray = computeFeatureArray(patches, pixels)
                individualColourArray = calculateVarianceArray(individualColourArray)
                oneImageArray = individualColourArray
            oneImageArray = np.array(oneImageArray).flatten()
            if len(oneImageArray) > 0:
                sparseFeatureArray.append(oneImageArray)
                labelArray.append(label)
# initialize the raw pixel intensities matrix and label matrix

def measure_knn_acc(classArray, labelArray, validationClassArray, validationLabelArray, patches):
    sparseFeatureArray = []
    sparseLabelArray = []
    sparseValidationFeatures = []
    sparseValidationLabels = []
    for (i, classFeatures) in enumerate(classArray):
        classFeatureArray = []
        classLabelArray = []
        computeSparseWithLabel(classFeatures, classFeatureArray, labelArray[i], classLabelArray, patches)
        sparseFeatureArray = sparseFeatureArray + classFeatureArray
        sparseLabelArray = sparseLabelArray + classLabelArray
    for j in range(0, len(validationClassArray)):
        classValidationFeatures = []
        classValidationLabels = []
        computeSparseWithLabel(validationClassArray[j], classValidationFeatures,validationLabelArray[j], classValidationLabels, patches)
        sparseValidationFeatures = sparseValidationFeatures + classValidationFeatures
        sparseValidationLabels = sparseValidationLabels + classValidationLabels
    sparseFeatureArray = np.array(sparseFeatureArray)
    sparseLabelArray = np.array(sparseLabelArray)
    sparseValidationFeatures = np.array(sparseValidationFeatures)
    validationLabels = np.array(sparseValidationLabels)
    model = KNeighborsClassifier(n_neighbors= 4,
    	n_jobs= 1)
    model.fit(sparseFeatureArray, sparseLabelArray)
    acc = model.score(sparseValidationFeatures, validationLabels)
    print("ACCURACY CHECK", acc)
    return acc

def shift_unlabelled_into_labelled(label, acc, validationFeatures, validationLabels):
    for (i, imagePath) in enumerate(list(paths.list_images('unlabelled_image_sets/' + label))):
        cat_patches = KMeans(dog_image_paths)

        cat_patches = normalize(cat_patches, axis=1, norm='l1')
        image_paths_length = len(list(paths.list_images('image_sets/' + label)))
        currentDir = os.getcwd();
        print "Current directory length"
        print image_paths_length
        extension = imagePath.split('.')[0]
        file_type = imagePath.split('.')[1]
        os.rename(imagePath, extension + '_unlabelled.' + file_type)
        fileName = ntpath.basename(extension + '_unlabelled.' + file_type)
        open(currentDir + '/image_sets/' + label +'/' + fileName, 'a+').close()
        shutil.move(extension + '_unlabelled.' + file_type, currentDir + '/image_sets/' + label + '/' + fileName)
        print "New directory length"
        print image_paths_length
        newSparseFeatures = []
        newSparseLabels = []
        filePaths = []
        computeSparseWithLabel(list(paths.list_images('image_sets/cat')),newSparseFeatures, 'cat', newSparseLabels, cat_patches)
        computeSparseWithLabel(list(paths.list_images('image_sets/dog')), newSparseFeatures,  'dog', newSparseLabels, cat_patches)
        sparse_train_features = np.array(newSparseFeatures)
        train_labels = np.array(newSparseLabels)
        model = KNeighborsClassifier(n_neighbors= 3,
        	n_jobs= 1)
        print "Check Shape:"
        print sparse_train_features.shape
        print train_labels.shape
        model.fit(sparse_train_features, train_labels)
        print "current accuracy"
        print acc
        new_acc = model.score(validationFeatures, validationLabels)
        print "New Accuracy"
        print new_acc
        if new_acc < acc:
            print "final removed div"
            print '/image_sets/' + label +'/' + fileName
            os.remove(currentDir + '/image_sets/' + label +'/' + fileName)
        else:
            acc = new_acc
def new_shift_unlabelled_to_labelled(label, acc, validationFeatures, validationLabels):
    for (i, imagePath) in enumerate(list(paths.list_images('unlabelled_image_sets/' + label))):
        image_paths_length = len(list(paths.list_images('image_sets/' + label)))
        currentDir = os.getcwd();
        print "Current directory length"
        print image_paths_length
        extension = imagePath.split('.')[0]
        file_type = imagePath.split('.')[1]
        os.rename(imagePath, extension + '_unlabelled.' + file_type)
        fileName = ntpath.basename(extension + '_unlabelled.' + file_type)
        open(currentDir + '/image_sets/' + label +'/' + fileName, 'a+').close()
        if len(list(paths.list_images('unlabelled_image_sets/' + label))) > 0:
            shutil.move(extension + '_unlabelled.' + file_type, currentDir + '/image_sets/' + label + '/' + fileName)
            print "New directory length"
            print image_paths_length
sparseValidationFeatures = []
validationFeatures = []
validationLabels = []

sparseFeatures = []
features = []
filePaths = []
labels = []

benchmark_features = []
benchmark_labels = []

cat_image_paths = list(paths.list_images('image_sets/cat'))[0:150]
dog_image_paths = list(paths.list_images('image_sets/dog'))[0:150]
neither_image_paths = list(paths.list_images('image_sets/neither'))[0:150]
new_cat_image_paths = list(paths.list_images('unlabelled_image_sets/cat'))
new_dog_image_paths = list(paths.list_images('unlabelled_image_sets/dog'))
total_cat_image_paths = cat_image_paths + new_cat_image_paths
total_dog_image_paths = dog_image_paths + new_dog_image_paths
#
# form_NMF_matrices(new_cat_image_paths)
cat_patches = KMeans(dog_image_paths)

cat_patches = normalize(cat_patches, axis=1, norm='l1')

# sparsePCA(dog_image_paths)
# cat_benchmark_image_paths = list(paths.list_images('benchmark_image_sets/cat'))
# dog_benchmark_image_paths = list(paths.list_images('benchmark_image_sets/dog'))

cat_validation_image_paths = list(paths.list_images('validation/cat'))
dog_validation_image_paths = list(paths.list_images('validation/dog'))

appendImagePixels(cat_validation_image_paths, 'cat', validationFeatures, validationLabels, [False])
appendImagePixels(dog_validation_image_paths, 'dog', validationFeatures, validationLabels, [False])
appendImagePixels(cat_image_paths, 'cat', features, labels, [True, filePaths])
appendImagePixels(dog_image_paths, 'dog', features, labels, [True,filePaths])
appendImagePixels(neither_image_paths, 'neither', features, labels, [True, filePaths])

computeSparseFeatures(cat_validation_image_paths, sparseValidationFeatures, cat_patches)
computeSparseFeatures(dog_validation_image_paths, sparseValidationFeatures, cat_patches)

computeSparseFeatures(cat_image_paths, sparseFeatures, cat_patches)
computeSparseFeatures(dog_image_paths, sparseFeatures, cat_patches)
computeSparseFeatures(neither_image_paths, sparseFeatures, cat_patches)



validationFeatures = np.array(validationFeatures)

train_features = np.array(features)

train_labels = np.array(labels)
# benchmarkFeatures = np.array(benchmark_features)
# benchmarkLabels = np.array(benchmark_labels)

sparse_train_features = np.array(sparseFeatures)

sparseValidationFeatures = np.array(sparseValidationFeatures)

# (trainRI, testRI, trainRL, testRL) = train_test_split(
# 	rawImages, labels, test_size=0.25, random_state=42)
print("[INFO] evaluating histogram accuracy...")
model = KNeighborsClassifier(n_neighbors= 5,
	n_jobs= 1)

model.fit(train_features, train_labels)
acc = model.score(validationFeatures, validationLabels)
print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))

print("[INFO] evaluating sparse coding accuracy...")
model = KNeighborsClassifier(n_neighbors= 5,
	n_jobs= 1)

model.fit(sparse_train_features, train_labels)
acc = model.score(sparseValidationFeatures, validationLabels)
print("[INFO] sparse accuracy: {:.2f}%".format(acc * 100))



# def createBestPatches(unlabelled_image_set):
#     for i in range(100):
#         for index, image in enumerate(unlabelled_image_set):
#             model = KNeighborsClassifier(n_neighbors= 1,
#             	n_jobs= 1)
#             counter = counter + 1
#             print()




# Put good images that contribute to KNN performance from unlabelled_image_set into image_set
# new_cat_image_paths = list(paths.list_images('unlabelled_image_sets/cat'))
# new_dog_image_paths = list(paths.list_images('unlabelled_image_sets/dog'))
#
# new_shift_unlabelled_to_labelled('cat', acc, sparseValidationFeatures, validationLabels)
# new_shift_unlabelled_to_labelled('dog', acc, sparseValidationFeatures, validationLabels)




buffer_features = sparseFeatures[:]
buffer_labels = labels[:]

# Antagonistic inference - removing data points that worsen knn performance
def cleanData(features, labels, buffer_labels, buffer_features, acc):
    for i in range(200):
        buffer_index = 0
        cat_image_paths = list(paths.list_images('image_sets/cat'))
        dog_image_paths = list(paths.list_images('image_sets/dog'))
        filePaths = cat_image_paths + dog_image_paths
        print "filePaths Length"
        print(len(filePaths))
        max_accuracy = 0
        max_file_path = ''
        counter = 0
        buffer_features = features[:]
        buffer_labels = labels[:]
        for index, feature in enumerate(features):
            model = KNeighborsClassifier(n_neighbors= 4,
            	n_jobs= 1)
            counter = counter + 1
            print "BUFFER LENGTH"
            print len(buffer_features)
            print len(buffer_labels)
            features = buffer_features[:]
            del features[index]
            new_train_features = np.array(features)
            labels = buffer_labels[:]
            del labels[index]
            new_train_labels = np.array(labels)
            model.fit(new_train_features, new_train_labels)
            new_acc = model.score(sparseValidationFeatures, validationLabels)
            print("[INFO] new sparse accuracy: {:.2f}%".format(new_acc * 100))
            if new_acc > max_accuracy:
                max_accuracy = new_acc
                max_file_path = filePaths[index]
                print max_file_path
                buffer_index = index
                print "INDEx"
                print index

        if max_accuracy > acc:
            print "GET RID OF THIS ONe"
            print max_accuracy
            print acc
            print max_file_path
            print buffer_index
            acc = max_accuracy
            del buffer_features[buffer_index]
            del buffer_labels[buffer_index]
            print "buffer length 2"
            print len(buffer_features)
            print len(buffer_labels)
            print "Max file Path Test"
            print max_file_path
            os.remove(max_file_path)
            # try:
            #     print "Max file Path Test"
            #     print max_file_path
            #     os.remove(max_file_path)
            # except OSError:
            #     print "Coulnd't remove file"
            #     pass
        else:
            print "BREAKING OFF"
            print max_accuracy
            break

# cleanData(sparseFeatures, labels,buffer_labels, buffer_features, acc)

# benchmarkModel = model = KNeighborsClassifier(n_neighbors= 1,
# 	n_jobs= -1)
# print("EVALUATING BENCHMARK ACCURACY")
# benchmarkModel.fit(benchmark_features, benchmark_labels)
# acc = model.score(validationFeatures, validationLabels)
# print("[INFO] benchmark raw pixel accuracy: {:.2f}%".format(acc * 100))
