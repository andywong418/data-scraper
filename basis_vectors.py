import time

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys
sys.path.append('/Users/androswong/.virtualenvs/cv/lib/python2.7/site-packages/')
import cv2
import PIL
from PIL import Image
from sklearn import datasets
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.image import extract_patches_2d
from scipy import signal
from scipy import ndimage
from sklearn import decomposition

rng = np.random.RandomState(0)
kmeans = MiniBatchKMeans(n_clusters=81, random_state=rng, verbose=True)
patch_size = (3, 3)
import imutils
from imutils import paths
from sklearn.preprocessing import normalize
from scipy import ndimage

# Create an N matrix by M images for NNMF
n_row, n_col = 4, 8
n_components = 32
# patch_size = (10, 10)
def resize_data_images(chosen_path):
    V = []
    for(i, imagePath) in enumerate(chosen_path):
        image = cv2.imread(imagePath)

        if not image is None:
            N_image = cv2.resize(image, patch_size).flatten()
            V.append(N_image)
    return V

def plot_gallery(images, n_col=n_col, n_row=n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        print comp.shape
        plt.imshow(comp.reshape(3,3,3),
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)

def estimator(estimator, data):
    estimator.fit(data)
    if hasattr(estimator, 'cluster_centers_'):
        components_ = estimator.cluster_centers_
    else:
        components_ = estimator.components_
    # if (hasattr(estimator, 'noise_variance_') and
    #         estimator.noise_variance_.shape != ()):
    #     plot_gallery(estimator.noise_variance_.reshape(1, -1), n_col=1,n_row=1)
    # plot_gallery(components_[:n_components])
    return components_
def form_NMF_matrices(chosen_path):
    V = resize_data_images(chosen_path)
    V_matrix = np.array(V)
    model = decomposition.NMF(n_components=n_components, init='nndsvda', tol=5e-3)
    W = estimator(model, V_matrix)
    # W = model.fit_transform(V_matrix, y=None, W=None, H=None)
    #
    # for i in range(1,50):
    #     plt.subplot(10,10, i)
    #     # plot_one_patch = np.reshape(W[i],(20,20))
    #     # plt.imshow(plot_one_patch)
    #     # plt.gray()
    #     patch = np.reshape(W[i],(10,10))
    #     plot_one_patch = signal.convolve2d(test_signal, patch, mode='full', boundary='fill', fillvalue=0)
    #     plt.imshow(plot_one_patch)
    #     plt.gray()

    plt.show()
    return W


def form_dictionary(chosen_path):
    rng = np.random.RandomState(0)
    model = decomposition.MiniBatchDictionaryLearning(n_components=n_components, alpha=0.1,
                                                  n_iter=50, batch_size=3,
                                                  random_state=rng)
    data = resize_data_images(chosen_path)
    data = np.array(data)
    n_samples, n_features = data.shape
    print data.shape
    # global centering
    data_centered = data - data.mean(axis=0)

    # local centering
    data_centered -= data_centered.mean(axis=1).reshape(n_samples, -1)
    dictionary = estimator(model, data_centered)
    plt.show()
    return dictionary

def sparsePCA(chosen_path):
    model = decomposition.MiniBatchSparsePCA(n_components=n_components, alpha=0.8,
                                      n_iter=100, batch_size=3,
                                      random_state=rng)
    data = resize_data_images(chosen_path)
    data = np.array(data)
    # global centering
    n_samples, n_features = data.shape
    data_centered = data - data.mean(axis=0)

    # local centering
    data_centered -= data_centered.mean(axis=1).reshape(n_samples, -1)
    PCA_patches = estimator(model, data_centered)
    plt.show()
    return PCA_patches

def fastICA(chosen_path):
    model = decomposition.FastICA(n_components=n_components, whiten=True)
    data = resize_data_images(chosen_path)
    data = np.array(data)
    # global centering
    n_samples, n_features = data.shape
    data_centered = data - data.mean(axis=0)

    # local centering
    data_centered -= data_centered.mean(axis=1).reshape(n_samples, -1)
    ICA_patches = estimator(model, data_centered)
    plt.show()
    return ICA_patches
def KMeans(chosen_path):
    model = MiniBatchKMeans(n_clusters=n_components, tol=1e-3, batch_size=20,
                        max_iter=50, random_state=rng)
    data = resize_data_images(chosen_path)
    data = np.array(data)
    n_samples, n_features = data.shape
    # print "DATA shape"
    # print data.shape
    # global centering
    data_centered = data - data.mean(axis=0)

    # local centering
    data_centered -= data_centered.mean(axis=1).reshape(n_samples, -1)
    KMeans_patches = estimator(model, data_centered)
    plt.show()
    return KMeans_patches


# def convolveImage(imagePaths, test_filter):
#     featureArray = []
#     for (i, imagePath) in enumerate(imagePaths):
#         image = cv2.imread(imagePath, 0)
#         if not image is None:
#             pixels = cv2.resize(image, (32, 32))
#             test_filter = np.reshape(test_filter, patch_size)
#             convolvedImage = signal.convolve2d(pixels, test_filter)
#             featureArray.append(convolvedImage)
#     return featureArray

def computeFeatureArray(patches, image):
    #calculate 50 filters that calculates the median of
    patches = normalize(patches, axis=1, norm= 'l1')
    featureArray = []
    for patch in patches:
        pixels = cv2.resize(image, (32, 32))
        patch_filter = np.reshape(patch, (3,3,3))
        convolvedImage = ndimage.convolve(pixels, patch_filter)
        featureArray.append(convolvedImage)
    return featureArray




def calculateVarianceArray(featureArray):
    #get list of total variances in each image
    varianceMatrix = []
    for i, item in enumerate(featureArray):
        varianceItem = ndimage.variance(item)
        varianceMatrix.append(varianceItem)
    return varianceMatrix


def convolveImage(imagePaths, featureArray, test_filter):
    for (i, imagePath) in enumerate(imagePaths):
        image = cv2.imread(imagePath, 0)
        if not image is None:
            pixels = cv2.resize(image, (32, 32))
            test_filter = np.reshape(test_filter, patch_size)
            convolvedImage = signal.convolve2d(pixels, test_filter)
            featureArray.append(convolvedImage)
    return featureArray


# plt.show()

# buffer = []
# index = 1
# t0 = time.time()
# data = extract_patches_2d(test_image, patch_size, max_patches=50, random_state=rng)
# data = np.reshape(data, (-1,len(data)))
# print data.shape
# u,s,v = np.linalg.svd(data, full_matrices=True)
#
# # reshaped_patch = np.reshape(u[2], (20,20))
# # plot_one_patch = np.reshape(u[1],(20,20))
# # plt.imshow(plot_one_patch)
# # plt.gray()
# # plt.show()
# # print reshaped_patch
# print s
# print v
# # plot_one_patch = signal.convolve2d(test_image, reshaped_patch, mode='same')
# for i in range(1,50):
#     plt.subplot(20,20, i)
#     plot_one_patch = np.reshape(u[i],(20,20))
#     plt.imshow(plot_one_patch)
#     plt.gray()
# plt.show()
# kmeans.partial_fit(data)
# dt = time.time() - t0
# print('done in %.2fs.' % dt)
#
# plt.figure(figsize=(4.2, 4))
# for i, patch in enumerate(kmeans.cluster_centers_):
#     plt.subplot(9, 9, i + 1)
#     plt.imshow(patch.reshape(patch_size), cmap=plt.cm.gray,
#                interpolation='nearest')
#     plt.xticks(())
#     plt.yticks(())
#
#
# plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
#
# plt.show()
