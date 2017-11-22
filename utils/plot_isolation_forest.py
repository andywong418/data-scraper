import numpy as np
import sys
sys.path.append('/Users/androswong/.virtualenvs/cv/lib/python2.7/site-packages/')
import cv2
import imutils
from imutils import paths
from sklearn.externals import joblib
from sklearn import svm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import keras
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import operator
from keras.applications.vgg16 import preprocess_input
from scipy import stats
K.set_image_dim_ordering('th')
img_width, img_height = 150, 150
outliers_fraction = 0.1
clf = joblib.load('svm_one_class_pca.pkl')
train_data = np.load(open('svm_train_features.npy'))
X_train_special = np.reshape(train_data, (2704, 8192))
print train_data.shape
TwoDim_dataset = train_data.reshape(2704,-1)
X_train = PCA(n_components=2).fit_transform(TwoDim_dataset)
scores_pred = clf.decision_function(X_train)
threshold = stats.scoreatpercentile(scores_pred,100 * outliers_fraction)
# print reduced_data.shape
print X_train.shape
y_pred_train = clf.predict(X_train)
print y_pred_train
np.savetxt('OneClassPCA.txt', y_pred_train, delimiter=',')
# plot the line, the samples, and the nearest vectors to the plane
xx, yy = np.meshgrid(np.linspace(-30, 30, 200), np.linspace(-30, 30, 200))
print np.c_[xx.ravel(), yy.ravel()].shape
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
relevant_data = []
noise = []
for index, i in enumerate(y_pred_train.tolist()):
    if i == 1:
        relevant_data.append(index)
    else:
        noise.append(index)

print len(relevant_data)
print len(noise)
plt.title("One Class")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
a = plt.contour(xx, yy, Z, levels=[threshold],
                            linewidths=2, colors='red')
plt.contourf(xx, yy, Z, levels=[threshold, Z.max()],
                         colors='orange')
relevant_data = X_train[relevant_data]
noise = X_train[noise]
b1 = plt.scatter(relevant_data[:, 0], relevant_data[:, 1], c='white')
# b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green')
c = plt.scatter(noise[:, 0], noise[:, 1], c='red')
plt.axis('tight')
plt.xlim((-30, 30))
plt.ylim((-30, 30))
plt.legend([a.collections[0],b1, c],
           ["Learned decision function","training observations", "noise"],
           loc="upper left")
plt.show()
clf = joblib.load('svm_one_class.pkl')
y_pred_all_features = clf.predict(X_train_special)
relevant_data = []
noise = []
for index, i in enumerate(y_pred_all_features.tolist()):
    if i == 1:
        relevant_data.append(index)
    else:
        noise.append(index)

print len(relevant_data)
print noise
np.savetxt('svm_one_class_noise.txt', noise)
np.savetxt('OneClass.txt', y_pred_all_features, delimiter=',')

model = load_model('initial_model_save.h5')
final_results = []
# img = load_img('image_sets/neither/image_12_unlabelled.jpg', target_size=(img_width, img_height))
# x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
# layer_output = model.predict(x)
# # print layer_output.shape
# identifier = np.reshape(layer_output, (1, 8192))
# print clf.predict(identifier)
for i, imagePath in enumerate(list(paths.list_images('image_sets/dog'))):

    img = load_img(imagePath, target_size=(img_width, img_height))
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape( (1,) + x.shape )
    x /= 255
    layer_output = model.predict(x)
    # print layer_output.shape
    identifier = np.reshape(layer_output, (1, 8192))
    # print identifier.shape
    # result = PCA(n_components=4).fit_transform(identifier)
    # print result.shape
    final_results.append(clf.predict(identifier))
relevant_counter = 0
not_relevant_counter = 0
for i in np.array(final_results).flatten().tolist():
    if i ==1:
        relevant_counter += 1
    else:
        not_relevant_counter += 1
print relevant_counter
print not_relevant_counter
np.savetxt('cat_one_class_test.txt', np.array(final_results).flatten())
