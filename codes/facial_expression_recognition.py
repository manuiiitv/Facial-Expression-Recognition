from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
import numpy as np
import pandas as pd
import dicom
import pylab
from scipy import ndimage as ndi
from skimage.measure import label
from pylab import *
from PIL import Image
import PIL.ImageOps  
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg19 import VGG19
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import matplotlib.image as mpimg
import theano
from PIL import Image
from numpy import *
from scipy.misc import toimage
from scipy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from skimage.filters.rank import otsu
from skimage.morphology import disk
from skimage.filters import threshold_otsu

import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
import skimage.morphology, skimage.data
from skimage.morphology import erosion, dilation, opening, closing, white_tophat,remove_small_objects,remove_small_holes
from skimage.morphology import black_tophat, skeletonize, convex_hull_image,binary_closing
from skimage.morphology import disk, ball
from skimage.morphology import reconstruction

import mahotas as mh

from skimage.segmentation import find_boundaries
from skimage.morphology import convex_hull_image
from skimage import data, util
from skimage.measure import label
from skimage.measure import regionprops
# input image dimensions
img_rows, img_cols = 48,48

# number of channels
img_channels = 1

#seed = 6
#np.random.seed(seed)

#taking images as input
ar = []




with open('fer2013.csv', 'r') as fp:
	for line in fp:
		ar.append(line.split(','))


imlist = []
label=[]
for i in range(1,28710):
	number_string = ar[i][1].split(' ')
	number_string = [int(k) for k in number_string]
	number_string = np.asarray(number_string)	
	img = number_string.reshape(img_rows,img_cols)
	pylab.imshow(img, cmap = 'gray')
	pylab.show()
	imlist.append(img) 
	label.append(ar[i][0])
	




immatrix = array([array(imlist[i]).flatten() for i in range(0,len(imlist))],'f') 
print (immatrix.shape)

label=np.array(label)
data,Label = shuffle(immatrix,label, random_state=1)
train_data = [data,Label]





test_imlist = []
test_label = []
for i in range(28711,32299):
	number_string = ar[i][1].split(' ')
	number_string = [int(k) for k in number_string]
	number_string = np.asarray(number_string)	
	img = number_string.reshape(img_rows,img_cols)
	pylab.imshow(img, cmap = 'gray')
	pylab.show()
	test_imlist.append(img) 
	test_label.append(ar[i][0])
	




test_immatrix = array([array(test_imlist[i]).flatten() for i in range(0,len(test_imlist))],'f') 
print (test_immatrix.shape)

test_label=np.array(test_label)
testing_data,testing_Label = shuffle(test_immatrix,test_label, random_state=1)
test_data = [testing_data,testing_Label]


#batch_size to train
batch_size = 300
# number of output classes
nb_classes = 7
# number of epochs to train
nb_epoch = 5


# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3


(X_test, y_test) = (test_data[0],test_data[1])
(X_train, y_train) = (train_data[0],train_data[1])

# STEP 1: split X and y into training and testing sets


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=1)

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_val = X_val.reshape(X_val.shape[0], 1, img_rows, img_cols)

X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_train = X_train.astype('float32')
X_val = X_val.astype('float32')

X_test = X_test.astype('float32')

X_train /= (256)
X_val /= (256)
X_test /= (256)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_val = np_utils.to_categorical(y_val, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


"""
model = Sequential() 		
model.add(Convolution2D(32, 3, 3,input_shape=(1,img_rows,img_cols), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))		
num_classes=7	
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
"""
model = Sequential()

model.add(Convolution2D(32, 3, 3,input_shape=(1,img_rows,img_cols), activation='relu'))

model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(32, 3, 3, activation='relu'))

model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop', metrics=['accuracy'])
hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              show_accuracy=True, verbose=1, validation_data=(X_val, Y_val))
           

# visualizing losses and accuracy

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(nb_epoch)

plt.figure(1,figsize=(7,5))
plt.subplot(211)
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
print (plt.style.available) # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

#plt.figure(2,figsize=(7,5))
plt.subplot(212)
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.tight_layout()
plt.show()



score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print(model.predict_classes(X_test[1:5]))
print(Y_test[1:5])



# Confusion Matrix

from sklearn.metrics import classification_report,confusion_matrix
y_pred = model.predict_classes(X_test)
p=model.predict_proba(X_test) # to predict probability
target_names = ['class 0(benign)', 'class 1(malignant)', 'class 0(benign)', 'class 1(malignant)', 'class 0(benign)',
                'class 1(malignant)', 'class 0(benign)']
print(classification_report(np.argmax(Y_test,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(Y_test,axis=1), y_pred))

