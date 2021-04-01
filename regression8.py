from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
#import tensorflow as tf
#from tensorflow.python.client import device_lib 
#from keras.preprocessing import image
from keras.utils import np_utils
from skimage.transform import resize
import glob
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout, Activation
from keras.utils import plot_model
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D, MaxPooling2D

import keras
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential, Model


images_path1="C:/Users/MaryamHashemi/Desktop/Additive_manufacturing/Additive_manufacturing/4-18/ROI/"
images_path2="C:/Users/MaryamHashemi/Desktop/Additive_manufacturing/Additive_manufacturing/7-18/ROI/"
images_path3="C:/Users/MaryamHashemi/Desktop/Additive_manufacturing/Additive_manufacturing/7-13/ROI/"
images_path4="C:/Users/MaryamHashemi/Desktop/Additive_manufacturing/Additive_manufacturing/4-13/ROI/"
images_path5="C:/Users/MaryamHashemi/Desktop/Additive_manufacturing/Additive_manufacturing/1-18/ROI/"
images_path6="C:/Users/MaryamHashemi/Desktop/Additive_manufacturing/Additive_manufacturing/1-13/ROI/"


images1=glob.glob(images_path1+"*.bmp")
images2=glob.glob(images_path2+"*.bmp")
images3=glob.glob(images_path3+"*.bmp")
images4=glob.glob(images_path4+"*.bmp")
images5=glob.glob(images_path5+"*.bmp")
images6=glob.glob(images_path6+"*.bmp")

images=[images1,images2,images3,images4,images5,images6]


labelclass = np.zeros((2166,1), dtype="uint8")

labelclass[:361] =4
labelclass[361:722] = 7
labelclass[722:1083] = 7
labelclass[1083:1444] = 4
labelclass[1444:1805] = 1
labelclass[1805:] = 1

X_train = [ ]     # creating an empty array
X_valid=[]
y_train=[]
y_valid=[]
X_test=[]
Y_test=[]
traincount=-1
for i in images:
    for j in i:
        traincount+=1
        img = cv2.imread( j)
        if traincount%6==0:
            X_test.append(img)
            Y_test.append(labelclass[traincount])
        elif traincount%6==1:
            X_valid.append(img)  # storing each image in array X
            y_valid.append(labelclass[traincount])
        else:
            X_train.append(img)
            y_train.append(labelclass[traincount])
    
    
X_valid = np.array(X_valid)    # converting list to array
X_train=np.array(X_train)
X_test=np.array(X_test)

dummy_y_train = np_utils.to_categorical(y_train)    # one hot encoding Classes
dummy_y_valid_images = np_utils.to_categorical(y_valid)    # one hot encoding Classes
dummy_y_test = np_utils.to_categorical(Y_test)

del images, images1 ,images2
del images_path1, images_path2
del img
del labelclass



images_train = []
for i in range(0,X_train.shape[0]):
    a = resize(X_train[i], preserve_range=True, output_shape=(224,224)).astype(int)      # reshaping to 224*224*3
    images_train.append(a)
X_train = np.array(images_train)

images_valid = []
for i in range(0,X_valid.shape[0]):
    a = resize(X_valid[i], preserve_range=True, output_shape=(224,224)).astype(int)      # reshaping to 224*224*3
    images_valid.append(a)
X_valid = np.array(images_valid)

images_test = []
for i in range(0,X_test.shape[0]):
    a = resize(X_test[i], preserve_range=True, output_shape=(224,224)).astype(int)      # reshaping to 224*224*3
    images_test.append(a)
X_test = np.array(images_test)

del a

from keras.applications.vgg16 import preprocess_input
X_train = preprocess_input(X_train, mode='tf')      # preprocessing the input data
X_valid = preprocess_input(X_valid, mode='tf')      # preprocessing the input data
X_test = preprocess_input(X_test, mode='tf') 






base_model = VGG16(weights='imagenet', include_top=False, input_shape=X_train.shape[1:])    # include_top=False to remove the top layer




model=Sequential()
model.add(Flatten(input_shape=base_model.output_shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Dense(8, activation="linear"))

model = Model(inputs=base_model.input, outputs=model(base_model.output))

# initiate RMSprop optimizer
opt =keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='mse',
              optimizer=opt,
              )



history=model.fit(X_train, dummy_y_train, validation_data=(X_test, dummy_y_test),
	epochs=100, batch_size=10)


prediction = model.predict(X_valid)