# -*- coding: utf-8 -*-
"""
Created on Wed May  1 09:01:03 2019

@author: MaryamHashemi
"""

import numpy as np
#import pandas as pd
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

#oc curve and auc
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from keras.models import model_from_json
from sklearn.metrics import average_precision_score

from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.optimizers import SGD, Adadelta, Adagrad



images_path1="C:/Users/MaryamHashemi/Desktop/Additive_manufacturing/Additive_manufacturing/4-18/ROI/"
images_path2="C:/Users/MaryamHashemi/Desktop/Additive_manufacturing/Additive_manufacturing/7-18/ROI/"
images_path3="C:/Users/MaryamHashemi/Desktop/Additive_manufacturing/Additive_manufacturing/1-18/ROI/"
images_path4="C:/Users/MaryamHashemi/Desktop/Additive_manufacturing/Additive_manufacturing/7-13/ROI/"
images_path5="C:/Users/MaryamHashemi/Desktop/Additive_manufacturing/Additive_manufacturing/4-13/ROI/"
images_path6="C:/Users/MaryamHashemi/Desktop/Additive_manufacturing/Additive_manufacturing/1-13/ROI/"
images_path7="C:/Users/MaryamHashemi/Desktop/Additive_manufacturing/Additive_manufacturing/4-9/ROI/"
images_path8="C:/Users/MaryamHashemi/Desktop/Additive_manufacturing/Additive_manufacturing/7-9/ROI/"
images_path9="C:/Users/MaryamHashemi/Desktop/Additive_manufacturing/Additive_manufacturing/1-9/ROI/"


images1=glob.glob(images_path1+"*.bmp")
images2=glob.glob(images_path2+"*.bmp")
images3=glob.glob(images_path3+"*.bmp")
images4=glob.glob(images_path4+"*.bmp")
images5=glob.glob(images_path5+"*.bmp")
images6=glob.glob(images_path6+"*.bmp")
images7=glob.glob(images_path7+"*.bmp")
images8=glob.glob(images_path8+"*.bmp")
images9=glob.glob(images_path9+"*.bmp")
images=[images1,images2,images3,images4,images5,images6,images7,images8,images9]



labelclass = np.zeros((3249,1), dtype="uint8")

labelclass[:361] =1
labelclass[361:722] = 2
labelclass[722:1083] = 0
labelclass[1083:1444] = 2
labelclass[1444:1805] = 1
labelclass[1805:2166] = 0
labelclass[2166:2527] = 1
labelclass[2527:2888] = 2
labelclass[2888:] = 0



labelname=[]
for i in images:
    for j in i:
        img=cv2.imread(j)
        labelname.append(img)

labelname=np.array(labelname)
 
y_train, y_test,X_train, X_test  = train_test_split(labelclass, labelname, test_size=0.4, shuffle=True)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.6, shuffle=True)
X_test, X_notuse, y_test, y_notuse = train_test_split(X_test, y_test, test_size=0.25, shuffle=True)
  
X_valid = np.array(X_valid)    # converting list to array
X_train=np.array(X_train)
X_test=np.array(X_test)
X_notuse=np.array(X_notuse)

dummy_y_train = np_utils.to_categorical(y_train)    # one hot encoding Classes
dummy_y_valid_images = np_utils.to_categorical(y_valid)    # one hot encoding Classes
dummy_y_test = np_utils.to_categorical(y_test)
dummy_y_notuse = np_utils.to_categorical(y_notuse)


del images, images1 ,images2,images3,images4,images5,images6,images7,images8,images9
del images_path1, images_path2,images_path3,images_path4,images_path5,images_path6,images_path7,images_path8,images_path9
del img
del labelclass,labelname



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

images_notuse = []
for i in range(0,X_notuse.shape[0]):
    a = resize(X_notuse[i], preserve_range=True, output_shape=(224,224)).astype(int)      # reshaping to 224*224*3
    images_notuse.append(a)
X_notuse = np.array(images_notuse)

del a

from keras.applications.vgg16 import preprocess_input
X_train = preprocess_input(X_train, mode='tf')      # preprocessing the input data
X_valid = preprocess_input(X_valid, mode='tf')      # preprocessing the input data
X_test = preprocess_input(X_test, mode='tf') 
X_notuse = preprocess_input(X_notuse, mode='tf')






base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))    # include_top=False to remove the top layer

#plot_model(base_model, to_file='Model picture.pdf',show_shapes=True)



X_train = base_model.predict(X_train)
X_valid = base_model.predict(X_valid)
X_test = base_model.predict(X_test)
X_notuse = base_model.predict(X_notuse)
print(X_train.shape, X_valid.shape,X_test.shape)



X_train = X_train.reshape(len(X_train), 7*7*512)      # converting to 1-D
X_valid = X_valid.reshape(len(X_valid), 7*7*512)
X_test = X_test.reshape(len(X_test), 7*7*512)
X_notuse = X_notuse.reshape(len(X_notuse),7*7*512)

train = X_train/X_train.max()      # centering the data
X_valid = X_valid/X_train.max()
X_test=X_test/X_test.max()
X_notuse = X_notuse/X_notuse.max()


model = Sequential()
model.add(InputLayer((7*7*512,)))    # input layer
model.add(Dense(units=1024, activation='relu')) # hidden layer
model.add(Dense(3, activation='softmax'))    # output layer
#model.add(Dense(2))
#model.add(Activation('sigmoid'))


model.summary()


sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])




import datetime
start=datetime.datetime.now()
history=model.fit(train, dummy_y_train, batch_size=32,epochs=50, validation_data=(X_valid, dummy_y_valid_images))



score1 = model.evaluate(X_valid, dummy_y_valid_images, batch_size=32)
x_valid_output_images=model.predict(X_valid)



# Get training and test loss histories
training_loss = history.history['loss']
test_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Valid Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();

plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()




end=datetime.datetime.now()
elapsed=end-start
#plot_model(model, to_file='modelvgg19networkZJU.png')
print('training time',str(elapsed))
print('Cost:', score1[0])
print('Validation accuracy:', score1[1])

# Save the weights
#model.save_weights('model_weights_vgg19_zju.h5')

# Save the model architecture
#with open('model_architecture_vgg19_zju.json', 'w') as f:
#    f.write(model.to_json())





probs1 = model.predict(X_valid)
# keep probabilities for the positive outcome only
#probs = probs[:, 1]
# calculate AUC
auc1 = roc_auc_score(dummy_y_valid_images, probs1)
print('AUC1 (validation): %.3f' % auc1)
# calculate roc curve
score2 = model.evaluate(X_test, dummy_y_test, batch_size=32)
print("Test accuracy:",score2[1])

probs2 = model.predict(X_test)
# keep probabilities for the positive outcome only
#probs = probs[:, 1]
# calculate AUC
auc2 = roc_auc_score(dummy_y_test, probs2)
print('AUC2 (Test): %.3f' % auc2)


score3 = model.evaluate(X_notuse, dummy_y_notuse, batch_size=32)
print("Not use accuracy:",score3[1])

probs3 = model.predict(X_notuse)
# keep probabilities for the positive outcome only
#probs = probs[:, 1]
# calculate AUC
auc3 = roc_auc_score(dummy_y_notuse, probs3)
print('AUC3 (notuse): %.3f' % auc3)
