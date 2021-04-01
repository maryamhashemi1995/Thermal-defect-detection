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

#oc curve and auc
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from keras.models import model_from_json

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import LocallyConnected1D
from keras.utils import np_utils
from keras.optimizers import SGD, Adadelta, Adagrad

from six.moves import cPickle as pickle

import tensorflow as tf
images_path1="E:/Poland-internship/Additive_manufacturing/Additive_manufacturing/4-18/ROI/"
images_path2="E:/Poland-internship/Additive_manufacturing/Additive_manufacturing/7-18/ROI/"
images_path3="E:/Poland-internship/Additive_manufacturing/Additive_manufacturing/1-18/ROI/"
images_path4="E:/Poland-internship/Additive_manufacturing/Additive_manufacturing/7-13/ROI/"
images_path5="E:/Poland-internship/Additive_manufacturing/Additive_manufacturing/4-13/ROI/"
images_path6="E:/Poland-internship/Additive_manufacturing/Additive_manufacturing/1-13/ROI/"
images_path7="E:/Poland-internship/Additive_manufacturing/Additive_manufacturing/4-9/ROI/"
images_path8="E:/Poland-internship/Additive_manufacturing/Additive_manufacturing/7-9/ROI/"
images_path9="E:/Poland-internship/Additive_manufacturing/Additive_manufacturing/1-9/ROI/"


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

labelclass[:361] =2
labelclass[361:722] = 2
labelclass[722:1083] = 2
labelclass[1083:1444] = 1
labelclass[1444:1805] = 1
labelclass[1805:2166] = 1
labelclass[2166:2527] = 0
labelclass[2527:2888] = 0
labelclass[2888:] = 0



labelname=[]
for i in images:
    for j in i:
        img=cv2.imread(j)
        labelname.append(img)

labelname=np.array(labelname)
 
y_train, y_test,X_train, X_test  = train_test_split(labelclass, labelname, test_size=0.6, shuffle=True)
#y_train, y_test,X_train, X_test  = train_test_split(labelclass, labelname, test_size=0.4, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.4, shuffle=True)

  
X_valid = np.array(X_valid)    # converting list to array
X_train=np.array(X_train)
X_test=np.array(X_test)


dummy_y_train = np_utils.to_categorical(y_train)    # one hot encoding Classes
dummy_y_valid_images = np_utils.to_categorical(y_valid)    # one hot encoding Classes
dummy_y_test = np_utils.to_categorical(y_test)


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


del a

batch_size = 32
nb_classes = 3
epochs =25




from keras.applications.vgg16 import preprocess_input
X_train = preprocess_input(X_train, mode='tf')      # preprocessing the input data
X_valid = preprocess_input(X_valid, mode='tf')      # preprocessing the input data
X_test = preprocess_input(X_test, mode='tf') 

#X_train = train_dataset
#X_train = X_train.reshape((X_train.shape[0], X_train.shape[3]) + X_train.shape[1:3])
#Y_train = train_labels
#
#X_test = test_dataset
#X_test = X_test.reshape((X_test.shape[0], X_test.shape[3]) + X_test.shape[1:3])
#Y_test = test_labels
#
#
#X_valid = valid_dataset
#X_valid = X_valid.reshape((X_valid.shape[0], X_valid.shape[3]) + X_valid.shape[1:3])
#Y_valid = valid_labels
## print shape of data while model is building
#print("{1} train samples, {4} channel{0}, {2}x{3}".format("" if X_train.shape[1] == 1 else "s", *X_train.shape))
#print("{1}  test samples, {4} channel{0}, {2}x{3}".format("" if X_test.shape[1] == 1 else "s", *X_test.shape))
#print("{1} valid samples, {4} channel{0}, {2}x{3}".format("" if X_valid.shape[1] == 1 else "s", *X_valid.shape))
#
## input image dimensions
#_, img_channels, img_rows, img_cols = X_train.shape

# convert class vectors to binary class matrices
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(32, (3, 3), padding='same',
                        input_shape=(224,224,3)))
model.add(Activation('relu'))
model.add(Convolution2D(24, (3, 3), data_format='channels_first'),)
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


#model.add(Convolution2D(64, (3, 3), padding='same', data_format='channels_first'))
#model.add(Activation('relu'))
#model.add(Convolution2D(64, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Activation('relu'))
#model.add(Convolution2D(64, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))



model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#model.add(Activation('relu'))
#model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.summary()

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd , metrics=['accuracy'])


import datetime
start=datetime.datetime.now()

history=model.fit(X_train, dummy_y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(X_valid, dummy_y_valid_images))

score = model.evaluate(X_test, dummy_y_test,  verbose=1)

print('Test cost:', score[0])
print('Test accuracy:', score[1])


# Get training and test loss histories
training_loss = history.history['loss']
test_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Val_Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();

plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Save the weights
model.save_weights('model_weights_FD-DNN_depth.h5')
with open('model_architecture_fd-dnn_depth.json', 'w') as f:
    f.write(model.to_json())

end=datetime.datetime.now()
elapsed=end-start
plot_model(model, to_file='modelfd-dnn-depth.png')
print('training time',str(elapsed))

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



def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i,0]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i,0]!=y_hat[i]:
           FP += 1
        if y_actual[i,0]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i,0]!=y_hat[i]:
           FN += 1
    Recal=(TP)/(TP+FN) 
    Presicion=(TP)/(TP+FP)      
    return(TP, FP, TN, FN,"recall=",Recal,"presicion=",Presicion)
    
    
probs=[]    
for p in range (0,1170):
    if probs1[p,0]>0.5:
        probs.append(1)
    elif probs1[p,1]>0.5:
        probs.append(2)
    else:
        probs.append(0)
    
            


probstest=[]    
for p in range (0,780):
    if probs2[p,0]>0.5:
        probstest.append(1)
    elif probs2[p,1]>0.5:
        probstest.append(2)
    else:
        probstest.append(0)
            
    



dummy_y_valid_array=[]    
for p in range (0,1170):
    if dummy_y_valid_images[p,0]==1:
        dummy_y_valid_array.append(1)
    elif dummy_y_valid_images[p,1]==1:
        dummy_y_valid_array.append(2)
    else:
        dummy_y_valid_array.append(0)
        
        
dummy_y_test_array=[]    
for p in range (0,780):
    if dummy_y_test[p,0]==1:
        dummy_y_test_array.append(1)
    elif dummy_y_test[p,1]==1:
        dummy_y_test_array.append(2)
    else:
        dummy_y_test_array.append(0)


from sklearn.metrics import confusion_matrix
confusion_matrix(dummy_y_valid_array, probs)
confusion_matrix(dummy_y_test_array, probstest)



