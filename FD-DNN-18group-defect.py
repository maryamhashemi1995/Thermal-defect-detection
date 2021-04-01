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



images_path1="E:/Poland-internship/Additive_manufacturing/Additive_manufacturing/1-9/Normal/"
images_path2="E:/Poland-internship/Additive_manufacturing/Additive_manufacturing/1-18/Normal/"
images_path3="E:/Poland-internship/Additive_manufacturing/Additive_manufacturing/7-18/ROI/"
images_path4="E:/Poland-internship/Additive_manufacturing/Additive_manufacturing/7-18/Normal/"
images_path5="E:/Poland-internship/Additive_manufacturing/Additive_manufacturing/7-13/ROI/"
images_path6="E:/Poland-internship/Additive_manufacturing/Additive_manufacturing/7-13/Normal/"
images_path7="E:/Poland-internship/Additive_manufacturing/Additive_manufacturing/1-9/ROI/"
images_path8="E:/Poland-internship/Additive_manufacturing/Additive_manufacturing/1-9/Normal/"
images_path9="E:/Poland-internship/Additive_manufacturing/Additive_manufacturing/1-13/ROI/"
images_path10="E:/Poland-internship/Additive_manufacturing/Additive_manufacturing/1-13/Normal/"
images_path11="E:/Poland-internship/Additive_manufacturing/Additive_manufacturing/4-18/ROI/"
images_path12="E:/Poland-internship/Additive_manufacturing/Additive_manufacturing/4-18/Normal/"
images_path13="E:/Poland-internship/Additive_manufacturing/Additive_manufacturing/4-9/ROI/"
images_path14="E:/Poland-internship/Additive_manufacturing/Additive_manufacturing/4-9/Normal/"
images_path15="E:/Poland-internship/Additive_manufacturing/Additive_manufacturing/7-9/ROI/"
images_path16="E:/Poland-internship/Additive_manufacturing/Additive_manufacturing/7-9/Normal/"
images1=glob.glob(images_path1+"*.bmp")
images2=glob.glob(images_path2+"*.bmp")
images3=glob.glob(images_path3+"*.bmp")
images4=glob.glob(images_path4+"*.bmp")
images5=glob.glob(images_path5+"*.bmp")
images6=glob.glob(images_path6+"*.bmp")
images7=glob.glob(images_path7+"*.bmp")
images8=glob.glob(images_path8+"*.bmp")
images9=glob.glob(images_path9+"*.bmp")
images10=glob.glob(images_path10+"*.bmp")
images11=glob.glob(images_path11+"*.bmp")
images12=glob.glob(images_path12+"*.bmp")
images13=glob.glob(images_path13+"*.bmp")
images14=glob.glob(images_path14+"*.bmp")
images15=glob.glob(images_path15+"*.bmp")
images16=glob.glob(images_path16+"*.bmp")

images=[images1,images2,images3,images4,images5,images6,images7,images8,images9,
        images10,images11,images12,images13,images14,images15,images16]
#images=[images1]

labelname=[]
labelclass=[]
index=0
for i in images:
    index+=1
    countimg=0
    if index%2==0:
        countlabel=1
    else:
        countlabel=0
    for j in i:
        countimg+=1
        labelclass.append(countlabel)
        img=cv2.imread(j)
        labelname.append(img)
        
labelname=np.array(labelname)






y_train, y_test,X_train, X_test  = train_test_split(labelclass, labelname, test_size=0.4, shuffle=True)
#y_train, y_test,X_train, X_test  = train_test_split(labelclass, labelname, test_size=0.4, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.6, shuffle=True)

X_valid = np.array(X_valid)    # converting list to array
X_train=np.array(X_train)
X_test=np.array(X_test)


dummy_y_train = np_utils.to_categorical(y_train)    # one hot encoding Classes
dummy_y_valid_images = np_utils.to_categorical(y_valid)    # one hot encoding Classes
dummy_y_test = np_utils.to_categorical(y_test)



del images, images1 ,images2,images3,images4,images5,images6,images7,images8,images9,images10,images11,images12,images13,images14,images15,images16
del images_path1, images_path2,images_path3,images_path4,images_path5,images_path6,images_path7,images_path8
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
nb_classes = 2
epochs =25




from keras.applications.vgg16 import preprocess_input
X_train = preprocess_input(X_train, mode='tf')      # preprocessing the input data
X_valid = preprocess_input(X_valid, mode='tf')      # preprocessing the input data
X_test = preprocess_input(X_test, mode='tf') 


model = Sequential()

model.add(Convolution2D(32, (3, 3), padding='same',
                        input_shape=(224,224,3)))
model.add(Activation('relu'))
model.add(Convolution2D(24, (3, 3), data_format='channels_first'),)
model.add(Activation('relu'))
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

model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))
model.summary()

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd , metrics=['accuracy'])


import datetime
start=datetime.datetime.now()

history=model.fit(X_train, dummy_y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(X_valid, dummy_y_valid_images))

score = model.evaluate(X_test, dummy_y_test,  verbose=1)

print('Test score:', score[0])
print('Test accuracy:', score[1])


# Get training and test loss histories
training_loss = history.history['loss']
test_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();

plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

## Save the weights
#model.save_weights('model_weights_FD-DNN_defect.h5')
## Save the model architecture
#with open('model_architecture_vgg19_zju.json', 'w') as f:
#    f.write(model.to_json())
#
#
#end=datetime.datetime.now()
#elapsed=end-start
#plot_model(model, to_file='modelfd-dnn-defect.png')
#print('training time',str(elapsed))

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
for p in range (0,924):
    if probs1[p,0]>0.5:
        probs.append(1)
    else:
        probs.append(0)
            
    
perf_measure(dummy_y_valid_images,probs )


probstest=[]    
for p in range (0,1387):
    if probs2[p,0]>0.5:
        probstest.append(1)
    else:
        probstest.append(0)
            
    
perf_measure(dummy_y_test,probstest )




import numpy as np
import os
import cv2
def noisy(noise_typ,image):
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 1
      var = 10
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
   elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 3
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
   elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy



image1= cv2.imread("E:/Poland-internship/Additive_manufacturing/Additive_manufacturing/1-13/Normal/250.bmp")
image2= cv2.imread("E:/Poland-internship/Additive_manufacturing/Additive_manufacturing/4-9/ROI/24.bmp")
image3= cv2.imread("E:/Poland-internship/Additive_manufacturing/Additive_manufacturing/7-18/Normal/103.bmp")
image4= cv2.imread("E:/Poland-internship/Additive_manufacturing/Additive_manufacturing/1-13/ROI/307.bmp")
image5= cv2.imread("E:/Poland-internship/Additive_manufacturing/Additive_manufacturing/7-9/Normal/85.bmp")
image6= cv2.imread("E:/Poland-internship/Additive_manufacturing/Additive_manufacturing/4-18/ROI/284.bmp")

noisy1=noisy("gauss",image1)
cv2.imshow("",noisy1)
cv2.waitKey()
cv2.destroyAllWindows()
noisy2=noisy("poisson",image2)
noisy3=noisy("poisson",image3)
noisy4=noisy("poisson",image4)
noisy5=noisy("poisson",image5)
noisy6=noisy("poisson",image6)

from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from skimage.transform import resize


noisy1=np.array(noisy1)
noisy1=resize(noisy1, preserve_range=True, output_shape=(224,224,3))
x1 = image.img_to_array(noisy1)
x1 = np.expand_dims(x1, axis=0)
x1 = preprocess_input(x1,mode='tf')

noisy2=np.array(noisy2)
noisy2=resize(noisy2, preserve_range=True, output_shape=(224,224,3))
x2 = image.img_to_array(noisy2)
x2 = np.expand_dims(x2, axis=0)
x2 = preprocess_input(x2,mode='tf')
noisy3=np.array(noisy3)
noisy3=resize(noisy3, preserve_range=True, output_shape=(224,224,3))
x3 = image.img_to_array(noisy3)
x3 = np.expand_dims(x3, axis=0)
x3= preprocess_input(x3,mode='tf')

noisy4=np.array(noisy4)
noisy4=resize(noisy4, preserve_range=True, output_shape=(224,224,3))
x4 = image.img_to_array(noisy4)
x4 = np.expand_dims(x4, axis=0)
x4 = preprocess_input(x4,mode='tf')

noisy5=np.array(noisy5)
noisy5=resize(noisy5, preserve_range=True, output_shape=(224,224,3))
x5 = image.img_to_array(noisy5)
x5 = np.expand_dims(x5, axis=0)
x5 = preprocess_input(x5,mode='tf')

noisy6=np.array(noisy6)
noisy6=resize(noisy6, preserve_range=True, output_shape=(224,224,3))
x6 = image.img_to_array(noisy6)
x6 = np.expand_dims(x6, axis=0)
x6 = preprocess_input(x6,mode='tf')
#import datetime
#start=datetime.datetime.now()
predict1=model.predict(x1,batch_size=None,verbose=1,steps=None)
predict2=model.predict(x2,batch_size=None,verbose=1,steps=None)
predict3=model.predict(x3,batch_size=None,verbose=1,steps=None)
predict4=model.predict(x4,batch_size=None,verbose=1,steps=None)
predict5=model.predict(x5,batch_size=None,verbose=1,steps=None)
predict6=model.predict(x6,batch_size=None,verbose=1,steps=None)
