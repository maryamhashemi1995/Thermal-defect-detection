
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import os
from keras.utils import plot_model
import cv2

np.random.seed(1337)  # for reproducibility



from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, Adadelta, Adagrad
import glob
from six.moves import cPickle as pickle
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from skimage.transform import resize


images_path1="C:/Users/MaryamHashemi/Desktop/Additive_manufacturing/Additive_manufacturing/1-9/Normal/"

images1=glob.glob(images_path1+"*.bmp")

images=[images1]
predictarray=[]
for i in images:
    for j in i:
        img=cv2.imread(j)
        img=resize(img, preserve_range=True, output_shape=(224,224))
        
        h=img.shape[0]
        w=img.shape[1]
        center = (w / 2, h / 2)
        
        angle90 = 90
        angle180 = 180
        angle270 = 270
        angle120=120
        angle60=60
        angle240=240
        angle330=330
        scale = 1.0
        
        M = cv2.getRotationMatrix2D(center, angle180, scale)
        rotated90 = cv2.warpAffine(img, M, (h, w))
                
        x = image.img_to_array(rotated90)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x,mode='tf')
        features = base_model.predict(x)
        features=features.reshape(len(features),7*7*512)
        features=features/features.max()
        #import datetime
        #start=datetime.datetime.now()
        predict=model.predict(features,batch_size=None,verbose=1,steps=None)
        predictarray.append(predict)
#        end=datetime.datetime.now()
        #elapsed=end-start
        #print('detection time',str(elapsed))
        print(predict)



#img=cv2.imread('C:/Users/MaryamHashemi/Desktop/Additive_manufacturing/Additive_manufacturing/1-9/ROI/145.bmp')
##img=cv2.resize(img,(224,224))
#img=resize(img, preserve_range=True, output_shape=(224,224))
#h=img.shape[0]
#w=img.shape[1]
#center = (w / 2, h / 2)
#
#angle90 = 90
#angle180 = 180
#angle270 = 270
#angle120=120
#angle60=60
#angle240=240
#angle330=330
#scale = 1.0
#
#M = cv2.getRotationMatrix2D(center, angle90, scale)
#rotated90 = cv2.warpAffine(img, M, (h, w))
#
#x = image.img_to_array(rotated90)
#x = np.expand_dims(x, axis=0)
#x = preprocess_input(x,mode='tf')
#features = base_model.predict(x)
#features=features.reshape(len(features),7*7*512)
#features=features/features.max()
##import datetime
##start=datetime.datetime.now()
#predict=model.predict(features,batch_size=None,verbose=1,steps=None)
##end=datetime.datetime.now()
##elapsed=end-start
##print('detection time',str(elapsed))
#print(predict)
        
#        x = preprocess_input(img)
#        x=np.reshape(img,(1,25088))
#        x=x/255       
#        predict=model.predict(x)
#        print(predict)
#        if predict<0.5:
#            flag+=1
#            if flag >= frame_check:
#                print ("Drowsy")
#                print()
            



