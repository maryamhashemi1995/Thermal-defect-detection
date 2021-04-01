
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

from keras.models import model_from_json
    
#     Model reconstruction from JSON file
with open('model_architecture_vgg19_zju.json', 'r') as f:
        model = model_from_json(f.read())
    
#     Load weights into the new model
model.load_weights('model_weights_FD-DNN_defect.h5')
 
   
images_path1="C:/Users/MaryamHashemi/Desktop/Additive_manufacturing/Additive_manufacturing/4-13/ROI/"

images1=glob.glob(images_path1+"*.bmp")

images=[images1]
predictarray=[]
correct=0
for i in images:
    for j in i:
        img=cv2.imread(j)
        img=np.array(img)
        img=resize(img, preserve_range=True, output_shape=(224,224,3))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x,mode='tf')
        #import datetime
        #start=datetime.datetime.now()
        predict=model.predict(x,batch_size=None,verbose=1,steps=None)
        if predict[0,0]>= predict[0,1]:
            correct=correct+1
        predictarray.append(predict)
#        end=datetime.datetime.now()
        #elapsed=end-start
        #print('detection time',str(elapsed))
        print(predict)
print(correct)