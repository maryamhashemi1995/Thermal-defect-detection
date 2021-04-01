
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
plot_model(model, to_file='modelfd-dnn-defect.png')

with open('model_architecture_fd-dnn_diameter.json', 'r') as f: 
    modeldi = model_from_json(f.read())
modeldi.load_weights('model_weights_FD-DNN_diameter.h5')

with open('model_architecture_fd-dnn_depth.json', 'r') as f: 
    modeldep = model_from_json(f.read())
modeldep.load_weights('model_weights_FD-DNN_depth.h5')
  
   
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
        predictdefect=model.predict(x,batch_size=None,verbose=1,steps=None)
        if predictdefect[1,0]>= predictdefect[0,0]:
            print("composit is great!")
        else:
            print ('defect detected')
            predictdia=modeldi.predict(x,batch_size=None,verbose=1,steps=None)
            if predictdia[0,0]>= predictdia[0,1] and predictdia[0,0]>= predictdia[0,2]:
                print("diameter=categori 0")
            elif predictdia[0,1]>= predictdia[0,0] and predictdia[0,1]>= predictdia[0,2]:
                print("diameter=categori 1")
            else:
                print("diameter=categori 2")
                
            predictdep=modeldep.predict(x,batch_size=None,verbose=1,steps=None)
            if predictdep[0,0]>= predictdep[0,1] and predictdep[0,0]>= predictdep[0,2]:
                print("diameter=categori 0")
            elif predictdep[0,1]>= predictdep[0,0] and predictdep[0,1]>= predictdep[0,2]:
                print("diameter=categori 1")
            else:
                print("diameter=categori 2")
    
#     Load weights into the new model
 
       