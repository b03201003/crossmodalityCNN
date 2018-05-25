
# coding: utf-8

# In[1]:


import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import numpy as np


# In[2]:


from keras.models import *
from keras.layers import Input, Lambda, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, core
from keras.layers.merge import concatenate

from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from keras import backend as keras


# In[ ]:


# dim of a 3D data: 32x32x28, where 28 is the number of slices
IMG_L = 32
IMG_W = 32
IMG_H = 28

NUM_OF_MODELS = 3
x = Input(shape=(IMG_L, IMG_W, NUM_OF_MODELS*IMG_H))


# In[1]:


def cross_model(x):
    x = tf.reshape(x, shape=[IMG_L, IMG_W, NUM_OF_MODELS, IMG_H])
    x = tf.transpose(x, perm=[0, 1, 3, 2])
    x = tf.reshape(x, shape=[IMG_L, IMG_W, NUM_OF_MODELS*IMG_H])
    return x[:, :, :, :10]


# In[2]:


y = Lambda(cross_model, output_shape=(IMG_L, IMG_W, NUM_OF_MODELS*IMG_H))(x)
# y = Lambda(cross_model)(x)


# In[5]:


cross_modelity = Model(x, y)
cross_modelity.summary()
cross_modelity.compile(loss='mse', optimizer='sgd')


# In[6]:


img = np.zeros((1, IMG_L, IMG_W, NUM_OF_MODELS*IMG_H), dtype=int)

for i in range(NUM_OF_MODELS):
        img[:, :, :, IMG_H*i:IMG_H*(i+1)] = (i+1)*(np.arange(1,IMG_H+1)+1)
    
img.shape


# Assume we have 3 models, where each model output a 3D image of the size 32x32x28

# In[7]:


for i in range(NUM_OF_MODELS):
    print(img[0, 0, 0, IMG_H*i:IMG_H*(i+1)-3])


# Now, we re-arrange the height of 3 models so that they are vertically flattern

# In[8]:


cross_modelity = K.Function(cross_modelity.inputs, cross_modelity.outputs)


# In[10]:


print("Input shape", img[0][0].shape)
print(img[0][0])

print("Output shape", cross_modelity([img])[0][0].shape)
print(cross_modelity([img])[0][0])

