
# coding: utf-8

# In[2]:


import os
import glob
import os
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="1"


# In[3]:


from keras.models import Sequential
from keras.layers import Conv3D, Activation, MaxPooling3D, BatchNormalization, Flatten, Dropout, Dense, GaussianNoise
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator as IDG


# In[4]:


def Get_data(paths):
    hdr = []
    data = []
    label = []
    for path in paths:
        if os.path.isdir(path) is True:
            pathnow = os.path.join(path)
            data_path = glob.glob(pathnow + '/*seg.nii.gz')
            nibfile = nib.load(data_path[0])
            data.append(nibfile.get_data())
            hdr.append(nibfile.header)
    return np.array(data), hdr


# In[5]:


paths = glob.glob('/data/BraTS/BraTS17/MICCAI_BraTS17_Data_Training/HGG/*')
data_HGG, hdr_HGG = Get_data(paths)
paths = glob.glob('/data/BraTS/BraTS17/MICCAI_BraTS17_Data_Training/LGG/*')
data_LGG, hdr_LGG = Get_data(paths)

label = np.zeros((285))
label[0:len(data_HGG)] = 1
label[len(data_HGG):len(data_HGG) + len(data_LGG)] = 0


# In[6]:


data = np.concatenate((data_HGG, data_LGG))/4.0


# In[21]:


data[0].shape


# In[7]:


# def Segmet(img, up, down, axis):
#     for i in range(up):
#         img = np.delete(img, 90 - up, axis)
#     for i in range(down):
#         img = np.delete(img, 0, axis)
#     return img


# In[8]:


# data = Segmet(data, 20, 20, 1)
# data = Segmet(data, 20, 20, 2)
# data = Segmet(data, 20, 20, 3)


# In[35]:


data.shape


# In[36]:


# data_train = data.reshape((285,200,200,115,1))


# In[7]:


data_train = data.reshape((285,240,240,155,1))


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(data_train, label, test_size = 0.2)


# In[38]:


X_train.shape[1:]


# In[12]:


def plot_image(img, x, y, z):
    plt.subplot(131)
    plt.imshow(img[::,::,z])
    plt.subplot(132)
    plt.imshow(img[::,y,::])
    plt.subplot(133)
    plt.imshow(img[x,::,::])


# In[18]:


#plot_image(data[5], 120, 120, 70)


# In[18]:


def model():
    
    model = Sequential()
    model.add(Conv3D(8, kernel_size=(2,2,2), input_shape=(X_train.shape[1:]), activation='relu'))
    model.add(MaxPooling3D(pool_size=([2,2,2])))
    
    model.add(Conv3D(32, kernel_size=(2,2,2), activation='relu'))
    model.add(MaxPooling3D(pool_size=([2,2,2])))
    
    model.add(Conv3D(64, kernel_size=(2,2,2), activation='relu'))
    model.add(MaxPooling3D(pool_size=([2,2,2])))
    
    model.add(Conv3D(128, kernel_size=([2,2,2]), activation='relu'))
    model.add(MaxPooling3D(pool_size=([3,3,3])))
    
    model.add(Conv3D(256, kernel_size=(2,2,2), activation='relu'))
    model.add(MaxPooling3D(pool_size=([3,3,3])))
    
    model.add(GaussianNoise(0.001))
    model.add(Flatten())
    #model.add(BatchNormalization())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
   
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    
    return model


# In[19]:


model = model()


# In[26]:


model.summary()


# In[32]:


hist = model.fit(X_train,
                 y_train,
                 validation_split=0.2,
                 batch_size = 3,
                 epochs = 10,
                 #callbacks = [early_stopping],
                 shuffle=True)


# In[15]:


def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))


# In[24]:


def plot_history(history):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig('./../result/model_accuracy.png')
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig('./../result/model_loss.png')
    plt.close()


# In[25]:


plot_history(hist)


# In[78]:


def plot_now(history):
    plt.subplot(121)
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')

    plt.subplot(122)
    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')


# In[79]:


plot_now(hist)


# In[81]:


loss, acc = model.evaluate(X_test, y_test, verbose=0)


# In[60]:


import pandas as pd


# In[74]:


test_pred = X_test[:17]
test_pred = model.predict_classes(test_pred)
test_true = y_test[:17]
test_true = test_true.astype(int)
test_pred = test_pred.squeeze()
pd.crosstab(test_true, test_pred, rownames=['label'], colnames=['predict'])


# In[75]:


test_pred = X_test[17:34]
test_pred = model.predict_classes(test_pred)
test_true = y_test[17:34]
test_true = test_true.astype(int)
test_pred = test_pred.squeeze()
pd.crosstab(test_true, test_pred, rownames=['label'], colnames=['predict'])


# In[76]:


test_pred = X_test[34:51]
test_pred = model.predict_classes(test_pred)
test_true = y_test[34:51]
test_true = test_true.astype(int)
test_pred = test_pred.squeeze()
pd.crosstab(test_true, test_pred, rownames=['label'], colnames=['predict'])


# In[77]:


test_pred = X_test[51:57]
test_pred = model.predict_classes(test_pred)
test_true = y_test[51:57]
test_true = test_true.astype(int)
test_pred = test_pred.squeeze()
pd.crosstab(test_true, test_pred, rownames=['label'], colnames=['predict'])


# In[65]:


(57-13.0) / 57

