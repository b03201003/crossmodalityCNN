import os
import glob
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
import sys
import keras
from keras import backend as K
from keras import regularizers
from keras.models import Sequential,Model
from keras.layers import Lambda,Input,Conv2D,Conv3D,Activation,UpSampling2D,MaxPooling2D,BatchNormalization,Dense,Dropout,Permute,Reshape
from keras.optimizers import Nadam
from sklearn.model_selection import train_test_split
from keras.models import load_model
from mylayers import *
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]

def Get_nib_dataNlabel(paths):#paths: list of path (glob)
	data = []
	label = []
	modalities = []
	c = 0
	for path in paths:
		if os.path.isdir(path):
			c+=5
			print "open file counter:",c
			dir_name = path.split('/')[-1]
			print "Now in ",dir_name
			#with nib.load(path+'/'+dir_name+'_flair.nii') as nibfile:
			nibfile = nib.load(path+'/'+dir_name+'_flair.nii')
			image = nibfile.get_data()
			print "flair image.shape:",image.shape
			modalities.append(image)
			#print "nibfile.header:",nibfile.header
			del nibfile
			#with nib.load(path+'/'+dir_name+'_t1ce.nii') as nibfile:	
			nibfile = nib.load(path+'/'+dir_name+'_t1ce.nii')
			image = nibfile.get_data()
			print "t1ce image.shape:",image.shape
			modalities.append(image)
			del nibfile
			#with nib.load(path+'/'+dir_name+'_t1.nii') as nibfile:
			nibfile = nib.load(path+'/'+dir_name+'_t1.nii')
			image = nibfile.get_data()
			print "t1 image.shape:",image.shape
			modalities.append(image)
			del nibfile
			#with nib.load(path+'/'+dir_name+'_t2.nii') as nibfile:
			nibfile = nib.load(path+'/'+dir_name+'_t2.nii')
			image = nibfile.get_data()
			print "t2image.shape:",image.shape
			modalities.append(image)
			del nibfile

			data.append(modalities)
			modalities = []

			#with nib.load(path+'/'+dir_name+'_seg.nii') as nibfile:
			nibfile = nib.load(path+'/'+dir_name+'_seg.nii')
			image = nibfile.get_data()
			print "label image.shape:",image.shape
			label.append(image)
			del nibfile

	return np.array(data),np.array(label)

if sys.argv[1]=='debug':
	HGGpaths = glob.glob('/home/u/b03201003/Testing/HGG/*')
	LGGpaths = glob.glob('/home/u/b03201003/Testing/LGG/*')
elif sys.argv[1]=='train':
	HGGpaths = glob.glob('/home/u/b03201003/BraTS17/HGG/*')
	LGGpaths = glob.glob('/home/u/b03201003/BraTS17/LGG/*')
else:
	raise AttributeError("Only debug and train mode can be choosed")
HGGdata,HGGlabel = Get_nib_dataNlabel(HGGpaths)
LGGdata,LGGlabel = Get_nib_dataNlabel(LGGpaths)
print "HGGdata.shape,HGGlabel.shape:",HGGdata.shape,HGGlabel.shape#(None, 4, 240, 240, 155) (None, 240, 240, 155)
print "LGGdata.shape,LGGlabel.shape:",LGGdata.shape,LGGlabel.shape#(None, 4, 240, 240, 155) (None, 240, 240, 155)
#20% test set
print "HGGdata[len(HGGdata)/5:].shape,LGGdata[len(LGGdata)/5:].shape:",HGGdata[len(HGGdata)/5:].shape,LGGdata[len(LGGdata)/5:].shape
data_train = np.concatenate((HGGdata[len(HGGdata)/5:],LGGdata[len(LGGdata)/5:]))/4.0
label = np.concatenate((HGGlabel[len(HGGlabel)/5:],LGGlabel[len(LGGlabel)/5:]))/4.0 #Needed?
X_test = np.concatenate((HGGdata[:len(HGGdata)/5],LGGdata[:len(LGGdata)/5]))/4.0
Y_test = np.concatenate((HGGlabel[:len(HGGlabel)/5],LGGlabel[:len(LGGlabel)/5]))/4.0
print "data_train.shape,label.shape,X_test.shape,Y_test.shape:",data_train.shape,label.shape,X_test.shape,Y_test.shape
data_train  = data_train.reshape((4,-1,240,240,1))
label = label.reshape((-1,240,240,1))
X_test =  X_test.reshape((4,-1,240,240,1))
Y_test = Y_test.reshape((-1,240,240,1))
print "reshaped data_train.shape,label.shape,X_test.shape,Y_test.shape:",data_train.shape,label.shape,X_test.shape,Y_test.shape


Flair_x = Input(shape=(240,240,1))
T1ce_x = Input(shape=(240,240,1))
T1_x = Input(shape=(240,240,1))
T2_x = Input(shape=(240,240,1))

filters_list = [4,8,16,32]
encodeORdecode = 'encode'
encodedFlair_x = MySegnet(filters_list=filters_list,encodeORdecode=encodeORdecode)(Flair_x)
encodedT1ce_x = MySegnet(filters_list=filters_list,encodeORdecode=encodeORdecode)(T1ce_x)
encodedT1_x = MySegnet(filters_list=filters_list,encodeORdecode=encodeORdecode)(T1_x)
encodedT2_x = MySegnet(filters_list=filters_list,encodeORdecode=encodeORdecode)(T2_x)
my_stack = Lambda(lambda x: K.stack([x[0],x[1],x[2],x[3]],axis=-1))
CNNinput = my_stack([encodedFlair_x,encodedT1ce_x,encodedT1_x,encodedT2_x])
#CNNinput = K.stack([encodedFlair_x,encodedT1ce_x,encodedT1_x,encodedT2_x],axis=-1)#(15,15,32,4)
CrossModalityCNNmodel = Model(inputs=[Flair_x,T1ce_x,T1_x,T2_x],outputs=CNNinput)
CrossModalityCNNmodel.compile(optimizer='nadam',loss = 'mse')
CrossModalityCNNmodel.summary()
#testing
CLSTMinput = CrossModalityCovolution()(CNNinput)

filters_list = [16,8,4,1]
encodeORdecode = 'decode'
decodedImage = MySegnet(filters_list=filters_list,encodeORdecode=encodeORdecode)(CLSTMinput)


CrossModalityCNNmodel = Model(inputs=[Flair_x,T1ce_x,T1_x,T2_x],outputs=decodedImage)
CrossModalityCNNmodel.compile(optimizer='nadam',loss = 'mse')
CrossModalityCNNmodel.summary()

save_model_file = 'myModel.h5'
if os.path.isfile(save_model_file):
	CrossModalityCNNmodel = load_model(save_model_file)


input_list = [data_train[0],data_train[1],data_train[2],data_train[3]]
batch_size=32
TrainEpochs = 100
for i in range(TrainEpochs):
	History = CrossModalityCNNmodel.fit(x=input_list,y=label,batch_size=batch_size,epochs=1,validation_split=0.2)

loss,acc = CrossModalityCNNmodel.evaluate(x=X_test,y=Y_test,batch_size=batch_size)
print "\nloss:%.2f,acc:%.2f%%"%(loss,acc*100)
CrossModalityCNNmodel.save('myModel.h5')

