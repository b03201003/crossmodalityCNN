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

def MMEncoder(inputTensor):#Conv+BN+ReLU / MaxPooling
	encodedTensor = Conv2D(filters=4,kernel_size=(3,3),padding='same')(inputTensor)
	encodedTensor = BatchNormalization()(encodedTensor)
	encodedTensor = Activation('relu')(encodedTensor)
	encodedTensor = MaxPooling2D(pool_size=(2,2))(encodedTensor)
	encodedTensor = Conv2D(filters=8,kernel_size=(3,3),padding='same')(encodedTensor)
	encodedTensor = BatchNormalization()(encodedTensor)
	encodedTensor = Activation('relu')(encodedTensor)
	encodedTensor = MaxPooling2D(pool_size=(2,2))(encodedTensor)
	encodedTensor = Conv2D(filters=16,kernel_size=(3,3),padding='same')(encodedTensor)
	encodedTensor = BatchNormalization()(encodedTensor)
	encodedTensor = Activation('relu')(encodedTensor)
	encodedTensor = MaxPooling2D(pool_size=(2,2))(encodedTensor)
	encodedTensor = Conv2D(filters=32,kernel_size=(3,3),padding='same')(encodedTensor)
	encodedTensor = BatchNormalization()(encodedTensor)
	encodedTensor = Activation('relu')(encodedTensor)
	encodedTensor = MaxPooling2D(pool_size=(2,2))(encodedTensor)
	return encodedTensor #(15,15,32)

Flair_x = Input(shape=(240,240,1))
T1ce_x = Input(shape=(240,240,1))
T1_x = Input(shape=(240,240,1))
T2_x = Input(shape=(240,240,1))


encodedFlair_x = MMEncoder(Flair_x)
encodedT1ce_x = MMEncoder(T1ce_x)
encodedT1_x = MMEncoder(T1_x)
encodedT2_x = MMEncoder(T2_x)
my_stack = Lambda(lambda x: K.stack([x[0],x[1],x[2],x[3]],axis=-1))
my_stack.trainable = True
CNNinput = my_stack([encodedFlair_x,encodedT1ce_x,encodedT1_x,encodedT2_x])
#CNNinput = K.stack([encodedFlair_x,encodedT1ce_x,encodedT1_x,encodedT2_x],axis=-1)#(15,15,32,4)
def CrossModalityCovolution(CNNinput):
	#cross modality part:
	CNNinput = Permute((1, 2, 4, 3))(CNNinput)#(15,15,4,32)
	#3D CNN part:
	CLSTMinput = Conv3D(filters=32,kernel_size=(1,1,4))(CNNinput) #(15,15,1,128)
	CLSTMinput = K.reshape(CLSTMinput,[-1,15,15,32])
	return CLSTMinput

CLSTMinput = CrossModalityCovolution(CNNinput)
# def CLSTM(CLSTMinput):#(15,15,128)??
# 	return CLSTMinput #
#CLSTMoutput = Lambda(CLSTM)(CLSTMinput)
def Decoder(CLSTMoutput):#Conv+BN+ReLU / Upsampling
	decodedImage = Conv2D(filters=16,kernel_size=(3,3),padding='same')(CLSTMoutput)
	decodedImage = BatchNormalization()(decodedImage)
	decodedImage = Activation('relu')(decodedImage)
	decodedImage = UpSampling2D(size=(2,2))(decodedImage)
	decodedImage = Conv2D(filters=8,kernel_size=(3,3),padding='same')(decodedImage)
	decodedImage = BatchNormalization()(decodedImage)
	decodedImage = Activation('relu')(decodedImage)
	decodedImage = UpSampling2D(size=(2,2))(decodedImage)
	decodedImage = Conv2D(filters=4,kernel_size=(3,3),padding='same')(decodedImage)
	decodedImage = BatchNormalization()(decodedImage)
	decodedImage = Activation('relu')(decodedImage)
	decodedImage = UpSampling2D(size=(2,2))(decodedImage)
	decodedImage = Conv2D(filters=1,kernel_size=(3,3),padding='same')(decodedImage)
	decodedImage = BatchNormalization()(decodedImage)
	decodedImage = Activation('relu')(decodedImage)
	decodedImage = UpSampling2D(size=(2,2))(decodedImage)
	return decodedImage
#decodedImage = Lambda(Decoder)(CLSTMoutput)
decodedImage = Decoder(CLSTMinput)


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

