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

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

def Get_nib_dataNlabel(paths):#paths: list of path (glob)
	data = []
	label = []
	modalities = []
	for path in paths:
		if os.path.isdir(path):
			dir_name = path.split('/')[-2]
			print "Now in ",dir_name
			nibfile = nib.load(path+'/'+dir_name+'_flair.nii')
			image = nibfile.get_data()
			print "flair image.shape:",image.shape
			modalities.append(image)
			nibfile = nib.load(path+'/'+dir_name+'_t1ce.nii')
			image = nibfile.get_data()
			print "t1ce image.shape:",image.shape
			modalities.append(image)
			nibfile = nib.load(path+'/'+dir_name+'_t1.nii')
			image = nibfile.get_data()
			print "t1 image.shape:",image.shape
			modalities.append(image)
			nibfile = nib.load(path+'/'+dir_name+'_t2.nii')
			image = nibfile.get_data()
			print "t2image.shape:",image.shape
			modalities.append(image)

			data.append(modalities)

			nibfile = nib.load(path+'/'+dir_name+'_seg.nii')
			image = nibfile.get_data()
			print "label image.shape:",image.shape
			label.append(image)

	return np.array(data),np.array(label)



#HGGpaths = glob.glob('/data/BraTS/BraTS17/MICCAI_BraTS17_Data_Training/HGG/*')
HGGpaths = glob.glob('~/Testing/HGG/*')
HGGdata,HGGlabel = Get_nib_data(HGGpaths)
#LGGpaths = glob.glob('/data/BraTS/BraTS17/MICCAI_BraTS17_Data_Training/LGG/*')
LGGpaths = glob.glob('~/Testing/LGG/*')
LGGdata,LGGlabel = Get_nib_data(LGGpaths)

#20% test set
data_train = np.concatenate((HGGdata[len(HGGdata)/5:],LGGdata[len(LGGdata)/5:]))/4.0
label = np.concatenate((HGGlabel[len(HGGlabel)/5:],LGGlabel[len(LGGlabel)/5:]))/4.0 #Needed?
X_test = np.concatenate((HGGdata[:len(HGGdata)/5],LGGdata[:len(LGGdata)/5]))/4.0
Y_test = np.concatenate((HGGlabel[:len(HGGlabel)/5],LGGlabel[:len(LGGlabel)/5]))/4.0
print "data_train.shape,label.shape,X_test.shape,Y_test.shape:",data_train.shape,label.shape,X_test.shape,Y_test.shape
data_train  = data_train.reshape((4,-1,240,240,1))
X_test =  X_test.reshape((4,-1,240,240,1))
print "reshaped data_train.shape,X_test.shape:",data_train.shape,X_test.shape

def MMEncoder(inputTensor):#Conv+BN+ReLU / MaxPooling
	encodedTensor = Conv2D(filters=4,kernai_size=(3,3),padding='same')(inputTensor)
	encodedTensor = BatchNormalization()(encodedTensor)
	encodedTensor = Activation('relu')(encodedTensor)
	encodedTensor = MaxPooling2D(pool_size=(2,2))(encodedTensor)
	encodedTensor = Conv2D(filters=8,kernai_size=(3,3),padding='same')(encodedTensor)
	encodedTensor = BatchNormalization()(encodedTensor)
	encodedTensor = Activation('relu')(encodedTensor)
	encodedTensor = MaxPooling2D(pool_size=(2,2))(encodedTensor)
	encodedTensor = Conv2D(filters=16,kernai_size=(3,3),padding='same')(encodedTensor)
	encodedTensor = BatchNormalization()(encodedTensor)
	encodedTensor = Activation('relu')(encodedTensor)
	encodedTensor = MaxPooling2D(pool_size=(2,2))(encodedTensor)
	encodedTensor = Conv2D(filters=32,kernai_size=(3,3),padding='same')(encodedTensor)
	encodedTensor = BatchNormalization()(encodedTensor)
	encodedTensor = Activation('relu')(encodedTensor)
	encodedTensor = MaxPooling2D(pool_size=(2,2))(encodedTensor)
	return encodedTensor #(15,15,32)

Flair_x = Input(Shape(240,240,1))
T1ce_x = Input(Shape(240,240,1))
T1_x = Input(Shape(240,240,1))
T2_x = Input(Shape(240,240,1))


encodedFlair_x = Lambda(MMEncoder)(Flair_x)
encodedT1ce_x = Lambda(MMEncoder)(T1ce_x)
encodedT1_x = Lambda(MMEncoder)(T1_x)
encodedT2_x = Lambda(MMEncoder)(T2_x)

def CrossModalityCovolution(encodedFlair_x,encodedT1ce_x,encodedT1_x,encodedT2_x):
	#cross modality part:
	CNNinput = K.stack([encodedFlair_x,encodedT1ce_x,encodedT1_x,encodedT2_x],axis=-1)#(15,15,32,4)
	CNNinput = Permute((0, 1, 3, 2))(CNNinput)#(15,15,4,32)
	#3D CNN part:
	CLSTMinput = Conv3D(filters=32,kernai_size=(1,1,4))(CNNinput) #(15,15,1,128)
	CLSTMinput = Reshape((15,15,32))
	return CLSTMinput

CLSTMinput = Lambda(CrossModalityCovolution)(encodedFlair_x,encodedT1ce_x,encodedT1_x,encodedT2_x)
def CLSTM(CLSTMinput):#(15,15,128)??


	return CLSTMinput #

#CLSTMoutput = Lambda(CLSTM)(CLSTMinput)
def Decoder(CLSTMoutput):#Conv+BN+ReLU / Upsampling
	decodedImage = Conv2D(filters=16,kernai_size=(3,3),padding='same')(CLSTMoutput)
	decodedImage = BatchNormalization()(decodedImage)
	decodedImage = Activation('relu')(decodedImage)
	decodedImage = UpSampling2D(size=(2,2))(decodedImage)
	decodedImage = Conv2D(filters=8,kernai_size=(3,3),padding='same')(encodedTensor)
	decodedImage = BatchNormalization()(decodedImage)
	decodedImage = Activation('relu')(decodedImage)
	decodedImage = UpSampling2D(size=(2,2))(decodedImage)
	decodedImage = Conv2D(filters=4,kernai_size=(3,3),padding='same')(encodedTensor)
	decodedImage = BatchNormalization()(decodedImage)
	decodedImage = Activation('relu')(decodedImage)
	decodedImage = UpSampling2D(size=(2,2))(decodedImage)
	decodedImage = Conv2D(filters=1,kernai_size=(3,3),padding='same')(encodedTensor)
	decodedImage = BatchNormalization()(decodedImage)
	decodedImage = Activation('relu')(decodedImage)
	decodedImage = UpSampling2D(size=(2,2))(decodedImage)
	return decodedImage
#decodedImage = Lambda(Decoder)(CLSTMoutput)
decodedImage = Lambda(Decoder)(CLSTMinput)

CrossModalityCNNmodel = Model(inputs=[Flair_x,T1ce_x,T1_x,T2_x],outputs=decodedImage)
CrossModalityCNNmodel.compile(optimizer='nadam',loss = 'mse')
CrossModalityCNNmodel.summary()




input_list = [data_train[0],data_train[1],data_train[2],data_train[3]]
batch_size=32
TrainEpochs = 100
for i in range(TrainEpochs):
	History = CrossModalityCNNmodel.fit(x=input_list,y=label,batch_size=batch_size,epochs=1,verbose=2,validation_split=0.2)

loss,acc = CrossModalityCNNmodel.evaluate(x=X_test,y=Y_test,batch_size=batch_size)
print "\nloss:%.2f,acc:%.2f%%"%(loss,acc*100)


