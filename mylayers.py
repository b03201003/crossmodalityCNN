import os
import numpy as np
import sys
import keras
from keras import backend as K
#from keras.engine.topology import Layer
from keras.engine import InputSpec,Layer
from keras.utils import conv_utils
from keras import activations
from keras import regularizers
from keras import initializers


from keras.models import Sequential,Model
from keras.layers import Lambda,Input,Conv2D,Conv3D,Activation,UpSampling2D,MaxPooling2D,BatchNormalization,Dense,Dropout,Permute,Reshape
from keras.optimizers import Nadam
from sklearn.model_selection import train_test_split

class MySegNetLayer(Layer):
    def __init__(self, **kwargs):#default channel last
        super(MySegNetLayer, self).__init__(**kwargs)
       	self.filters = 1
       	self.encodeORdecode = 'encode'
       	self.rank = 2
        self.kernel_size = conv_utils.normalize_tuple((3,3), self.rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple((1,1), self.rank, 'strides')
        self.padding = 'same'        
        self.dilation_rate = conv_utils.normalize_tuple((1,1), self.rank, 'dilation_rate')
        self.activation = activations.get('relu')
        self.use_bias = True
        self.kernel_initializer = initializers.get('glorot_uniform')
        self.bias_initializer = initializers.get('zeros')
        self.kernel_regularizer = None
        self.bias_regularizer = None
        self.activity_regularizer = None
        self.kernel_constraint = None
        self.bias_constraint = None
        self.input_spec = InputSpec(ndim=self.rank + 2)
        
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        channel_axis = -1 #default
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
       	self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True
        super(MySegNetLayer, self).build(input_shape)  # Be sure to call this at the end(what this???)
    def call(self, inputs):
    	outputs = K.conv2d(
                inputs,
                self.kernel,
                strides=self.strides,
                padding=self.padding,
                dilation_rate=self.dilation_rate)
    	outputs = BatchNormalization()(outputs)
    	outputs = self.activation(outputs)
    	if self.encodeORdecode == 'encode':
    		outputs = MaxPooling2D(pool_size=(2,2))(outputs)
    	elif self.encodeORdecode == 'decode':
    		outputs = UpSampling2D(size=(2,2))(outputs)
    	else:
    		raise ValueError("Must be encode or decode")

        return outputs
    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
	        new_dim = conv_utils.conv_output_length(
	        space[i],
	        self.kernel_size[i],
	        padding=self.padding,
	        stride=self.strides[i],
	        dilation=self.dilation_rate[i])
	        new_space.append(new_dim)
        return (input_shape[0],) + tuple(new_space) + (self.filters,)

class MySegnet(MySegNetLayer):
    def __init__(self,filters_list,encodeORdecode, **kwargs):
    	self.encodeORdecode = encodeORdecode
        super(MySegnet, self).__init__(**kwargs)
        self.filters_list = filters_list
        if len(filters_list)!=4:
        	raise ValueError("The filters list length must be 4!")
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.

        super(MySegnet, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):#4 MySegNetLayer
    	self.filters = self.filters_list[0]
    	outputs = super(MySegnet, self).call(inputs)
    	self.filters = self.filters_list[1]
    	outputs = super(MySegnet, self).call(outputs)
    	self.filters = self.filters_list[2]
    	outputs = super(MySegnet, self).call(outputs)
    	self.filters = self.filters_list[3]
    	outputs = super(MySegnet, self).call(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        super(MySegnet, self).compute_output_shape(input_shape)



class CrossModalityCovolution(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, CNNinput):
        CNNinput = Permute((1, 2, 4, 3))(CNNinput)#(15,15,4,32)
        #3D CNN part:
        CLSTMinput = Conv3D(filters=32,kernel_size=(1,1,4))(CNNinput) #(15,15,1,128)
        CLSTMinput = K.reshape(CLSTMinput,[-1,15,15,32])
        return CLSTMinput

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)



#TBC...
#
# class CLSTM(Layer):
#     def __init__(self, output_dim, **kwargs):
#         self.output_dim = output_dim
#         super(MyLayer, self).__init__(**kwargs)

#     def build(self, input_shape):
#         # Create a trainable weight variable for this layer.
#         self.kernel = self.add_weight(name='kernel', 
#                                       shape=(input_shape[1], self.output_dim),
#                                       initializer='uniform',
#                                       trainable=True)
#         super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

#     def call(self, x):
#         return K.dot(x, self.kernel)

#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.output_dim)