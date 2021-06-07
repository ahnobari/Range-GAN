import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from tqdm.autonotebook import trange
from tensorflow import keras
from tensorflow.keras import layers
import math

class ConditionalBatchNormalization1D(keras.layers.Layer):
    def __init__(self,dim):
        super(ConditionalBatchNormalization1D, self).__init__()
        
        self.batch_norm = keras.layers.BatchNormalization(trainable=False)
        self.gamma_dense = keras.layers.Dense(dim)
        self.beta_dense = keras.layers.Dense(dim)
        
    def call(self, inputs, condition):
        
        x = self.batch_norm(inputs)
        
        gamma = self.gamma_dense(condition)
        beta = self.beta_dense(condition)
        
        x = x + x*gamma + beta
        
        return x

class Regressor(keras.Model):
    def __init__(self, cond_dim = 1):
        super(Regressor, self).__init__()
        
        self.Dense1 = keras.layers.Dense(512)
        self.batchnorm1 = keras.layers.BatchNormalization()
        self.SELU1 = keras.layers.Activation(tf.keras.activations.selu)
        
        self.Dense2 = keras.layers.Dense(512)
        self.batchnorm2 = keras.layers.BatchNormalization()
        self.SELU2 = keras.layers.Activation(tf.keras.activations.selu)
        
        self.Dense3 = keras.layers.Dense(256)
        self.batchnorm3 = keras.layers.BatchNormalization()
        self.SELU3 = keras.layers.Activation(tf.keras.activations.selu)

        self.Dense4 = keras.layers.Dense(256)
        self.batchnorm4 = keras.layers.BatchNormalization()
        self.SELU4 = keras.layers.Activation(tf.keras.activations.selu)

        self.Dense5 = keras.layers.Dense(128)
        self.batchnorm5 = keras.layers.BatchNormalization()
        self.SELU5 = keras.layers.Activation(tf.keras.activations.selu)
        
        self.Dense6 = keras.layers.Dense(cond_dim)
        self.out_activation = keras.layers.Activation(keras.activations.sigmoid)
    
    def call(self, inputs, training=True):
        
        x = self.Dense1(inputs)
        res = x
        x = self.batchnorm1(x,training=training)
        x = self.SELU1(x)

        x = self.Dense2(x)
        x = x + res
        x = self.batchnorm2(x,training=training)
        x = self.SELU2(x)

        x = self.Dense3(x)
        res = x
        x = self.batchnorm3(x,training=training)
        x = self.SELU3(x)

        x = self.Dense4(x)
        x = x + res
        x = self.batchnorm4(x,training=training)
        x = self.SELU2(x)

        x = self.Dense5(x)
        x = self.batchnorm5(x,training=training)
        x = self.SELU5(x)

        x = self.Dense6(x)
        x = self.out_activation(x)
        
        return x
    
class Regressor_u(keras.Model):
    def __init__(self, cond_dim = 1):
        super(Regressor_u, self).__init__()
        
        self.Dense1 = keras.layers.Dense(512)
        self.batchnorm1 = keras.layers.BatchNormalization()
        self.SELU1 = keras.layers.Activation(tf.keras.activations.selu)
        
        self.Dense2 = keras.layers.Dense(512)
        self.batchnorm2 = keras.layers.BatchNormalization()
        self.SELU2 = keras.layers.Activation(tf.keras.activations.selu)
        
        self.Dense3 = keras.layers.Dense(256)
        self.batchnorm3 = keras.layers.BatchNormalization()
        self.SELU3 = keras.layers.Activation(tf.keras.activations.selu)

        self.Dense4 = keras.layers.Dense(256)
        self.batchnorm4 = keras.layers.BatchNormalization()
        self.SELU4 = keras.layers.Activation(tf.keras.activations.selu)

        self.Dense5 = keras.layers.Dense(128)
        self.batchnorm5 = keras.layers.BatchNormalization()
        self.SELU5 = keras.layers.Activation(tf.keras.activations.selu)
        
        self.Dense6 = keras.layers.Dense(cond_dim)
        self.Dense_log_std = keras.layers.Dense(cond_dim)
        self.out_activation = keras.layers.Activation(keras.activations.sigmoid)
    
    def call(self, inputs, training=True):
        
        x = self.Dense1(inputs)
        res = x
        x = self.batchnorm1(x,training=training)
        x = self.SELU1(x)

        x = self.Dense2(x)
        x = x + res
        x = self.batchnorm2(x,training=training)
        x = self.SELU2(x)

        x = self.Dense3(x)
        res = x
        x = self.batchnorm3(x,training=training)
        x = self.SELU3(x)

        x = self.Dense4(x)
        x = x + res
        x = self.batchnorm4(x,training=training)
        x = self.SELU2(x)

        x = self.Dense5(x)
        x = self.batchnorm5(x,training=training)
        x = self.SELU5(x)
        
        
        lstd = self.Dense_log_std(x)
        x = self.Dense6(x)
        x = self.out_activation(x)
        
        return x, lstd
    
class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.Dense1 = keras.layers.Dense(2048)
        self.batchnorm1 = keras.layers.BatchNormalization()
        self.SELU1 = keras.layers.Activation(tf.keras.activations.selu)
        
        self.Dense2 = keras.layers.Dense(2048)
        self.batchnorm2 = keras.layers.BatchNormalization()
        self.SELU2 = keras.layers.Activation(tf.keras.activations.selu)
        
#         self.Dense3 = keras.layers.Dense(1024)
#         self.batchnorm3 = keras.layers.BatchNormalization()
#         self.SELU3 = keras.layers.Activation(tf.keras.activations.selu)
        
#         self.Dense4 = keras.layers.Dense(1024)
#         self.batchnorm4 = keras.layers.BatchNormalization()
#         self.SELU4 = keras.layers.Activation(tf.keras.activations.selu)
        
        self.Dense5 = keras.layers.Dense(1)
    
    def call(self, inputs, condition):

        x = inputs

        x = self.Dense1(x)
        x = self.batchnorm1(x)
        x = self.SELU1(x)
        
        x = self.Dense2(x)
        x = self.batchnorm2(x)
        x = self.SELU2(x)
        
#         x = self.Dense3(x)
#         x = self.batchnorm3(x)
#         x = self.SELU3(x)
        
#         x = self.Dense4(x)
#         x = self.batchnorm4(x)
#         x = self.SELU4(x)

        x = self.Dense5(x)

        return x
    
class Generator(keras.Model):
    def __init__(self, data_dim):
        super(Generator, self).__init__()
        
        self.cond_dense = keras.layers.Dense(128)
        self.cond_SELU = keras.layers.Activation(tf.keras.activations.selu)

        self.Dense1 = keras.layers.Dense(512)
        self.cond_batchnorm1  = ConditionalBatchNormalization1D(512)
        self.SELU1 = keras.layers.Activation(tf.keras.activations.selu)
        
        self.Dense2 = keras.layers.Dense(512)
        self.cond_batchnorm2  = ConditionalBatchNormalization1D(512)
        self.SELU2 = keras.layers.Activation(tf.keras.activations.selu)

        self.Dense3 = keras.layers.Dense(1024)
        self.cond_batchnorm3  = ConditionalBatchNormalization1D(1024)
        self.SELU3 = keras.layers.Activation(tf.keras.activations.selu)
        
        self.Dense4 = keras.layers.Dense(1024)
        self.cond_batchnorm4  = ConditionalBatchNormalization1D(1024)
        self.SELU4 = keras.layers.Activation(tf.keras.activations.selu)
        
        self.Dense5 = keras.layers.Dense(2048)
        self.cond_batchnorm5  = ConditionalBatchNormalization1D(2048)
        self.SELU5 = keras.layers.Activation(tf.keras.activations.selu)
        
        self.Dense6 = keras.layers.Dense(2048)
        self.cond_batchnorm6  = ConditionalBatchNormalization1D(2048)
        self.SELU6 = keras.layers.Activation(tf.keras.activations.selu)
        
        self.Dense7 = keras.layers.Dense(data_dim)
        self.out_activation = keras.layers.Activation(keras.activations.sigmoid)
        
    def call(self, inputs, condition):
        
        cond = self.cond_dense(condition)
        cond = self.cond_SELU(cond)

        x = inputs

        x = self.Dense1(x)
        x = self.cond_batchnorm1(x,cond)
        x = self.SELU1(x)
        
        x = self.Dense2(x)
        x = self.cond_batchnorm2(x,cond)
        x = self.SELU2(x)
        
        x = self.Dense3(x)
        x = self.cond_batchnorm3(x,cond)
        x = self.SELU3(x)
        
        x = self.Dense4(x)
        x = self.cond_batchnorm4(x,cond)
        x = self.SELU4(x)

        x = self.Dense5(x)
        x = self.cond_batchnorm5(x,cond)
        x = self.SELU5(x)
        
        x = self.Dense6(x)
        x = self.cond_batchnorm6(x,cond)
        x = self.SELU6(x)

        x = self.Dense7(x)
        x = self.out_activation(x)
        
        return x