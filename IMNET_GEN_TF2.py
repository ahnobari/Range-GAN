import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
import tensorflow_addons as tfa
import matplotlib.animation as animation
from IPython.display import Video
import mcubes
from converter import convert_points
import math
from tqdm.autonotebook import trange


class IMNET_GEN_TF2(tf.keras.Model):
    def __init__(self, gf_dim = 128):
        super(IMNET_GEN_TF2, self).__init__()
        
        #Decoder/Generator
        self.Dense1 = tf.keras.layers.Dense(gf_dim*8)
        self.LReLU1 = tf.keras.layers.LeakyReLU(alpha=0.02)
        
        self.Dense2 = tf.keras.layers.Dense(gf_dim*8)
        self.LReLU2 = tf.keras.layers.LeakyReLU(alpha=0.02)
        
        self.Dense3 = tf.keras.layers.Dense(gf_dim*8)
        self.LReLU3 = tf.keras.layers.LeakyReLU(alpha=0.02)
        
        self.Dense4 = tf.keras.layers.Dense(gf_dim*4)
        self.LReLU4 = tf.keras.layers.LeakyReLU(alpha=0.02)
        
        self.Dense5 = tf.keras.layers.Dense(gf_dim*2)
        self.LReLU5 = tf.keras.layers.LeakyReLU(alpha=0.02)
        
        self.Dense6 = tf.keras.layers.Dense(gf_dim)
        self.LReLU6 = tf.keras.layers.LeakyReLU(alpha=0.02)
        
        self.Dense7 = tf.keras.layers.Dense(1)
        self.out_activation = tf.keras.layers.Activation(tf.keras.activations.sigmoid)

        #Encoder
        self.conv_1 = tf.keras.layers.Conv3D(32,4, strides=[2,2,2],padding='SAME')
        self.IN_1 = tfa.layers.InstanceNormalization()

        self.conv_2 = tf.keras.layers.Conv3D(64,4, strides=[2,2,2],padding='SAME')
        self.IN_2 = tfa.layers.InstanceNormalization()

        self.conv_3 = tf.keras.layers.Conv3D(128,4, strides=[2,2,2],padding='SAME')
        self.IN_3 = tfa.layers.InstanceNormalization()

        self.conv_4 = tf.keras.layers.Conv3D(256,4, strides=[2,2,2],padding='SAME')
        self.IN_4 = tfa.layers.InstanceNormalization()

        self.conv_5 = tf.keras.layers.Conv3D(256,4, strides=[2,2,2],padding='VALID')
    
    def call(self, z, points):
        
        batch_size = tf.shape(points)[0]
        zs = tf.tile(z,[batch_size,1])
        x = tf.concat([points,zs],-1)
        
        x = self.Dense1(x)
        x = self.LReLU1(x)
        
        x = self.Dense2(x)
        x = self.LReLU2(x)
        
        x = self.Dense3(x)
        x = self.LReLU3(x)
        
        x = self.Dense4(x)
        x = self.LReLU4(x)
        
        x = self.Dense5(x)
        x = self.LReLU5(x)
        
        x = self.Dense6(x)
        x = self.LReLU6(x)

        x = self.Dense7(x)
        x = self.out_activation(x)
        
        return x
    
    def get_call_aux(self):
        @tf.function
        def call_model(z,p):
            return self.call(z,p)
        return call_model
    
    def load_from_tf1_checkpoint(self, checkpoint_path, size = 256):
        
        loaded_vars = tf.train.load_checkpoint(checkpoint_path)
        
        #initialize weights

        #Decoder/Generator
        self.call(tf.random.normal([1,size]),tf.random.normal([1,3]))
        
        self.Dense1.weights[0].assign(loaded_vars.get_tensor('simple_net/h1_lin/Matrix'))
        self.Dense1.weights[1].assign(loaded_vars.get_tensor('simple_net/h1_lin/bias'))
        
        self.Dense2.weights[0].assign(loaded_vars.get_tensor('simple_net/h2_lin/Matrix'))
        self.Dense2.weights[1].assign(loaded_vars.get_tensor('simple_net/h2_lin/bias'))
        
        self.Dense3.weights[0].assign(loaded_vars.get_tensor('simple_net/h3_lin/Matrix'))
        self.Dense3.weights[1].assign(loaded_vars.get_tensor('simple_net/h3_lin/bias'))
        
        self.Dense4.weights[0].assign(loaded_vars.get_tensor('simple_net/h4_lin/Matrix'))
        self.Dense4.weights[1].assign(loaded_vars.get_tensor('simple_net/h4_lin/bias'))
        
        self.Dense5.weights[0].assign(loaded_vars.get_tensor('simple_net/h5_lin/Matrix'))
        self.Dense5.weights[1].assign(loaded_vars.get_tensor('simple_net/h5_lin/bias'))
    
        self.Dense6.weights[0].assign(loaded_vars.get_tensor('simple_net/h6_lin/Matrix'))
        self.Dense6.weights[1].assign(loaded_vars.get_tensor('simple_net/h6_lin/bias'))
        
        self.Dense7.weights[0].assign(loaded_vars.get_tensor('simple_net/h7_lin/Matrix'))
        self.Dense7.weights[1].assign(loaded_vars.get_tensor('simple_net/h7_lin/bias'))

        #Encoder
        self.get_Z(tf.random.normal([1,64,64,64,1]))
        self.conv_1.weights[0].assign(loaded_vars.get_tensor('encoder/conv_1/Matrix'))
        self.conv_1.weights[1].assign(loaded_vars.get_tensor('encoder/conv_1/bias'))

        self.conv_2.weights[0].assign(loaded_vars.get_tensor('encoder/conv_2/Matrix'))
        self.conv_2.weights[1].assign(loaded_vars.get_tensor('encoder/conv_2/bias'))

        self.conv_3.weights[0].assign(loaded_vars.get_tensor('encoder/conv_3/Matrix'))
        self.conv_3.weights[1].assign(loaded_vars.get_tensor('encoder/conv_3/bias'))

        self.conv_4.weights[0].assign(loaded_vars.get_tensor('encoder/conv_4/Matrix'))
        self.conv_4.weights[1].assign(loaded_vars.get_tensor('encoder/conv_4/bias'))

        self.conv_5.weights[0].assign(loaded_vars.get_tensor('encoder/conv_5/Matrix'))
        self.conv_5.weights[1].assign(loaded_vars.get_tensor('encoder/conv_5/bias'))

        self.IN_1.weights[0].assign(loaded_vars.get_tensor('encoder/InstanceNorm/gamma'))
        self.IN_1.weights[1].assign(loaded_vars.get_tensor('encoder/InstanceNorm/beta'))

        self.IN_2.weights[0].assign(loaded_vars.get_tensor('encoder/InstanceNorm_1/gamma'))
        self.IN_2.weights[1].assign(loaded_vars.get_tensor('encoder/InstanceNorm_1/beta'))

        self.IN_3.weights[0].assign(loaded_vars.get_tensor('encoder/InstanceNorm_2/gamma'))
        self.IN_3.weights[1].assign(loaded_vars.get_tensor('encoder/InstanceNorm_2/beta'))

        self.IN_4.weights[0].assign(loaded_vars.get_tensor('encoder/InstanceNorm_3/gamma'))
        self.IN_4.weights[1].assign(loaded_vars.get_tensor('encoder/InstanceNorm_3/beta'))
        
    
    def create_mesh(self,values, rez, save_fname=None):
        '''
        Create a mesh given values 
        '''
        values_grid = values.reshape(rez, rez, rez)
        vertices, triangles = mcubes.marching_cubes(values_grid, 0.5)
        if save_fname is not None:
            # Save mesh
            mcubes.export_obj(vertices, triangles, save_fname)

        return (vertices, triangles)
    
    def gen_grid(self,rez):
        X, Y, Z = np.mgrid[:rez, :rez, :rez]
        points_int = np.stack([X, Y, Z], axis=-1).reshape(-1,3)
        points = convert_points(points_int, rez)
        return points
    
    def get_Z(self,voxels, training=False):
        x = self.conv_1(voxels)
        x = self.LReLU1(self.IN_1(x,training=training))

        x = self.conv_2(x)
        x = self.LReLU1(self.IN_2(x,training=training))

        x = self.conv_3(x)
        x = self.LReLU1(self.IN_3(x,training=training))

        x = self.conv_4(x)
        x = self.LReLU1(self.IN_4(x,training=training))

        x = self.conv_5(x)
        z = tf.math.sigmoid(x)

        return z

    def synthesize(self, zs, res, batch_size=256**2):
        
        points = self.gen_grid(res).astype(np.float32)
        n_points = points.shape[0]
        
        call_graph = self.get_call_aux()
        
        if batch_size > n_points:
            batch_size = n_points
        vals = []
        ratios = []
        volumes = []
        total_steps = math.ceil(n_points/batch_size) * tf.shape(zs)[0]
        progress = trange(total_steps)
        for i in range(tf.shape(zs)[0]):
            z = zs[i:i+1,:]
            value = []
            for j in range(math.ceil(n_points/batch_size)):
                p_in = points[j*batch_size:(j+1)*batch_size]
                val = call_graph(z,p_in).numpy()
                val = val>0.5
                value.append(val.astype(np.bool))
                progress.update(1)
            value = np.concatenate(value,0)
            vals.append(value.astype(np.bool))

            size = np.max(points[value[:,0]], axis=0) - np.min(points[value[:,0]], axis=0)
            aspect_ratio = size[2]/size[0]
            volume = np.sum(value)/value.shape[0]
            ratios.append(aspect_ratio)
            volumes.append(volume)
        return vals, ratios, volumes