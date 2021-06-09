import argparse
import os
import numpy as np
from glob import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
import tensorflow_addons as tfa
from GANs import SORangeGAN, MORangeGAN
import matplotlib.animation as animation
from IPython.display import Video
import mcubes
from converter import convert_points
import math
from tqdm.autonotebook import trange
from IMNET_GEN_TF2 import *
import tensorflow_probability as tfp
from utils import *

parser = argparse.ArgumentParser(description='Estimator Traning')
parser.add_argument('--data', type=str, help='The path to the data. Default: ./data use ./augmented if you create an augmented dataset', default='./data')
parser.add_argument('--random_seed', type=int, default=None, help='Set the random seed used to generate samples.')
parser.add_argument('--save_name', type=str, help='The file name of the checkpoint saved in the Weights folder. Default: experiment', default='experiment')
parser.add_argument('--batch_size', type=int, default=200000, help='Batch size used for IMAE. Default: 200000. Change based on GPU performance.')
parser.add_argument('--resolution', type=int, default=256, help='Resolution of the voxel model generated. Default: 256. Determines the quality of samples.')
parser.add_argument('--n_samples', type=int, default=10, help='Number of samples to generate.')

parser.add_argument('--param_type', type=str, default='ratio', help='Which type of parameter. Default: ratio. Either of: ratio, volume, both, custom. is not custom the actual label will be measured.')

parser.add_argument('--only_valid', type=bool, default=False, help='If this flag is raised only samples that meet the input condition will be saved (less than n_samples). Only for non-custom cases.')

parser.add_argument('--param1_start', type=float, default=0.0, help='Start of range for parameter 1. Defaulr: 0.0')
parser.add_argument('--param1_end', type=float, default=0.1, help='end of range for paramter 1. Default: 0.1')

parser.add_argument('--param2_start', type=float, default=None, help='Start of range for parameter 2. Default: None. Only input if multi-objective model is used.')
parser.add_argument('--param2_end', type=float, default=None, help='End of range for parameter 2. Default: None. Only input if multi-objective model is used.')

args = parser.parse_args()

data_train = h5py.File(args.data + '/airplane_z_train.hdf5', 'r')
data_test = h5py.File(args.data + '/airplane_z_test.hdf5', 'r')
Y_train = np.concatenate([data_train.get('aspect_ratio')[:],data_train.get('volume')[:]],1)
Y_test = np.concatenate([data_test.get('aspect_ratio')[:],data_test.get('volume')[:]],1)
Y = np.concatenate([Y_train,Y_test],0)

# Normalize Labels
min_y = Y.min(0)
max_y = Y.max(0)



if args.param2_start and args.param2_end:
    model = MORangeGAN(np.array([[0],[1],[2]]),lambda1=0.0,phi=0.0,lambda2=0.0)
    condition = [[args.param1_start,args.param1_end,args.param2_start,args.param2_end]]
else:
    model = SORangeGAN(np.array([[0,0],[1,1],[2,2]]),lambda1=0.0,phi=0.0,lambda2=0.0)
    condition = [[args.param1_start,args.param1_end]]

print('Loading Model...')
model.generator.load_weights('./Weights/Generator/'+args.save_name)

if args.random_seed:
    tf.random.set_seed(1234)

print('Generating Samples...')
noise = tf.random.normal([args.n_samples, 64])
zs = np.array(model.generator(noise,np.array(condition*args.n_samples)))

IMAE = IMNET_GEN_TF2()
IMAE.load_from_tf1_checkpoint('./Weights/IMAE/IMAE.model-399')

if args.only_valid and args.param_type != 'custom':
    print('Checking labels at low resolution...')
    vals,ratios,volumes = IMAE.synthesize(zs,64,args.batch_size)
    ratios = (np.array(ratios) - min_y[0])/(max_y[0]-min_y[0])/0.7
    volumes = (np.array(volumes) - min_y[1])/(max_y[1]-min_y[1])/0.5

    if args.param_type == 'both':
        zs = zs[np.where(np.logical_and(ratios>=args.param1_start,ratios<=args.param1_end,volumes>=args.param2_start,volumes<=args.param2_end))]
    elif args.param_type == 'ratio':
        zs = zs[np.where(np.logical_and(ratios>=args.param1_start,ratios<=args.param1_end))]
    elif args.param_type == 'volume':
        zs = zs[np.where(np.logical_and(volumes>=args.param1_start,volumes<=args.param1_end))]

vals,ratios,volumes = IMAE.synthesize(zs,args.resolution,args.batch_size)
ratios = (np.array(ratios) - min_y[0])/(max_y[0]-min_y[0])/0.7
volumes = (np.array(volumes) - min_y[1])/(max_y[1]-min_y[1])/0.5

for i in range(zs.shape[0]):
    IMAE.create_mesh(vals[i],args.resolution,'./samples/' + args.save_name + '_' + str(i) +  '_rat_' + str(ratios[i])  + '_vol_' + str(volumes[i]) + '.obj')