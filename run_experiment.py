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
parser.add_argument('--save_name', type=str, help='The file name of the checkpoint saved in the Weights folder. Default: experiment', default='experiment')
parser.add_argument('--estimator_lr', type=float, default=1e-4, help='Initial estimator learning rate before decay. Default: 1e-4')
parser.add_argument('--estimator_train_steps', type=int, default=10000, help='Number of training steps for estimator. Default: 10000')
parser.add_argument('--estimator_batch_size', type=int, default=128, help='Batch size for estimator Default: 128')
parser.add_argument('--phi', type=float, default=50.0, help='phi. Default: 50.0')
parser.add_argument('--lambda1', type=float, default=4.0, help='lambda1. Default: 4.0')
parser.add_argument('--lambda2', type=float, default=0.1, help='lambda2. Default: 0.1')
parser.add_argument('--disc_lr', type=float, default=1e-4, help='Initial discriminator learning rate before decay. Default: 1e-4')
parser.add_argument('--gen_lr', type=float, default=1e-4, help='Initial discriminator learning rate before decay. Default: 1e-4')
parser.add_argument('--train_steps', type=int, default=50000, help='Number of training steps. Default: 50000')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size used for GAN training. Default: 32')

parser.add_argument('--custom_data', type=bool, default=False, help='If custom data is being used then add this flag and indicate the names of parameters in custom_param values.')
parser.add_argument('--param', type=str, default='ratio', help='The parameter to train for. Default: ratio. Either one of: ratio, volume, both')

parser.add_argument('--custom_dataset', type=str, default=None, help='The name of the dataset in the data folder. Default: None. Depends on the custom dataset.')
parser.add_argument('--custom_param1', type=str, default=None, help='The parameter to train for(Only if using custom dataset). Default: None. Depends on the custom dataset.')
parser.add_argument('--custom_param2', type=str, default=None, help='The parameter to train for(Only if using custom dataset). Default: None. Depends on the custom dataset.')

args = parser.parse_args()

#load data
print('Loading Data...')

if not args.custom_data:
    data_train = h5py.File(args.data + '/airplane_z_train.hdf5', 'r')
    data_test = h5py.File(args.data + '/airplane_z_test.hdf5', 'r')
    Z_train = data_train.get('zs')[:]
    Z_test = data_test.get('zs')[:]
    Y_train = np.concatenate([data_train.get('aspect_ratio')[:],data_train.get('volume')[:]],1)
    Y_test = np.concatenate([data_test.get('aspect_ratio')[:],data_test.get('volume')[:]],1)
    Z = np.concatenate([Z_train,Z_test],0)
    Y = np.concatenate([Y_train,Y_test],0)

    # Normalize Labels
    min_y = Y.min(0)
    max_y = Y.max(0)
    Y = (Y-min_y)/(max_y-min_y)
    Y_train = (Y_train-min_y)/(max_y-min_y)
    Y_test = (Y_test-min_y)/(max_y-min_y)

    # Remove Very Sparse Regions
    Z = Z[np.logical_and(Y[:,0]<=0.7,Y[:,1]<=0.5),:]
    Y = Y[np.logical_and(Y[:,0]<=0.7,Y[:,1]<=0.5),:]
    Z_train = Z_train[np.logical_and(Y_train[:,0]<=0.7,Y_train[:,1]<=0.5),:]
    Y_train = Y_train[np.logical_and(Y_train[:,0]<=0.7,Y_train[:,1]<=0.5),:]
    Z_test = Z_test[np.logical_and(Y_test[:,0]<=0.7,Y_test[:,1]<=0.5),:]
    Y_test = Y_test[np.logical_and(Y_test[:,0]<=0.7,Y_test[:,1]<=0.5),:]
    Y = Y/[0.7,0.5]
    Y_train = Y_train/[0.7,0.5]
    Y_test = Y_test/[0.7,0.5]

    if args.param == 'ratio':
        Y = Y[:,0:1]
        Y_test = Y_test[:,0:1]
        Y_train = Y_train[:,0:1]
    elif args.param == 'volume':
        Y = Y[:,1:2]
        Y_test = Y_test[:,1:2]
        Y_train = Y_train[:,1:2]


    if args.param == 'both':
        model = MORangeGAN(Y,lambda1=args.lambda1,phi=args.phi,lambda2=args.lambda2)
    else:
        model = SORangeGAN(Y=Y,lambda1=args.lambda1,phi=args.phi,lambda2=args.lambda2,inf_lstd=-10.0)

else:
    data_train = h5py.File(args.data + '/' + args.custom_dataset + '_train.hdf5', 'r')
    data_test = h5py.File(args.data + '/' + args.custom_dataset + '_test.hdf5', 'r')
    Z_train = data_train.get('zs')[:]
    Z_test = data_test.get('zs')[:]

    if args.param2:
        Y_train = np.concatenate([data_train.get(args.params1)[:],data_train.get(args.params2)[:]],1)
        Y_test = np.concatenate([data_test.get(args.params1)[:],data_test.get(args.params2)[:]],1)
    else:
        Y_train = np.concatenate([data_train.get(args.params1)[:]],1)
        Y_test = np.concatenate([data_test.get(args.params1)[:]],1)
    
    Z = np.concatenate([Z_train,Z_test],0)
    Y = np.concatenate([Y_train,Y_test],0)
    min_y = Y.min(0)
    max_y = Y.max(0)
    Y = (Y-min_y)/(max_y-min_y)
    Y_train = (Y_train-min_y)/(max_y-min_y)
    Y_test = (Y_test-min_y)/(max_y-min_y)


    if args.param2:
        model = MORangeGAN(Y,lambda1=args.lambda1,phi=args.phi,lambda2=args.lambda2)
    else:
        model = SORangeGAN(Y=Y,lambda1=args.lambda1,phi=args.phi,lambda2=args.lambda2,inf_lstd=-10.0)

#train the regressor
print('training regressor...')
model.train_regressor(Z_train, Y_train.astype(np.float32), Z_test, Y_test.astype(np.float32), early_stop_save='./Weights/Regressor/'+args.save_name, batch_size=args.estimator_batch_size , train_steps=args.estimator_train_steps, lr=args.estimator_lr)

#train the model
print('training Range-GAN...')
model.train(Z,Y,train_steps=args.train_steps,disc_lr=args.disc_lr,gen_lr=args.gen_lr,batch_size=args.batch_size)

#save model
print('saving model...')
model.generator.save_weights('./Weights/Generator/'+args.save_name)
model.discriminator.save_weights('./Weights/Discriminator/'+args.save_name)

#Loading IMAE
print('Loading IMAE...')
imnet_gen = IMNET_GEN_TF2()
imnet_gen.load_from_tf1_checkpoint('./Weights/IMAE/IMAE.model-399')

#evaluate model
print('Evaluating ...')
if not args.custom_data:
    if args.param == 'both':
        performance_plot_MO(model, Y, './results/'+args.save_name)
    elif args.param == 'ratio':
        performance_plot_SO(model, Y, './results/'+args.save_name, 'Aspect Ratio')
        real_performance_plot_SO(model, Y, './results/'+args.save_name, imnet_gen, min_y[0], max_y[0], 0.7, 64)
    elif args.param == 'volume':
        performance_plot_SO(model, Y, './results/'+args.save_name, 'Volume Ratio')
        real_performance_plot_SO(model, Y, './results/'+args.save_name, imnet_gen, min_y[1], max_y[1], 0.5, 64)
else:
    if args.param2:
        performance_plot_MO(model, Y, './results/'+args.save_name)
    else:
        performance_plot_SO(model, Y, './results/'+args.save_name, args.param1)
        real_performance_plot_SO(model, Y, './results/'+args.save_name, imnet_gen, min_y[0], max_y[0], 1.0, 64)