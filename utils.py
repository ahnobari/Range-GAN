import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tqdm.autonotebook import trange, tqdm
import tensorflow_probability as tfp
import os
import subprocess
import multiprocessing as mp
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
from scipy.spatial.distance import directed_hausdorff
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import matplotlib
from matplotlib import cm
from sklearn.metrics import pairwise_distances
import matplotlib.animation as animation

def quad_entrop(x):
    r = tf.reduce_sum(tf.math.square(x), axis=1, keepdims=True)
    D = r - 2 * tf.matmul(x, tf.transpose(x)) + tf.transpose(r)
    return tf.reduce_sum(D)/(x.shape[0]**2-x.shape[0])
    
    
def performance_plot_SO(model, Y, save_location, title):
    satisfaction_01 = []
    data_sat_01 = []
    entropy_01 = []
    range_size = 0.1
    batch_size = 2000
    n = 101
    for i in range(n):
        condition = [[i/(n-1)*(1-range_size),i/(n-1)*(1-range_size)+range_size]]
        ys = model.regressor(model.generator(tf.random.normal(shape=[batch_size,64]),np.array(condition*batch_size),training=False),training=False).numpy()
        sat = np.sum(np.logical_and(ys[:,0:1]<condition[0][1],ys[:,0:1]>condition[0][0]))/ys.shape[0]
        satisfaction_01.append(sat)
        sat = np.sum(np.logical_and(Y<condition[0][1],Y>condition[0][0]))/Y.shape[0]
        data_sat_01.append(sat)
        entropy_01.append(quad_entrop(ys))


    satisfaction_02 = []
    data_sat_02 = []
    entropy_02 = []
    range_size = 0.2
    batch_size = 2000
    n = 101
    for i in range(n):
        condition = [[i/(n-1)*(1-range_size),i/(n-1)*(1-range_size)+range_size]]
        ys = model.regressor(model.generator(tf.random.normal(shape=[batch_size,64]),np.array(condition*batch_size),training=False),training=False).numpy()
        sat = np.sum(np.logical_and(ys[:,0:1]<condition[0][1],ys[:,0:1]>condition[0][0]))/ys.shape[0]
        satisfaction_02.append(sat)
        sat = np.sum(np.logical_and(Y<condition[0][1],Y>condition[0][0]))/Y.shape[0]
        data_sat_02.append(sat)
        entropy_02.append(quad_entrop(ys))
        
    plt.rcParams.update({'font.size': 40})
    fig = plt.figure(figsize=(14,10))
    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.plot(np.linspace(0.1/2,(1-0.1/2),n),satisfaction_01, color='#004c6d',linewidth=5)
    ax.plot(np.linspace(0.2/2,(1-0.2/2),n),satisfaction_02, color='#9dc6e0',linewidth=5)
    ax.plot(np.linspace(0.1/2,(1-0.1/2),n),data_sat_01, color='#000',linewidth=5)
    ax.plot(np.linspace(0.2/2,(1-0.1/2),n),data_sat_02, color='grey',linewidth=5)
    ax.legend(['Range Size = 0.1, RANGE-GAN','Range Size = 0.2, RANGE-GAN','Range Size = 0.1, Data','Range Size = 0.2, Data'],loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=False, shadow=False, framealpha=0.0, ncol=2)
    ax.set_title(title)
    ax.set_xlabel('Center of Input Condition Range')
    ax.set_ylabel('Condition Satisfaction')
    
    plt.savefig(save_location+'.pdf',dpi=300, bbox_inches='tight')
    
    plt.rcParams.update({'font.size': 40})
    fig = plt.figure(figsize=(14,10))
    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.plot(np.linspace(0.1/2,(1-0.1/2),n),entropy_01, color='#004c6d',linewidth=5)
    ax.set_title('Entropy At Samples in Range-Size 0.1')
    ax.set_xlabel('Center of Input Condition Range')
    ax.set_ylabel('Quadratic Entropy')
    
    plt.savefig(save_location+'_entropy_01.pdf',dpi=300, bbox_inches='tight')
    
    plt.rcParams.update({'font.size': 40})
    fig = plt.figure(figsize=(14,10))
    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.plot(np.linspace(0.1/2,(1-0.1/2),n),entropy_02, color='#004c6d',linewidth=5)
    ax.set_title('Entropy At Samples in Range-Size 0.2')
    ax.set_xlabel('Center of Input Condition Range')
    ax.set_ylabel('Quadratic Entropy')
    
    plt.savefig(save_location+'_entropy_02.pdf',dpi=300, bbox_inches='tight')
    
def performance_plot_MO(model, Y, save_location):
    satisfaction= []
    data_sat = []
    range_size = 0.1
    n = 21
    for j in trange(n):
        for i in range(n):
            condition = [[i/(n-1)*(1-range_size),i/(n-1)*(1-range_size)+range_size,j/(n-1)*(1-range_size),j/(n-1)*(1-range_size)+range_size]]
            ys = model.regressor(model.generator(tf.random.normal(shape=[2000,64]),np.array(condition*2000),training=False),training=False).numpy()
            sat = np.sum(np.logical_and(np.logical_and(ys[:,0:1]<condition[0][1],ys[:,0:1]>condition[0][0]),np.logical_and(ys[:,1:2]<condition[0][3],ys[:,1:2]>condition[0][2])))/ys.shape[0]
            satisfaction.append(sat)
            sat = np.sum(np.logical_and(np.logical_and(Y[:,0:1]<condition[0][1],Y[:,0:1]>condition[0][0]),np.logical_and(Y[:,1:2]<condition[0][3],Y[:,1:2]>condition[0][2])))/Y.shape[0]
            data_sat.append(sat)
            
    fig = plt.figure(figsize=(14,10))
    plt.rcParams.update({'font.size': 26})
    plt.imshow(np.array(satisfaction).reshape(n,n),cmap='ocean',origin='lower')
    plt.colorbar(label='Condition Satisfaction')
    ax = plt.gca()
    # plt.axis('off')
    # Major ticks
    ax.set_xticks(np.linspace(0, 20, 5))
    ax.set_yticks(np.linspace(0, 20, 5))

    # Labels for major ticks
    ax.set_xticklabels(np.round(np.linspace(0.05, 0.95, 5),2))
    ax.set_yticklabels(np.round(np.linspace(0.05, 0.95, 5),2))

    # Minor ticks
    ax.set_xticks(np.arange(-.5, 21, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 21, 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('Aspect Ratio')
    plt.ylabel('Volume Ratio')

    plt.title('Multi-Constriant Condition Satisfaction')
    plt.savefig(save_location+'.pdf',dpi=300, bbox_inches='tight')
    
def real_performance_plot_SO(model, Y, save_location, IMAE, param, min_y, max_y, scale, res):
    satisfaction = []
    data_sat = []
    range_size = 0.1
    n_ranges = 21
    batch_size = 50
    for i in range(n_ranges):
        condition = [[i/(n_ranges-1)*(1-range_size),i/(n_ranges-1)*(1-range_size)+range_size]]
        zs = model.generator(tf.random.normal(shape=[batch_size,64]),np.array(condition*batch_size),training=False).numpy()
        vals,ratios,volumes = IMAE.synthesize(zs,res,200000)
        if param == 'volume':
            ys = np.array(volumes)
        else:
            ys = np.array(ratios)

        ys = (ys - min_y)/(max_y-min_y)/scale
            
        sat = np.sum(np.logical_and(ys<condition[0][1],ys>condition[0][0]))/ys.shape[0]
        satisfaction.append(sat)

        sat = np.sum(np.logical_and(Y<condition[0][1],Y>condition[0][0]))/Y.shape[0]
        data_sat.append(sat)

    fig = plt.figure(figsize=(14,10))
    plt.rcParams.update({'font.size': 26})
    plt.figure(figsize=(14,10))
    plt.plot(np.linspace(0.0,(1-range_size),n_ranges),satisfaction,color='#004c6d',linewidth=5)
    plt.plot(np.linspace(0.0,(1-range_size),n_ranges),data_sat,color='grey',linewidth=5)
    plt.legend(['Range-GAN','Data'])
    plt.title('Real World Performancey')
    plt.savefig(save_location+'_real.pdf',dpi=300, bbox_inches='tight')