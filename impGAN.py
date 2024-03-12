#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
impGAN:
    Functions to define, train and predict missing data using GAN
    
Main functions:
    - impGAN() : define and train GAN
    - predictGAN() : use trained GAN to predict missing data

@author: ahmedkhan
April 9, 2022
"""

import os
import numpy as np
from tqdm import tqdm
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#%% GAN initialization and training

def impGAN(orig_data, net_param, restore_flag, save_dir, net_name, is_training=True):
    '''
    Initialize and train GAN to impute missing values in data matrix

    Parameters
    ----------
    data_x : array
        Data matrix with missing values (as NaN)
    net_param : dict
        Parameters for network 
            - "batch_size"
            - "hint_rate"
            - "alpha"
            - "iterations"
    restore_flag : bool
        If true, continue training from previous session
    save_dir : string
        Path to save network
    net_name : string
        Name for saved network
    is_training : bool, optional
        Set to false to predict missing values

    Returns
    -------
    imputed_data : array
        Imputed data
    '''
    
    # Mask matrix
    data_missing = 1-np.isnan(orig_data)
    
    # Training parameters
    batch_size = net_param['batch_size']
    hint_rate = net_param['hint_rate']
    alpha = net_param['alpha']
    iterations = net_param['iterations']
    
    N_data, dim = orig_data.shape
    
    # Hidden state dimensions
    N_hidden = int(dim)
    
    # Normalization
    norm_data, norm_param = normalization(orig_data)
    norm_data_x = np.nan_to_num(norm_data, 0)
    
    # Define network architecture
    # Placeholders for input
    X = tf.placeholder(tf.float32, shape = [None, dim]) # Data
    M = tf.placeholder(tf.float32, shape = [None, dim]) # Mask
    H = tf.placeholder(tf.float32, shape = [None, dim]) # Hint
    
    # Discriminator 
    D_W1 = tf.Variable(xavier_init([dim*2, N_hidden])) # Data + Hint as inputs
    D_b1 = tf.Variable(tf.zeros(shape = [N_hidden]))
    
    D_W2 = tf.Variable(xavier_init([N_hidden, N_hidden]))
    D_b2 = tf.Variable(tf.zeros(shape = [N_hidden]))
    
    D_W3 = tf.Variable(xavier_init([N_hidden, dim]))
    D_b3 = tf.Variable(tf.zeros(shape = [dim]))  # Multi-variate outputs
    
    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
  
    # Generator: data (with noise in missing values) and mask are inputs
    G_W1 = tf.Variable(xavier_init([dim*2, N_hidden]))  
    G_b1 = tf.Variable(tf.zeros(shape = [N_hidden]))
    
    G_W2 = tf.Variable(xavier_init([N_hidden, N_hidden]))
    G_b2 = tf.Variable(tf.zeros(shape = [N_hidden]))
    
    G_W3 = tf.Variable(xavier_init([N_hidden, dim]))
    G_b3 = tf.Variable(tf.zeros(shape = [dim]))
    
    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]
  
    # Generator
    def generator(x,m):
        # Concatenate data and mask matrices
        inputs = tf.concat(values = [x, m], axis = 1) 
        G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)   
        G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3) 
        return G_prob
      
    # Discriminator
    def discriminator(x, h):
        # Concatenate data and hint matrices
        inputs = tf.concat(values = [x, h], axis = 1) 
        D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)  
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        D_logit = tf.matmul(D_h2, D_W3) + D_b3
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob
    
    # Define network
    G_sample = generator(X, M) # Generate data
    Hat_X = X * M + G_sample * (1-M) # Combine with observed data
    D_prob = discriminator(Hat_X, H) # Discriminate
    
    # Loss function
    D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) \
                                + (1-M) * tf.log(1. - D_prob + 1e-8)) 
  
    G_loss_temp = -tf.reduce_mean((1-M) * tf.log(D_prob + 1e-8))
  
    MSE_loss = \
        tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)
  
    D_loss = D_loss_temp
    G_loss = G_loss_temp + alpha * MSE_loss 
  
    # Optimizer
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
  
    # Initialize
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
  
    # Save network
    if restore_flag:
        tf.reset_default_graph()
        saver = tf.train.import_meta_graph(os.path.join(save_dir, net_name + '.impgan' + '.meta'))
        saver.restore(sess, tf.train.latest_checkpoint(save_dir))
    else: 
        saver = tf.train.Saver()
  
    if is_training:
        # Start Iterations
        for it in tqdm(range(iterations)):    
        
            # Sample batch
            batch_idx = sample_batch_inds(N_data, batch_size)
            X_mb = norm_data_x[batch_idx, :]  
            M_mb = data_missing[batch_idx, :]  
            
            # Sample random vectors  
            Z_mb = get_uniform(0, 0.01, batch_size, dim) 
      
            # Sample hint vectors
            H_mb_temp = get_bernoulli(hint_rate, batch_size, dim)
            H_mb = M_mb * H_mb_temp
            
            # Combine random vectors with observed vectors
            X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
        
            _, D_loss_curr = sess.run([D_solver, D_loss_temp], 
                                feed_dict = {M: M_mb, X: X_mb, H: H_mb})
            _, G_loss_curr, MSE_loss_curr = \
                sess.run([G_solver, G_loss_temp, MSE_loss],
                feed_dict = {X: X_mb, M: M_mb, H: H_mb})
            
        # Save network parameters
        saver.save(sess, os.path.join(save_dir, net_name + '.impgan'))
  
    # Output imputed data      
    Z_mb = get_uniform(0, 0.01, N_data, dim) 
    M_mb = data_missing
    X_mb = norm_data_x          
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
      
    norm_imputed_data = sess.run([G_sample], feed_dict={X: X_mb, M: M_mb})[0]
  
    imputed_data = data_missing * norm_data_x + (1-data_missing) * norm_imputed_data
  
    # Un-normalize before output
    imputed_data = unnormalization(imputed_data, norm_param)  
   
    if is_training:
        return imputed_data, norm_imputed_data
    else:
        return imputed_data, norm_imputed_data
    
#%% GAN prediction

def predictGAN(orig_data, path, name):
    '''
    Use previously trained GAN for imputation

    Parameters
    ----------
    orig_data : array
        Data matrix with missing values (as NaN)
    path : string
        Path to saved networks
    name : string
        Saved network name

    Returns
    -------
    imputed_data : array
        Imputed data matrix

    '''
    
    # Mask matrix
    data_missing = 1-np.isnan(orig_data)
    N_data, dim = orig_data.shape
    
    # Load network weights
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(os.path.join(path, name + '.impgan'))
    saver.restore(sess, tf.train.latest_checkpoint(path))
    
    # Normalize data before passing it to the GAN
    norm_data, norm_param = normalization(orig_data)
    norm_data_x = np.nan_to_num(norm_data, 0)
    
    # Return imputed data      
    Z_mb = get_uniform(0, 0.01, N_data, dim) 
    M_mb = data_missing
    X_mb = norm_data_x          
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
          
    imputed_data = sess.run([G_sample], feed_dict={X: X_mb, M: M_mb})[0]
    
    # Mask imputed missing data onto original data
    imputed_data = data_missing * norm_data_x + (1-data_missing) * imputed_data
    
    # Un-normalize before output
    imputed_data = unnormalization(imputed_data, norm_param)  
    
    return imputed_data

#%% Helper functions

def normalization(data, param=None):
    '''
    Normalize data to [0,1]

    Parameters
    ----------
    data : array
        Data to normalize
    param : dict, optional
        Range for normalization

    Returns
    -------
    norm_data : array
        Normalized data.
    norm_param : dict
        Minimum and maximum values to later un-normalize data

    '''

    _, dim = data.shape
    norm_data = data.copy()
  
    if param is None:
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)
    
        for i in range(dim):
            min_val[i] = np.nanmin(norm_data[:,i])
            norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
            max_val[i] = np.nanmax(norm_data[:,i])
            norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) + 1e-6)   
      
        # Return norm_parameters for later un-normalization
        norm_param = {'min_val': min_val,
                       'max_val': max_val}

    else:
        min_val = param['min_val']
        max_val = param['max_val']
    
        for i in range(dim):
            norm_data[:,i] = norm_data[:,i] - min_val[i]
            norm_data[:,i] = norm_data[:,i] / (max_val[i] + 1e-6)  
      
        norm_param = param    
      
    return norm_data, norm_param

def unnormalization(norm_data, norm_param):
    '''
    Un-normalize data from [0,1] to original range

    Parameters
    ----------
    norm_data : array
        Normalized data
    norm_param : dict
        Contains minimum and maximum values from original data range

    Returns
    -------
    unnorm_data : array
        Un-normalized data

    '''
  
    min_val = norm_param['min_val']
    max_val = norm_param['max_val']

    _, dim = norm_data.shape
    unnorm_data = norm_data.copy()
    
    for i in range(dim):
        unnorm_data[:,i] = unnorm_data[:,i] * (max_val[i] + 1e-6)   
        unnorm_data[:,i] = unnorm_data[:,i] + min_val[i]
    
    return unnorm_data

def r2_calc(orig_data, imputed_data, data_missing):
    '''
    Calculate R^2 for imputed data

    Parameters
    ----------
    orig_data : array
        Original data with missing values
    imputed_data : array
        Imputed data 
    data_missing : array
        Mask array for values missing in original data

    Returns
    -------
    r2 : float
        R^2 for imputed data

    '''
    # Normalize using parameters fit to original data
    orig_data, norm_parameters = normalization(orig_data)
    imputed_data, _ = normalization(imputed_data, norm_parameters)
    N_data = orig_data.shape[0]
  
    # Calculate R^2 only for missing values
    ss_res = np.sum((((1-data_missing) * orig_data) - ((1-data_missing) * imputed_data)) ** 2, axis=0)
    means_orig = np.tile(np.mean((1-data_missing) * orig_data, axis=0), [N_data,1])
    ss_total = np.sum((means_orig - ((1-data_missing) * imputed_data)) ** 2, axis=0)
  
    r2 = 1 - (ss_res/ss_total)
  
    return r2

def rmse_loss(orig_data, imputed_data, data_missing):
    '''
    Calculate rmse for imputed data (only missing values)

    Parameters
    ----------
    orig_data : array
        Original data with missing values
    imputed_data : array
        Imputed data 
    data_missing : array
        Mask array for values missing in original data
        
    Returns
    -------
    rmse : float
        Root mean squared error

    '''
    # Normalize using parameters fit to original data
    orig_data, norm_parameters = normalization(orig_data)
    imputed_data, _ = normalization(imputed_data, norm_parameters)
    
    # Calculate RMSE only for missing values
    temp1 = np.sum(((1-data_missing) * orig_data - (1-data_missing) * imputed_data)**2, axis=0)
    temp2 = np.sum(1-data_missing, axis=0)
  
    rmse = np.sqrt(temp1/temp2)
  
    return rmse

def xavier_init(dims):
    '''
    Xavier initialization
    Parameters
    ----------
    dims : int or int array
        Vector size
        
    Returns
    -------
    vec : [size] array
        Initialized random vector

    '''
    xavier_stddev = 1. / tf.sqrt(dims[0] / 2.)
    vec = tf.random_normal(shape=dims, stddev=xavier_stddev)

    return vec
      
def get_bernoulli(p, N_rows, N_cols):
    '''
    Sample from Bernoulli distribution by thresholding a uniform distribution

    Parameters
    ----------
    p : float
        Probability
    N_rows : int
        Number of rows
    N_cols : int
        Number of columns

    Returns
    -------
    bin_rand_mat : [N_rows x N_cols] array
        Binarized matrix

    '''
    rand_mat = np.random.uniform(0., 1., size=[N_rows, N_cols])
    bin_rand_mat = 1 * (rand_mat < p)
    
    return bin_rand_mat

def get_uniform(low, high, N_rows, N_cols):
    '''
    Sample from uniform random distribution

    Parameters
    ----------
    low : float
        Lower bound
    high : float
        Upper bound
    N_rows : int
        Number of sampled rows
    N_cols : int
        Number of sampled columns      

    Returns
    -------
    rand_mat : [N_rows x N_cols] array
        Sampled uniform random matrix
   '''
   
    rand_mat = np.random.uniform(low, high, size = [N_rows, N_cols])      
   
    return rand_mat

def sample_batch_inds(N_total, N_batch):
    '''
    Sample batches for training

    Parameters
    ----------
    N_total : integer
        Total number of data points
    batch_size : integer
        Size of training batch

    Returns
    -------
    ind_batch : 1D array
        Batch indices

    '''
  
    ind_perm = np.random.permutation(N_total)
    ind_batch = ind_perm[:N_batch]
    
    return ind_batch