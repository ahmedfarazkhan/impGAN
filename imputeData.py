# -*- coding: utf-8 -*-
"""
imputeData:
    Script to load data, train network and predict missing values
    
Parameters to change for different data set:
    - miss_rate : missing data rate (tune to data matrix)
    - mods_to_imputed : imputation modalities/features 

@author: ahmedkhan
April 9, 2022
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import numpy as np
import random
import scipy.io
import datetime
import matplotlib.pyplot as plt

from impGAN import impGAN, predictGAN, rmse_loss, r2_calc

from sklearn.model_selection import train_test_split, StratifiedKFold

# Set random seeds
random.seed(0)
np.random.seed(0)

#%% Global variables

save_dir = "output"
net_name = "saved_net"
fname_input = "data/data2impute_06-Jan-2022"

#%%
def main(args):
    '''
    Train GAN for imputation. 
    
    Split (complete) data matrix into training and validation subsets, and
    randomly exclude data from training subset based on missing rate.
    
    Parameters
    ----------
    args: 
        data_training : array
            Complete data matrix for training
        mods_to_impute : array
            Indices of features in data matrix to impute
        miss_rate : float
            Probability of missing data
        batch_size : int
            Batch size
        hint_rate : float
            Hint for discriminator 
        alpha : float
            Hyperparameter
        iterations : int
            Number of iterations
        N_folds : integer
            Number of CV folds
    
    Returns
    -------
    imputed_data : array
        Imputed data matrix
    rmse : float
        Root mean squared error of imputated data
        
    '''  
    
    # Parameters
    data2impute = args.data_training
    mods_to_impute = args.mods_to_impute
    miss_rate = args.miss_rate
    N_folds = args.N_folds
    
    net_param = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}
    
    N_samples, dim = data2impute.shape
    print('Imputing with ' + str(N_samples) + " data points")
  
    # Cross-validation performance metrics
    r2 = np.zeros((N_folds, N_modalities))
    rmse = np.zeros((N_folds, N_modalities))
  
    restore_flag = False
    
    k = 0
        
    skf = StratifiedKFold(n_splits=10)
    for k, (train_ind, valid_ind) in enumerate(skf.split(data2impute, np.ones(data2impute.shape[0]))):

        # Randomly exclude data from selected modalities according to a missing data rate
        # Exclude data from training subset
        N_train = train_ind.shape[0]
        exclusion_matrix_train = np.ones([N_train, N_modalities])
        inds_exclude_train = np.asarray(random.sample(range(N_train), int(N_train * miss_rate )))
        exclusion_matrix_train[inds_exclude_train[:,None], mods_to_impute] = 0 
        imputation_mask_train = exclusion_matrix_train.copy()
        
        missing_data_train = data2impute[train_ind,:].copy()
        missing_data_train[imputation_mask_train == 0] = np.nan
        
        # Exclude data from validation subset
        N_valid = valid_ind.shape[0]
        exclusion_matrix_valid = np.ones([N_valid, N_modalities])
        inds_exclude_valid = np.asarray(random.sample(range(N_valid), int(N_valid * miss_rate )))
        exclusion_matrix_valid[inds_exclude_valid[:,None], mods_to_impute] = 0 
        imputation_mask_valid = exclusion_matrix_valid.copy()
        
        missing_data_valid = data2impute[valid_ind,:].copy()
        missing_data_valid[imputation_mask_valid == 0] = np.nan
        
        # Train GAN
        print('Training fold k=%d, training with %d data points' % (k, missing_data_train.shape[0]))
        [imputed_data_train, _] = impGAN(missing_data_train, net_param, restore_flag, save_dir, net_name + "-CV" + str(k))
      
        # Predict validation subset
        [imputed_data_valid, _] = impGAN(missing_data_valid, net_param, True, save_dir, net_name + "-CV" + str(k), False)
        #impGAN(missing_data, net_param, restore_flag, save_dir, net_name)
        
        # Record the CV imputation performance
        rmse[k,:] = rmse_loss(data2impute[valid_ind,:], imputed_data_valid, imputation_mask_valid)
        r2[k,:] = r2_calc(data2impute[valid_ind,:], imputed_data_valid, imputation_mask_valid)
        
        print('r2 = %.3f,%.3f,%.3f' % (r2[k,3], r2[k,4], r2[k,5]))
  
    print('Mean R2:')
    for i in mods_to_impute:
        print('Mean R2 for modality %d: %.4f' % (i, np.mean(r2[:,i])))
        
    today = datetime.date.today() 
    fname = 'imputation_metrics-%s.mat' % (today.strftime('%y-%m-%d')) 
    scipy.io.savemat(os.path.join(save_dir,fname), {'r2s':r2, 'rmses':rmse, 'net_param':net_param})
    
    # Train on full training set and save network
    restore_flag = False
    for k, (train_ind, valid_ind) in enumerate(skf.split(data2impute, np.ones(data2impute.shape[0]))):

        # Randomly exclude data from selected modalities according to a missing data rate
        # Exclude data from training subset
        N_train = train_ind.shape[0]
        exclusion_matrix_train = np.ones([N_train, N_modalities])
        inds_exclude_train = np.asarray(random.sample(range(N_train), int(N_train * miss_rate )))
        exclusion_matrix_train[inds_exclude_train[:,None], mods_to_impute] = 0 
        imputation_mask_train = exclusion_matrix_train.copy()
        
        missing_data_train = data2impute[train_ind,:].copy()
        missing_data_train[imputation_mask_train == 0] = np.nan

        # Train GAN        
        [imputed_data_train, _] = impGAN(missing_data_train, net_param, restore_flag, save_dir, net_name)
        restore_flag = True # Continue updating this model

if __name__ == '__main__':  
  
    # Load data
    data = scipy.io.loadmat(fname_input)
    data_raw = data['data_2impute']
    
    # Specific to test data
    N_regs = 163
    N_modalities = int(data_raw.shape[-1] / N_regs)
      
    # Check missing rate by modality
    nan_count = np.isnan(data_raw)
    reg_nan_count = np.zeros([data_raw.shape[0] * N_regs, N_modalities])
    for i in range(data_raw.shape[0]):
        for j in range(N_modalities):
            reg_nan_count[N_regs*i:N_regs*(i+1),j] = nan_count[i,N_regs*j:N_regs*(j+1)]
    for i in range(N_modalities):
        print('Modality %d: %.3f %% missing' % (i+1, 100*np.count_nonzero(reg_nan_count[:,i])/reg_nan_count.shape[0]))
    
    # Count number of missing modalities per data point
    missing_mod_count = np.zeros(reg_nan_count.shape[0])
    for i in range(reg_nan_count.shape[0]):
        missing_mod_count[i] = np.count_nonzero(reg_nan_count[i,:])
    for i in range(N_modalities+1):
        curr_count = len(np.where(missing_mod_count == i)[0])
        print('Missing %d modalities: %.3f %% of data' % (i, 100*curr_count/reg_nan_count.shape[0]))
        
    plt.hist(missing_mod_count)
    plt.title('Missing modalities')
  
    # Exclude subjects with any NaN for training
    original_nan_mask = ~np.any(np.isnan(data_raw), axis=1)
    data_raw = data_raw[original_nan_mask]
  
    # For whole brain neuroimaging
    data_raw_regional = np.zeros([data_raw.shape[0]*N_regs, N_modalities])
    for i in range(data_raw.shape[0]):
        for j in range(N_modalities):
            data_raw_regional[N_regs*i:N_regs*(i+1),j] = data_raw[i,N_regs*j:N_regs*(j+1)]

    # Hyperparameter search
    batch_sizes = [64, 128, 256]
    hint_rates = [0.5,0.8, 0.9, 0.95]
    alphas = [50, 100, 500, 1000]
  
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--miss_rate',
        help='missing data probability',
        default=0.33, # ~33% missing data in PPMI
        type=float)
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch',
        default=128, # Default = 128
        type=int)
    parser.add_argument(
        '--hint_rate',
        help='hint probability',
        default=0.9,
        type=float)
    parser.add_argument(
        '--alpha',
        help='hyperparameter',
        default=200, 
        type=float)
    parser.add_argument(
        '--iterations',
        help='number of training interations',
        default=1000, 
        type=int)
    parser.add_argument(
        '--data_training',
        help='training data',
        default=data_raw_regional,
        type=int)
    parser.add_argument(
        '--N_folds',
        help='number of cross validation folds',
        default=10,
        type=int)
    parser.add_argument(
        '--mods_to_impute',
        help='modalities/features to impute',
        default=list(range(3,N_modalities)), 
        type=int)
  
    args = parser.parse_args() 
  
    # Calls main function  
    main(args) 
    
#%% Prediction
net_param = {'batch_size': 128,
             'hint_rate': 0.9,
             'alpha': 100,
             'iterations': 1000}

data = scipy.io.loadmat(fname_input)
data_all = data['data_2impute']
N_samples_total = data_all.shape[0] 

data_all_nan_mask = np.isnan(data_all)

# For whole brain neuroimaging data
data_raw_regional = np.zeros([data_all.shape[0]*N_regs, N_modalities])
for i in range(N_samples_total):
  for j in range(N_modalities):
    data_raw_regional[N_regs*i:N_regs*(i+1),j] = data_all[i,N_regs*j:N_regs*(j+1)]
    
# Mask out nan
data_regional_nan_mask = np.isnan(data_raw_regional)

# Input requires NaN for missing data
imputed_data_regional, normalized_imputed_data_regional = \
    impGAN(data_raw_regional, net_param, True, save_dir, net_name, False)

imputed_data = np.zeros([N_samples_total, N_modalities*N_regs])
for i in range(N_samples_total):
  for j in range(N_modalities):
    imputed_data[i,N_regs*j:N_regs*(j+1)] = imputed_data_regional[N_regs*i:N_regs*(i+1),j]
    
today = datetime.date.today()
fname = "impGAN_imputed_data.mat" % (today.strftime('%y-%m-%d'))
scipy.io.savemat(os.path.join(save_dir,fname), {"imputed_data":imputed_data})

# Don't output this, because the data needs to be normalized wrt. healthy controls
plt.imshow(imputed_data)
