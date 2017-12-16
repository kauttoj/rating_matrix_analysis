# -*- coding: utf-8 -*-
"""
TESTING 'SURPRISE' and 'FANCYIMPUTE' packages with artificial random data

Very rough first testing that should give some idea of performance

Also saves data in text format and Matlab

Created on Sat Oct  7 18:09:22 2017
@author: Jannek

"""

from surprise import SVD,SVDpp, Reader, NMF,BaselineOnly
from surprise import Dataset, GridSearch
from surprise import evaluate, print_perf
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from fancyimpute import NuclearNormMinimization, IterativeSVD, SoftImpute
#from fancyimpute import SoftImpute

def get_baseline(datamat,lr=0.01,n_epochs=50,reg=0):
    
    global_mean = np.nanmean(datamat)
        
    bu = global_mean*np.ones(datamat.shape[1])
    bi = global_mean*np.ones(datamat.shape[0])
    
    ind = np.where(np.isnan(datamat)==False)
    
    dataset = []
    for r,c in zip(*ind):
        dataset.append((r,c,datamat[r,c]))
    
    for dummy in range(n_epochs):
        for i, u, r in dataset:
            err = (r - (global_mean + bu[u] + bi[i]))
            bu[u] += lr * (err - reg * bu[u])
            bi[i] += lr * (err - reg * bi[i])

    return bu, bi,global_mean


# path to dataset file
file_path_incomplete = os.path.expanduser('D:\GoogleDrive\mydata_incomplete.txt')
file_path_complete = os.path.expanduser('D:\GoogleDrive\mydata_complete.txt')

N_sub=150   # number of raters
N_movie=200 # number of items (movies, books, songs, etc)
density=20/200 # density of rating matrix per subject
limits = [1,8] # rating limits

#%%
movs = np.arange(0,N_movie)


# here we define how close individual are to the ideal/real rating vector
ran = limits[1]-limits[0]
model_likeness = np.linspace(0,1,N_sub)**0.70
model_likeness[(N_sub-round(N_sub/4)):] = 0.80
model_likeness[0:round(N_sub/4)] = 0.05
model_likeness = np.flip(model_likeness,axis=0)

rating_model = ran*np.random.rand(N_movie)+limits[0]

items = []
ratings = np.array([])
users = []

datamat_full = np.zeros((N_movie,N_sub),dtype=np.float)
datamat_full.fill(np.nan)
datamat_missing = datamat_full.copy()

with open(file_path_incomplete,'w') as file_ic, open(file_path_complete,'w') as file_c:
    for i in range(0,N_sub):     
        item = np.arange(1,N_movie+1)
        
        rating = np.round((1.0-model_likeness[i])*(ran*np.random.rand(N_movie)+limits[0]) + model_likeness[i]*rating_model)    
    
        datamat_full[:,i]=rating
        
        for j in range(0,N_movie):
            file_c.write('%i\t%i\t%i\n' % (i+1,item[j],rating[j]))         
        
        # how many ratings are available, add some small 0-5 variation here
        n = np.minimum(np.int(np.ceil(density*N_movie +5*np.random.rand())),N_movie)
        
        sel = np.sort(np.random.choice(movs,size=n,replace=False))      
    
        user = np.repeat(np.array([i+1]),len(sel),axis=0)
        item = item[sel]
        rating = rating[sel]
        
        ratings=np.append(ratings,rating)
        
        for k in range(0,len(item)):
            items=np.append(items,'i%i' % item[k])            
            users=np.append(users,'u%i' % user[k])
        
        datamat_missing[item-1,user-1] = rating
        
        for j in range(0,len(sel)):
            file_ic.write('%i\t%i\t%i\n' % (user[j],item[j],rating[j]))
        
assert(np.max(datamat_full)<=limits[1] and np.min(datamat_full)>=limits[0])    

ratings_dict = {'item': items,'user': users,'rating': ratings}
df = pd.DataFrame(ratings_dict)
reader = Reader(rating_scale=(limits[0],limits[1]))    
data = Dataset.load_from_df(df[['user','item','rating']], reader)

df = pd.DataFrame(ratings_dict)
#reader = Reader(line_format='user item rating', sep='\t')

# A reader is still needed but only the rating_scale param is requiered.

data.split(n_folds=20)  # data can now be used normally

data_full = data.build_full_trainset()

bu, bi, global_mean = get_baseline(datamat_missing,lr=0.01,n_epochs=50,reg=0)
best_item_est_oma = np.mean(bu) + bi + global_mean

algo_baseline = BaselineOnly(reg_u=0,reg_i=0)
algo_baseline.train(data_full)
best_item_est = algo_baseline.trainset._global_mean + np.mean(algo_baseline.bu) + algo_baseline.bi

algo_SVD = SVD(verbose=True,n_factors = 5,n_epochs=50,reg_bu=0,reg_bi=0,reg_pu=0.1,reg_qi=0.1, biased=True)
algo_SVD.train(data_full)
best_item_est_svd = algo_SVD.trainset._global_mean + np.mean(algo_SVD.bu) + algo_SVD.bi

f, axarr = plt.subplots(nrows=2)
im1 = axarr[0].imshow(datamat_missing)
plt.colorbar(im1,ax=axarr[0])

mean_rating = np.nanmean(datamat_missing,axis=1)

axarr[1].plot(mean_rating,marker='s')
axarr[1].plot(best_item_est_oma,marker='o')
axarr[1].plot(best_item_est,marker='<')
axarr[1].plot(best_item_est_svd,marker='>')
f.show()

raise('ddd')

#%% SVD
param_grid_SVD = {'n_factors':[5,10,15,20,40,80],'n_epochs': [35], 'lr_all': [0.007,0.005,0.003],'reg_all': [0.005,0.01,0.02,0.05]}

grid_search = GridSearch(SVD, param_grid_SVD, measures=['MAE','RMSE'])

grid_search.evaluate(data)

# best MAE
print('best params (MAE): ' + str(grid_search.best_params['MAE']))

# combination of parameters that gave the best FCP score
print('best params (RMSE): ' + str(grid_search.best_params['RMSE']))

params = grid_search.best_params['MAE']
algo_SVD = SVD(
    verbose =True,
    n_factors = params['n_factors'],
    n_epochs  = params['n_epochs'],
    lr_all  = params['lr_all'],
    reg_all  = params['reg_all'])                    
algo_SVD.train(data_full)

#%% 

