# -*- coding: utf-8 -*-
"""
TESTING 'SURPRISE' and 'FANCYIMPUTE' packages with artificial random data

Very rough first testing that should give some idea of performance

Also saves data in text format and Matlab

Created on Sat Oct  7 18:09:22 2017
@author: Jannek

"""

from surprise import SVD,SVDpp, Reader, NMF
from surprise import Dataset, GridSearch
from surprise import evaluate, print_perf
import numpy as np
import pandas as pd
import os
from fancyimpute import NuclearNormMinimization, IterativeSVD, SoftImpute
#from fancyimpute import SoftImpute



# path to dataset file
file_path_incomplete = os.path.expanduser('D:\GoogleDrive\mydata_incomplete.txt')
file_path_complete = os.path.expanduser('D:\GoogleDrive\mydata_complete.txt')

N_sub=330   # number of raters
N_movie=250 # number of items (movies, books, songs, etc)
density=0.08 # density of rating matrix per subject
limits = [1,8] # rating limits

#%%
movs = np.arange(0,N_movie)

# here we define how close individual are to the ideal/real rating vector
ran = limits[1]-limits[0]
model_likeness = np.linspace(0,1,N_sub)**0.70
model_likeness[(N_sub-round(N_sub/4)):] = 0.90
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
 
# matrix completion using convex optimization to find low-rank solution
# that still matches observed values. Slow!
#datamat_filled_nnm = NuclearNormMinimization().complete(datamat_missing)

# Instead of solving the nuclear norm objective directly, instead
# induce sparsity using singular value thresholding
#datamat_filled_soft = SoftImpute().complete(datamat_missing)
    
ratings_dict = {'item': items,'user': users,'rating': ratings}
df = pd.DataFrame(ratings_dict)
reader = Reader(rating_scale=(limits[0],limits[1]))    
data = Dataset.load_from_df(df[['user','item','rating']], reader)

df = pd.DataFrame(ratings_dict)
#reader = Reader(line_format='user item rating', sep='\t')

# A reader is still needed but only the rating_scale param is requiered.

data.split(n_folds=10)  # data can now be used normally

data_full = data.build_full_trainset()

#
obj = IterativeSVD(
            rank = 20,
            max_iters=700,
            min_value=limits[0],
            max_value=limits[1],
            verbose=True)

datamat_filled_SVD_fancy = obj.complete(datamat_missing)

obj = SoftImpute(
            shrinkage_value=None,
            max_iters=700,
            max_rank=20,
            n_power_iterations=1,
            init_fill_method="zero",
            min_value=limits[0],
            max_value=limits[1],
            normalizer=None,
            verbose=True)

datamat_filled_SOFT_fancy = obj.complete(datamat_missing)

obj = NuclearNormMinimization(require_symmetric_solution=False,
            min_value=limits[0],
            max_value=limits[1],
            error_tolerance=0.0001,
            fast_but_approximate=True,
            verbose=True)

datamat_filled_NNM_fancy = obj.complete(datamat_missing)

#%% NMF
param_grid_NMF = {'n_factors':[5,10,20],'n_epochs': [70],'biased':[False,True]}

#grid_search = GridSearch(SVDpp, param_grid, measures=['RMSE', 'MAE'])
grid_search = GridSearch(NMF, param_grid_NMF, measures=['RMSE'])

# Evaluate performances of our algorithm on the dataset.
grid_search.evaluate(data)

print('best: ' + str(grid_search.best_score['RMSE']))

# combination of parameters that gave the best FCP score
print('best params: ' + str(grid_search.best_params['RMSE']))

params = grid_search.best_params['RMSE']
algo_NMF = NMF(
    verbose =True,
    n_factors = params['n_factors'],
    n_epochs  = params['n_epochs'],
    biased  = params['biased'])
algo_NMF.train(data_full)

#%% SVD
param_grid_SVD = {'n_factors':[5,10,20],'n_epochs': [70], 'lr_all': [0.005,0.003,0.001],'reg_all': [0.005,0.01,0.02]}

#grid_search = GridSearch(SVDpp, param_grid, measures=['RMSE', 'MAE'])
grid_search = GridSearch(SVD, param_grid_SVD, measures=['RMSE'])

# Evaluate performances of our algorithm on the dataset.
grid_search.evaluate(data)

# best MAE
print('best: ' + str(grid_search.best_score['RMSE']))

# combination of parameters that gave the best FCP score
print('best params: ' + str(grid_search.best_params['RMSE']))

params = grid_search.best_params['RMSE']
algo_SVD = SVD(
    verbose =True,
    n_factors = params['n_factors'],
    n_epochs  = params['n_epochs'],
    lr_all  = params['lr_all'],
    reg_all  = params['reg_all'])                    
algo_SVD.train(data_full)

#%% 

datamat_filled_SVD = datamat_missing.copy().astype(np.float)
datamat_filled_NMF = datamat_missing.copy().astype(np.float)
for i in range(0,datamat_full.shape[0]):  # movie
    for j in range(0,datamat_full.shape[1]):   # user

        val = algo_SVD.predict('u%i'%(j+1),'i%i'%(i+1)).est
        datamat_filled_SVD[i,j]=val  
        
        val = algo_NMF.predict('u%i'%(j+1),'i%i'%(i+1)).est
        datamat_filled_NMF[i,j]=val        
   
#%% compute correlations between real and recovered ratings
corvals_SVD = np.zeros(datamat_full.shape[1])    
corvals_NMF = np.zeros(datamat_full.shape[1])  
corvals_SVD_fancy = np.zeros(datamat_full.shape[1])    
corvals_NNM_fancy = np.zeros(datamat_full.shape[1])    
corvals_SOFT_fancy = np.zeros(datamat_full.shape[1]) 
for j in range(0,datamat_full.shape[1]):   # user
    corvals_SVD[j] = np.corrcoef(datamat_full[:,j],datamat_filled_SVD[:,j])[0,1]
    corvals_SVD_fancy[j] = np.corrcoef(datamat_full[:,j],datamat_filled_SVD_fancy[:,j])[0,1]
    corvals_NMF[j] = np.corrcoef(datamat_full[:,j],datamat_filled_NMF[:,j])[0,1]
    corvals_NNM_fancy[j] = np.corrcoef(datamat_full[:,j],datamat_filled_NNM_fancy[:,j])[0,1]
    corvals_SOFT_fancy[j] = np.corrcoef(datamat_full[:,j],datamat_filled_SOFT_fancy[:,j])[0,1]

print('\n\n')
print('mean corvals SVD: %f\n' % np.mean(corvals_SVD))
print('mean corvals SVD fancy: %f\n' % np.mean(corvals_SVD_fancy))
print('mean corvals NNM fancy: %f\n' % np.mean(corvals_NNM_fancy))
print('mean corvals SOFT fancy: %f\n' % np.mean(corvals_SOFT_fancy))
print('mean corvals NMF: %f\n' % np.mean(corvals_NMF))

#%% Save results and make some plots
    
from scipy.io import savemat

savemat('surprise_results.mat',
        {'datamat_missing':datamat_missing,
         'datamat_full':datamat_full,
         'datamat_filled_SVD':datamat_filled_SVD,
         'datamat_filled_NMF':datamat_filled_NMF,
         'datamat_filled_SVD_fancy':datamat_filled_SVD_fancy,
         'datamat_filled_NNM_fancy':datamat_filled_NNM_fancy,
         'datamat_filled_SOFT_fancy':datamat_filled_SOFT_fancy,      
         'rating_model':rating_model,'model_likeness':model_likeness},
         appendmat=False)

# simple wrapper to make some plots
import jannes_utils as jutil

jutil.plot(model_likeness,'MODEL')
jutil.plot(corvals_SVD,'SVD')
jutil.plot(corvals_SVD_fancy,'SVD fancy')
jutil.plot(corvals_NMF,'NMF')
jutil.plot(corvals_NNM_fancy,'NNM fancy')
jutil.plot(corvals_SOFT_fancy,'SOFT fancy')

jutil.plot_correlation_matrix(datamat_full.T,'MODEL')
jutil.plot_correlation_matrix(datamat_filled_NMF.T,'NMF')
jutil.plot_correlation_matrix(datamat_filled_NNM_fancy.T,'NNM fancy')
jutil.plot_correlation_matrix(datamat_filled_SVD_fancy.T,'SVD fancy')
jutil.plot_correlation_matrix(datamat_filled_SVD.T,'SVD')
jutil.plot_correlation_matrix(datamat_filled_SOFT_fancy.T,'SOFT fancy')
#perf = evaluate(SVD(), data, measures=['RMSE', 'MAE'])
#print_perf(perf)