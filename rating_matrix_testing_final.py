# -*- coding: utf-8 -*-
"""
TESTING 'SURPRISE' and 'FANCYIMPUTE' packages with artificial random data

Very rough first testing that should give some idea of performance

Also saves data in text format and Matlab

Created on Sat Oct  7 18:09:22 2017
@author: Jannek

"""

from surprise import SVD,Reader,BaselineOnly,Dataset
from surprise.model_selection import GridSearchCV,KFold
import numpy as np
#import matplotlib.pyplot as plt
import os
import random
#from fancyimpute import NuclearNormMinimization, IterativeSVD, SoftImpute
#from fancyimpute import SoftImpute

if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    SEED = 666

    for dimension in ['sentiment','infovalue','reliability','subjectivity','textlogic','writestyle']:

        print('\n----- Processing dataset %s -----' % dimension)
        # set RNG
        np.random.seed(SEED)
        random.seed(SEED)
        kf = KFold(n_splits=20,random_state=SEED,shuffle=True)  # folds will be the same for all algorithms.

        # path to dataset file
        file_path = os.path.expanduser(r'text'+dimension+'.txt')
        reader = Reader(line_format='user item rating', sep='\t', rating_scale=(1, 8))
        data = Dataset.load_from_file(file_path, reader=reader)

        #%% BASELINE
        param_grid = {}
        grid_search = GridSearchCV(BaselineOnly, param_grid, measures=['rmse', 'mae'], cv=kf)
        grid_search.fit(data)
        print('Baseline')
        print('Baseline best score and params for MAE: '+str(grid_search.best_score['mae'])+'  '+str(grid_search.best_params['mae']))
        print('Baseline best score and params for RMSE: '+str(grid_search.best_score['rmse'])+'  '+str(grid_search.best_params['rmse']))
        #algo = grid_search.best_estimator['mae']
        #algo.fit(data.build_full_trainset())


        #%% SVD
        for iter in range(1,4):
            param_grid = {'n_factors': [3,6, 8, 10, 12, 14, 16,20], 'n_epochs': [80],
                          'lr_all': [0.001,0.003,0.007], 'reg_all': [0.0002,0.001, 0.002, 0.004]}
            grid_search = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=kf)
            grid_search.fit(data)
            print('SVD iter %i' % iter)
            print('...SVD best score and params for MAE: '+str(grid_search.best_score['mae'])+'  '+str(grid_search.best_params['mae']))
            print('...SVD best score and params for RMSE: '+str(grid_search.best_score['rmse'])+'  '+str(grid_search.best_params['rmse']))
            #algo = grid_search.best_estimator['mae']
            #algo.fit(data.build_full_trainset())


    # #%% SVDpp
    # param_grid = {'n_factors':[2,4,6,8,10,12,14,16,20],'n_epochs': [70], 'lr_all': [0.001,0.003,0.007,0.014],'reg_all': [0.001,0.002,0.005]}
    # grid_search = GridSearchCV(SVDpp, param_grid, measures=['rmse', 'mae'], cv=kf)
    # grid_search.fit(data)
    # print('SVDpp best score and params for MAE: '+str(grid_search.best_score['mae'])+'  '+str(grid_search.best_params['mae']))
    # print('SVDpp best score and params for RMSE: '+str(grid_search.best_score['rmse'])+'  '+str(grid_search.best_params['rmse']))
    # algo = grid_search.best_estimator['mae']
    # algo.fit(data.build_full_trainset())

    # # %% NMF
    # param_grid = {'n_factors': [5, 10, 15, 20, 40], 'biased': [True], 'n_epochs': [40],
    #               'reg_pu': [0.03,0.06],'reg_qi': [0.03,0.06],'reg_bu': [0.01,0.02],'reg_bi': [0.01,0.02],'lr_bu': [0.002,0.005],'lr_bi': [0.002,0.005]}
    # grid_search = GridSearchCV(NMF, param_grid, measures=['rmse', 'mae'], cv=kf)
    # grid_search.fit(data)
    # print('NMF best score and params for MAE: ' + str(grid_search.best_score['mae']) + '  ' + str(
    #     grid_search.best_params['mae']))
    # print('NMF best score and params for RMSE: ' + str(grid_search.best_score['rmse']) + '  ' + str(
    #     grid_search.best_params['rmse']))
    # algo = grid_search.best_estimator['mae']
    # algo.fit(data.build_full_trainset())






#%% 

