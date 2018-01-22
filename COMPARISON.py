# This code compares Surprise methods with randomized hyperparameter optimization and k-fold
# after running this few times you should have a good idea of good parameters
# code will be run for multiple rating types
# -Janne K.

from surprise import SVD,Reader,BaselineOnly,Dataset,NMF,KNNWithMeans,SlopeOne,accuracy,KNNWithZScore
from surprise.model_selection import GridSearchCV,KFold,cross_validate
import surprise.accuracy
import numpy as np
#import matplotlib.pyplot as plt
import os
import random

if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    TEXT_PATH = r'D:/GoogleDrive/kysely/'  # data folder

    SEED = 666  # data partition seed
    ranks = [3,5,10,15,20,30]  # number of components in SVD and NMF, all will be tested
    N_FOLD = 15  # data folds to test
    N_cores = 4  # how manu parallel jobs
    N_samples = 80  # number of random grid search samples
    N_print_interval=20

    def print_output(s,end='\n'):
        print(s,end=end)
        print(s,end=end,file=outfile)

    def print_top(head,arr,param,n):
        k = np.argsort(arr)
        print_output(head)
        for i in range(n):
            print_output(' %i ... %f with params %s' % (i+1,arr[k[i]],str(param[k[i]])))

    # function to randomize parameters, same parameters are not repeated
    def random_search(params):
        count = params.copy()
        total = 1
        for key in params:
            count[key] = len(params[key])
            total *= len(params[key])
        if 2 * N_samples > total:  # make sure our search space is large (otherwise can stuck in loop for now)
            raise (Exception('Too many samples (%i) or parameters (%i)!' % (N_samples, total)))
        param_set = []
        used = set()
        while len(param_set) < N_samples:
            p = {}
            s = ''
            for key in params:
                k = int(np.random.choice(count[key], 1))
                p[key] = params[key][k]
                s += str(k)
            if s not in used:
                used.add(s)
                param_set.append(p)
        return param_set

    # run for each rating type ("dimension")
    for dimension in ['reliability','sentiment','infovalue','subjectivity','textlogic','writestyle']:

        # set RNG
        np.random.seed(SEED)
        random.seed(SEED)
        kf = KFold(n_splits=N_FOLD,random_state=SEED,shuffle=True)  # folds will be the same for all algorithms.

        # path to dataset file
        file_path = os.path.expanduser(TEXT_PATH + r'Laurea_limesurvey_experiment_16.01.2018_RESPONSES_TEXT_'+dimension+'.txt')
        reader = Reader(line_format='user item rating', sep='\t', rating_scale=(1, 8))
        data = Dataset.load_from_file(file_path, reader=reader)

        f = 'output_%s.txt' % dimension
        outfile = open(f,'w')

        print_output('\n----- Processing dataset \'%s\' -----\n' % dimension)

        #%% BASELINE
        grid_search = GridSearchCV(BaselineOnly, param_grid = {}, measures=['rmse', 'mae'], cv=kf)
        grid_search.fit(data)
        print_output('Baseline best score and params for MAE: '+str(grid_search.best_score['mae']))
        print_output('Baseline best score and params for RMSE: '+str(grid_search.best_score['rmse']))

        #%% SVD testing
        param_grid = {'lr_bu':  [0.02,0.005,0.002],
                      'lr_bi':  [0.02,0.005,0.002],
                      'lr_pu':  [0.02,0.005,0.002],
                      'lr_qi':  [0.02,0.005,0.002],
                      'reg_bu': [0.15,0.08,0.02,0.01],
                      'reg_pu': [0.8,0.4,0.15,0.05,0.02],
                      'reg_bi': [0.8,0.4,0.15,0.08,0.02,0.01],
                      'reg_qi': [0.8,0.4,0.15,0.08,0.02,0.01]
                      }
        params=random_search(param_grid)
        mae_all=[]
        rmse_all=[]
        print_output('\nRunning SVD with %i parameter sets' % (len(ranks) * len(params)))
        iter=0
        params_all=[]
        for rank in ranks:
            for p in params:
                iter+=1
                p['n_factors']=rank
                p['n_epochs']=30
                algo = SVD(**p)
                mae_arr=[]
                rmse_arr=[]
                res = cross_validate(algo, data, measures=['rmse', 'mae'], cv=kf,
                               return_train_measures=False, n_jobs=3,verbose=False)
                mae_all.append(np.mean(res['test_mae']))
                rmse_all.append(np.mean(res['test_rmse']))
                params_all.append(p)
                if (iter-1) % N_print_interval  == 0:
                    print_output('... set %i' % iter, end='')
                    print_output('  mae=%f  rmse=%f  params=%s' % (mae_all[-1], rmse_all[-1],str(params_all[-1])))
        print_top('SVD best score and params for MAE:',mae_all,params_all,3)
        print_top('SVD best score and params for RMSE:',rmse_all,params_all,3)

        # NMF testing
        param_grid = {'reg_pu': [0.1,0.3, 0.5, 0.8],
                      'reg_qi': [0.1,0.3, 0.5, 0.8],
                      'reg_bu': [0.5,0.3,0.1,0.02],
                      'reg_bi': [0.5,0.3,0.1,0.02],
                      'lr_bu' : [0.03,0.005,0.002],
                      'lr_bi' : [0.03,0.005,0.002]
                      }
        params = random_search(param_grid)
        mae_all = []
        rmse_all = []
        print_output('\nRunning NMF with %i parameter sets' % (len(ranks) * len(params)))
        iter=0
        params_all = []
        for rank in ranks:
            for p in params:
                iter+=1
                p['n_factors'] = rank
                p['n_epochs'] = 60
                p['biased'] = True
                algo = NMF(**p)
                mae_arr = []
                rmse_arr = []
                res = cross_validate(algo, data, measures=['rmse', 'mae'], cv=kf,
                               return_train_measures=False, n_jobs=3,verbose=False)
                mae_all.append(np.mean(res['test_mae']))
                rmse_all.append(np.mean(res['test_rmse']))
                params_all.append(p)
                if (iter-1) % N_print_interval == 0:
                    print_output('... set %i' % iter, end='')
                    print_output('  mae=%f  rmse=%f  params=%s' % (mae_all[-1], rmse_all[-1],str(params_all[-1])))
        print_top('NMF best score and params for MAE:',mae_all,params_all,3)
        print_top('NMF best score and params for RMSE:',rmse_all,params_all,3)

        outfile.close()
