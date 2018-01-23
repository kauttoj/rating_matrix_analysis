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
import itertools

if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    TEXT_PATH = r'D:/GoogleDrive/kysely/'  # data folder

    SEED = 666  # data partition seed
    ranks = [4,6,8,10,12,14,16,18,20]  # number of components in SVD and NMF, all will be tested
    N_FOLD = 20  # data folds to test
    N_cores = 4  # how many parallel jobs
    N_samples = 100  # number of random grid search samples
    N_print_interval=30

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
        s_end=''
        for key in params:
            count[key] = [len(params[key]),0]
            s_end+=str(count[key][0])
            total *= len(params[key])
        param_set = []
        if 2 * N_samples > total:  # make sure our search space is large (otherwise can stuck in loop for now)
            print_output('!!!!!! running all parameters')
            keys, values = zip(*params.items())
            param_set = [dict(zip(keys, v)) for v in itertools.product(*values)]
        else:
            used = set()
            while len(param_set) < N_samples:
                p = {}
                s = ''
                for key in params:
                    k = int(np.random.choice(count[key][0], 1))
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
        param_grid = {'method':['sgd'],
                  'n_epochs': [25],
                  'learning_rate': [0.005,0.01,0.03,0.001],
                  'reg': [0.1,0.07,0.02,0.01,0.005]
                  }
        params = random_search(param_grid)
        mae_all = []
        rmse_all = []
        print_output('\nRunning SGD Baseline with %ix%i=%i parameter sets' % (len(ranks), len(params), len(ranks) * len(params)))
        iter = 0
        params_all = []
        for p in params:
            iter += 1
            algo = BaselineOnly(bsl_options=p)
            res = cross_validate(algo, data, measures=['rmse', 'mae'], cv=kf,
                                 return_train_measures=False, n_jobs=N_cores, verbose=False)
            mae_all.append(np.mean(res['test_mae']))
            rmse_all.append(np.mean(res['test_rmse']))
            params_all.append(p.copy())
            if (iter - 1) % N_print_interval == 0:
                print_output('... set %i' % iter, end='')
                print_output('  mae=%f  rmse=%f  params=%s' % (mae_all[-1], rmse_all[-1], str(params_all[-1])))
        print_top('SGD Baseline best score and params for MAE:', mae_all, params_all, 5)
        print_top('SGD Baseline score and params for RMSE:', rmse_all, params_all, 5)

        param_grid = {'method':['als'],
                      'n_epochs': [15],
                      'reg_u': [1, 5, 15, 20, 30],
                      'reg_i': [1, 5, 10, 15, 30]
                      }
        params = random_search(param_grid)
        mae_all = []
        rmse_all = []
        print_output('\nRunning ALS Baseline with %ix%i=%i parameter sets' % (len(ranks), len(params), len(ranks) * len(params)))
        iter = 0
        params_all = []
        for p in params:
            iter += 1
            algo = BaselineOnly(bsl_options=p)
            res = cross_validate(algo, data, measures=['rmse', 'mae'], cv=kf,
                                 return_train_measures=False, n_jobs=N_cores, verbose=False)
            mae_all.append(np.mean(res['test_mae']))
            rmse_all.append(np.mean(res['test_rmse']))
            params_all.append(p.copy())
            if (iter - 1) % N_print_interval == 0:
                print_output('... set %i' % iter, end='')
                print_output('  mae=%f  rmse=%f  params=%s' % (mae_all[-1], rmse_all[-1], str(params_all[-1])))
        print_top('ALS Baseline best score and params for MAE:', mae_all, params_all, 5)
        print_top('ALS Baseline score and params for RMSE:', rmse_all, params_all, 5)


        # #%% SVD testing
        # param_grid = {'lr_bu':  [0.05,0.005,0.001],
        #               'lr_bi':  [0.05,0.005,0.001],
        #               'lr_pu':  [0.05,0.005,0.001],
        #               'lr_qi':  [0.05,0.005,0.001],
        #               'reg_bu': [0.5,0.05,0.005,0.0005],
        #               'reg_pu': [0.5,0.05,0.005,0.0005],
        #               'reg_bi': [0.5,0.05,0.005,0.0005],
        #               'reg_qi': [0.5,0.05,0.005,0.0005],
        #               }
        # params=random_search(param_grid)
        # mae_all=[]
        # rmse_all=[]
        # print_output('\nRunning SVD with %i parameter sets' % (len(ranks) * len(params)))
        # iter=0
        # params_all=[]
        # for rank in ranks:
        #     for p in params:
        #         iter+=1
        #         p['n_factors']=rank
        #         p['n_epochs']=30
        #         algo = SVD(**p)
        #         res = cross_validate(algo, data, measures=['rmse', 'mae'], cv=kf,
        #                        return_train_measures=False, n_jobs=N_cores,verbose=False)
        #         mae_all.append(np.mean(res['test_mae']))
        #         rmse_all.append(np.mean(res['test_rmse']))
        #         params_all.append(p)
        #         if (iter-1) % N_print_interval  == 0:
        #             print_output('... set %i' % iter, end='')
        #             print_output('  mae=%f  rmse=%f  params=%s' % (mae_all[-1], rmse_all[-1],str(params_all[-1])))
        # print_top('SVD best score and params for MAE:',mae_all,params_all,5)
        # print_top('SVD best score and params for RMSE:',rmse_all,params_all,5)

        # NMF testing
        param_grid = {'reg_pu': [0.1,0.5,1.0,1.5],
                      'reg_qi': [0.5,1.0,1.5],
                      'reg_bu': [0.5,0.3,0.1],
                      'reg_bi': [0.1,0.01,0.005,0],
                      'lr_bu' : [0.01,0.005,0.0025],
                      'lr_bi' : [0.005,0.0025]
                      }
        params = random_search(param_grid)
        mae_all = []
        rmse_all = []
        print_output('\nRunning NMF with %ix%i=%i parameter sets' % (len(ranks),len(params),len(ranks) * len(params)))
        iter=0
        params_all = []
        for rank in ranks:
            for p in params:
                iter+=1
                p['n_factors'] = rank
                p['n_epochs'] = 60
                p['biased'] = True
                algo = NMF(**p)
                res = cross_validate(algo, data, measures=['rmse', 'mae'], cv=kf,
                               return_train_measures=False, n_jobs=N_cores,verbose=False)
                mae_all.append(np.mean(res['test_mae']))
                rmse_all.append(np.mean(res['test_rmse']))
                params_all.append(p.copy())
                if (iter-1) % N_print_interval == 0:
                    print_output('... set %i' % iter, end='')
                    print_output('  mae=%f  rmse=%f  params=%s' % (mae_all[-1], rmse_all[-1],str(params_all[-1])))
        print_top('NMF best score and params for MAE:',mae_all,params_all,5)
        print_top('NMF best score and params for RMSE:',rmse_all,params_all,5)

        outfile.close()
