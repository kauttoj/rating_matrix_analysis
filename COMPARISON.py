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
import pandas
import pickle

if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    TEXT_PATH = r'D:/GoogleDrive/kysely/'  # data folder

    SEED = 666  # data partition seed
    ranks = [2,4,6,8,10,15,20]  # number of components in SVD and NMF, all will be tested
    N_FOLD = 20  # data folds to test
    N_cores = 4  # how many parallel jobs
    N_samples = 150  # number of random grid search samples
    N_print_interval=30

    ALL_RESULTS = {}

    def get_average():
        values = {x:[] for x in data_full._raw2inner_id_items.keys()}
        for user in data_full._raw2inner_id_users.keys():
            user_internal = data_full._raw2inner_id_users[user]
            for raw_item in data_full.ur[user_internal]:
                item = data_full.to_raw_iid(raw_item[0])
                values[item].append(raw_item[1])
        return {x:np.mean(values[x]) for x in values}

    def get_matrix(algo):
        datamat = pandas.DataFrame(index=data_full._raw2inner_id_items.keys(),
                                   columns=data_full._raw2inner_id_users.keys())
        item_bias = {}
        for item in data_full._raw2inner_id_items.keys():
            raw_item = data_full._raw2inner_id_items[item]
            item_bias[item]=algo.bi[raw_item]
        for user in data_full._raw2inner_id_users.keys():
            user_internal = data_full._raw2inner_id_users[user]
            values={x[0]:x[1] for x in data_full.ur[user_internal]}
            for item in data_full._raw2inner_id_items.keys():
                raw_item = data_full._raw2inner_id_items[item]
                if raw_item in values:
                    datamat.loc[item, user] = values[raw_item]
                else:
                    datamat.loc[item, user] = algo.predict(user, item, verbose=False).est
        return datamat,item_bias

    def print_output(s,end='\n'):
        print(s,end=end)
        print(s,end=end,file=outfile)

    def print_top(head,arr,param,n):
        k = np.argsort(arr)
        print_output(head)
        for i in range(n):
            print_output(' %i ... %f with params %s' % (i+1,arr[k[i]],str(param[k[i]])))
        return {'error':arr[k[0]],'params':param[k[0]]}

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
        if 2 * N_samples > total:  # if whole space is only double of samples, do all parameters
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
        data_full = data.build_full_trainset()

        ALL_RESULTS[dimension]={}

        ALL_RESULTS[dimension]['item_mean'] = get_average()

        f = 'output_%s.txt' % dimension
        outfile = open(f,'w')

        print_output('\n----- Processing dataset \'%s\' -----\n' % dimension)

        #%% BASELINE with SGD
        param_grid = {'method':['sgd'],
                  'n_epochs': [50], # ei muutostarvetta
                  'learning_rate': [0.002,0.01],
                  'reg': [0.8,0.5,0.15,0.12,0.1,0.05,0.001]
                  }
        params = random_search(param_grid)
        mae_all = []
        rmse_all = []
        print_output('\nRunning SGD Baseline with %i parameter sets' % len(params))
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
        best_mae_params = print_top('SGD Baseline best score and params for MAE:', mae_all, params_all, 5)
        best_rmse_params = print_top('SGD Baseline score and params for RMSE:', rmse_all, params_all, 5)
        algo = BaselineOnly(bsl_options=best_rmse_params['params'])
        algo.fit(data_full)
        matrix,item_bias = get_matrix(algo)
        ALL_RESULTS[dimension]['BASELINE_SGD'] = {'mae_cv':best_mae_params['error'],'rmse_cv':best_rmse_params['error'],'rating_matrix':matrix,'item_bias':item_bias}

        # %% BASELINE with ALS
        param_grid = {'method':['als'],
                      'n_epochs': [30], # ei muutostarvetta
                      'reg_u': [0.1,0.7,1,3,5,15,20],
                      'reg_i': [0.1,0.5,1,3,5,8]
                      }
        params = random_search(param_grid)
        mae_all = []
        rmse_all = []
        print_output('\nRunning ALS Baseline with %i parameter sets' % len(params))
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
        best_mae_params=print_top('ALS Baseline best score and params for MAE:', mae_all, params_all, 5)
        best_rmse_params=print_top('ALS Baseline score and params for RMSE:', rmse_all, params_all, 5)
        algo = BaselineOnly(bsl_options=best_rmse_params['params'])
        algo.fit(data_full)
        matrix,item_bias = get_matrix(algo)
        ALL_RESULTS[dimension]['BASELINE_ALS'] = {'mae_cv':best_mae_params['error'],'rmse_cv':best_rmse_params['error'],'rating_matrix':matrix,'item_bias':item_bias}

        pickle.dump(ALL_RESULTS, open(TEXT_PATH + r'SURPRISE_RESULTS.pickle', 'wb'))

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
        param_grid = {'n_epochs':[50], # ei muutostarvetta
                      'reg_pu': [0.01,0.1,0.5,1.0,1.5,2.0,3.0,5.0],
                      'reg_qi': [0.1,0.5,1.0,1.5,2.0,3.0,5.0],
                      'reg_bu': [0.6,0.3,0.1,0.05,0.01,0.001],
                      'reg_bi': [0.05,0.001,0.0001,0.00001], # yleensä pieni
                      'lr_bu' : [0.01,0.005,0.001], # ei muutostarvetta
                      'lr_bi' : [0.01,0.005]  # ei muutostarvetta
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
        best_mae_params=print_top('NMF best score and params for MAE:',mae_all,params_all,5)
        best_rmse_params=print_top('NMF best score and params for RMSE:',rmse_all,params_all,5)
        algo = NMF(**best_rmse_params['params'])
        algo.fit(data_full)
        matrix,item_bias = get_matrix(algo)
        ALL_RESULTS[dimension]['NMF'] = {'mae_cv':best_mae_params['error'],'rmse_cv':best_rmse_params['error'],'rating_matrix':matrix,'item_bias':item_bias}

        outfile.close()
        pickle.dump(ALL_RESULTS, open(TEXT_PATH + r'SURPRISE_RESULTS.pickle', 'wb'))
