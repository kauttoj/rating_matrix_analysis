'''This module runs a 5-Fold CV for all the algorithms (default parameters) on
the movielens datasets, and reports average RMSE, MAE, and total computation
time.  It is used for making tables in the README.md file'''

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import time
import datetime
import random

import numpy as np
import six
from tabulate import tabulate

from surprise import Reader
import os
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from surprise import NormalPredictor
from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNBaseline
from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from surprise import SlopeOne
from surprise import CoClustering

if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    # The algorithms to cross-validate
    classes = (SVD, SVDpp, NMF, SlopeOne, KNNBasic, KNNWithMeans, KNNBaseline,
               CoClustering, BaselineOnly, NormalPredictor)

    # ugly dict to map algo names and datasets to their markdown links in the table
    stable = 'http://surprise.readthedocs.io/en/stable/'
    LINK = {'SVD': '[{}]({})'.format('SVD',''),
            'SVDpp': '[{}]({})'.format('SVD++',''),
            'NMF': '[{}]({})'.format('NMF',''),
            'SlopeOne': '[{}]({})'.format('Slope One',''),
            'KNNBasic': '[{}]({})'.format('k-NN',''),
            'KNNWithMeans': '[{}]({})'.format('Centered k-NN',''),
            'KNNBaseline': '[{}]({})'.format('k-NN Baseline',''),
            'CoClustering': '[{}]({})'.format('Co-Clustering',''),
            'BaselineOnly': '[{}]({})'.format('Baseline',''),
            'NormalPredictor': '[{}]({})'.format('Random',''),
            'laurea_test': '[{}]({})'.format('pieni testidata','none'),
            }

    # set RNG
    np.random.seed(0)
    random.seed(0)

    limits = [1,8] # rating limits
    # path to dataset file
    file_path = os.path.expanduser(r'text')
    # As we're loading a custom dataset, we need to define a reader. In the
    # movielens-100k dataset, each line has the following format:
    # 'user item rating timestamp', separated by '\t' characters.
    reader = Reader(line_format='user item rating', sep='\t',rating_scale=(1,8))
    data = Dataset.load_from_file(file_path, reader=reader)

    kf = KFold(n_splits=10,random_state=0,shuffle=True)  # folds will be the same for all algorithms.

    table = []
    for klass in classes:
        start = time.time()
        out = cross_validate(klass(), data, ['rmse', 'mae'], kf)
        cv_time = str(datetime.timedelta(seconds=int(time.time() - start)))
        link = LINK[klass.__name__]
        mean_rmse = '{:.3f}'.format(np.mean(out['test_rmse']))
        mean_mae = '{:.3f}'.format(np.mean(out['test_mae']))

        new_line = [link, mean_rmse, mean_mae, cv_time]
        print(tabulate([new_line], tablefmt="pipe"))  # print current algo perf
        table.append(new_line)

    header = [LINK['laurea_test'],
              'RMSE',
              'MAE',
              'Time'
              ]
    print(tabulate(table, header, tablefmt="pipe"))