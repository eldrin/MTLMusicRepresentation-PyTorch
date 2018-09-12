import os
os.environ['OMP_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['NUMBA_NUM_THREADS'] = "2"

from os.path import join, dirname, basename
import sys
import glob
from functools import partial
import time

import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC, SVR
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, r2_score

from tqdm import tqdm

# adding RecSys submodule to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../recsys/'))

from cfmodels.utils import read_data, densify, df2csr
from cfmodels.validation import split_outer
from cfmodels.factorization import WMFA
from cfmodels.factorization.wmf.wmf import linear_transform
from cfmodels.metrics import AveragePrecision, NDCG, Recall


TASK2LABEL_FN = {
    'eBallroom':'./eval/data/eBallroom_filtered_dataset.info',
    'FMA': './eval/data/FMA_filtered_dataset.info',
    'GTZAN': './eval/data/GTZAN_filtered_dataset.info',
    'IRMAS': './eval/data/IRMAS_filtered_dataset.info',
    'MusicEmoArousal': './eval/data/MusicEmoArousal_filtered_dataset.info',
    'MusicEmoValence': './eval/data/MusicEmoValence_filtered_dataset.info',
    'Jam': {'triplet': './eval/data/Jam/Jam.subset.no_blacklist_mel_safe.triplet',
            'items': './eval/data/Jam/Jam.subset.no_blacklist_mel_safe.jamhash',
            'users': './eval/data/Jam/Jam.subset.no_blacklist_mel_safe.userhash'},
    'Lastfm1k': {'triplet': './eval/data/Lastfm1k/lastfm55.subset.audiosafe.triplet',
                 'items': './eval/data/Lastfm1k/lastfm55.subset.audiosafe.item.hash',
                 'users': './eval/data/Lastfm1k/lastfm55.subset.audiosafe.user.hash'}
}

TASKTYPE = {
    'eBallroom': 'classification',
    'FMA': 'classification',
    'GTZAN': 'classification',
    'IRMAS': 'classification',
    'MusicEmoArousal': 'regression',
    'MusicEmoValence': 'regression',
    'Jam': 'recommendation',
    'Lastfm1k': 'recommendation'
}

TASKKEY = {
    'eBallroom': 'eBallroom.npy',
    'FMA': 'FMA.npy',
    'GTZAN': 'GTZAN.npy',
    'IRMAS': 'IRMAS.npy',
    'MusicEmoArousal': 'MusicEmotion.npy',
    'MusicEmoValence': 'MusicEmotion.npy',
    'Jam': 'Jam.npy',
    'Lastfm1k': 'Lastfm1k.npy'
}

TASKMODELS = {
    'classification': {'LinearSVC': [LinearSVC],
                       'MLPClassifier': [partial(MLPClassifier, hidden_layer_sizes=(256, 256))]},
    'regression': {'SVR': [StandardScaler, SVR],
                   'MLPRegressor': [partial(MLPRegressor, hidden_layer_sizes=(256, 256))]},
    'recommendation': None  # wmfa (will be plugged in soon)
}

TASKMETRICS = {
    'classification': accuracy_score,
    'regression': r2_score,
    'recommendation': None # ap@40 (will be plugged in soon)
}

RECSYS_SETUP = {
    'n_factors':50,
    'alpha':0.1,
    'reg_phi':0.1,
    'reg_wh':0.00001,
    'init':0.01,
    'n_epochs':15,
    'verbose':1
}

RECSYS_MONITORS = [AveragePrecision(k=80), NDCG(k=None), Recall(k=80)]


def load_data(fn, task):
    """"""
    assert task in TASKTYPE

    # loading feature
    X = np.load(fn)

    # loading targets
    if TASKTYPE[task] == 'recommendation':
        # triplet
        # (csr_matrix::interaction_matrix, pd.DataFrame::triplet)
        Y = read_data(TASK2LABEL_FN[task]['triplet'],
                      columns=['user', 'item', 'value'])
    else:
        # id----filename----split----label
        Y = pd.read_csv(TASK2LABEL_FN[task],
                        header=None, index_col=None, sep='\t')
    return X, Y


def split_generator(X, Y, n_cv=5, random_seed=1234):
    """"""
    if Y[2].isna().unique() == True:
        label = Y[3].values
        # returning generator
        split_gen = StratifiedKFold(
            n_splits=n_cv, shuffle=True,
            random_state=random_seed
        ).split(X, label)
    else:
        train_ix = Y[Y[2].isin({'train', 'valid'})][0].values
        test_ix = Y[Y[2].isin({'test'})][0].values

        split_gen = []
        for _ in range(n_cv):
            np.random.shuffle(train_ix)
            split_gen.append((train_ix, test_ix))

    for train_ix, test_ix in split_gen:
        yield train_ix, test_ix


def eval_recsys(id, triplet, R, X, train_ratio, model_setup,
                monitors=[AveragePrecision(k=120),
                          NDCG(k=None), Recall(k=120)]):
    """"""
    task = 'recommendation'
    result = []

    # split (for out-of-matrix (Hao Wang 2016.) evaluation)
    train, test = split_outer(triplet, ratio=train_ratio)

    Rtr = df2csr(train, R.shape)
    Rts = df2csr(test, R.shape)

    # evaluate with our setup
    model = WMFA(monitors=monitors, report_every=None,
                 **model_setup)

    # fit & score
    # # set number of thread explicitly
    # # : for recsys, numba threading is more important
    model.fit((Rtr, X), (Rts, X))
    res = model.score(Rtr, Rts, X)

    # record
    for metric_name, metric_value in res.items():
        result.append(
            {'id': id, 'task': task, 'model': 'WMFA', 'split': 'dense',
             'metric': metric_name, 'value': metric_value}
        )

    return result


def densify_triplet(triplet, X, user_min=20, item_min=50):
    """"""
    ix2tracks = {v:k for k, v in enumerate(triplet['item'].unique())}
    ix2users = {v:k for k, v in enumerate(triplet['user'].unique())}
    triplet_dense = densify(triplet, user_min=user_min, item_min=item_min)

    # reindexing
    triplet_dense.loc[:, 'user'] = triplet_dense['user'].map(ix2users)
    triplet_dense.loc[:, 'item'] = triplet_dense['item'].map(ix2tracks)

    # get new id
    uniq_users = {v:k for k, v in enumerate(triplet_dense['user'].unique())}
    uniq_items = {v:k for k, v in enumerate(triplet_dense['item'].unique())}

    triplet_dense.loc[:, 'user'] = triplet_dense['user'].map(uniq_users)
    triplet_dense.loc[:, 'item'] = triplet_dense['item'].map(uniq_items)

    # build CSR
    R_ = sp.coo_matrix(
        (triplet_dense['value'], (triplet_dense['user'], triplet_dense['item']))
    ).tocsr()
    X_ = X[list(uniq_items.keys())]

    return triplet_dense, R_, X_


def run(feature_fn, task, out_root, n_cv=5):
    """"""
    assert task in TASKTYPE

    result = []
    id = basename(dirname(feature_fn))
    out_fn = join(out_root, '{}_{}.csv'.format(id, task))
    X, Y = load_data(feature_fn, task)

    if TASKTYPE[task] != 'recommendation':  # CLASSIFICATION / REGRESSION

        for name, Models in TASKMODELS[TASKTYPE[task]].items():
            # set number of thread explicitly
            # : for clf / reg, numpy (openmp) threading is more important
            os.environ['OMP_NUM_THREADS'] = "2"
            os.environ['NUMBA_NUM_THREADS'] = "2"

            y = Y[3].values
            if task == 'regression':
                y = y.astype(float)

            for train_ix, test_ix in split_generator(X, Y, n_cv):
                # model initialization & training
                pipes = [('{}_{:d}'.format(name, i), Pipe())
                         for i, Pipe in enumerate(Models)]
                model = Pipeline(pipes).fit(X[train_ix], y[train_ix])
                y_pred = model.predict(X[test_ix])
                y_true = y[test_ix]
                res = TASKMETRICS[TASKTYPE[task]](y_true, y_pred)

                # store
                result.append({'id': id, 'task': task, 'model': name, 'value': res,})

    else:  # RECOMMENDATION

        # standardizing
        sclr = StandardScaler()
        X = sclr.fit_transform(X)

        # item_hash = pd.read_csv(TASK2LABEL_FN[task]['items'V],
        #                         header=None, index_col=None)
        R, triplet = Y  # parse the data

        # 1. Our setup
        # min_items_per_user = 5
        # min_users_per_item = 5
        # held-out tracks : 5%
        # metrics : R@40 / AP@40 / NDCG
        for i in range(n_cv):
            tic = time.time()
            result.extend(
                eval_recsys(
                    id, triplet, R, X.T, 0.95,
                    model_setup=RECSYS_SETUP,
                    monitors=[
                        AveragePrecision(k=80),
                        NDCG(k=500), Recall(k=80)
                    ]
                )
            )
            toc = time.time()
            print('[Our setup] Processed {:d}th fold. ({:.2f}s taken)'
                  .format(i, toc - tic))

        # 2. H. Wang's setup (DENse filtering for the data)
        # min_items_per_user = 20
        # min_users_per_item = 50
        # held-out tracks : 5%
        # metrics : R@40 / AP@40 / NDCG
        triplet_dense, R_, X_ = densify_triplet(triplet, X)
        for i in range(n_cv):
            tic = time.time()
            result.extend(
                eval_recsys(
                    id, triplet_dense, R_, X_.T, 0.95,
                    model_setup=RECSYS_SETUP,
                    monitors=RECSYS_MONITORS
                )
            )
            toc = time.time()
            print('[Wang\'s setup] Processed {:d}th fold. ({:.2f}s taken)'
                  .format(i, toc - tic))

    # save
    pd.DataFrame(result).to_csv(out_fn, index=None)
