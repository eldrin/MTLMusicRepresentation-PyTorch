import glob
from os.path import join, dirname
from functools import partial

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC, SVR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, r2_score

from tqdm import tqdm


TASK2LABEL_FN = {
    'eBallroom':'./eval/data/eBallroom_filtered_dataset.info',
    'FMA': './eval/data/FMA_filtered_dataset.info',
    'GTZAN': './eval/data/GTZAN_filtered_dataset.info',
    'IRMAS': './eval/data/IRMAS_filtered_dataset.info',
    'MusicEmoArousal': './eval/data/MusicEmoArousal_filtered_dataset.info',
    'MusicEmoValence': './eval/data/MusicEmoValence_filtered_dataset.info',
    'Jam': './eval/data/Jam/Jam.subset.no_blacklist_mel_safe.triplet',
    'Lastfm1k': './eval/data/Lastfm1k/lastfm55.subset.audiosafe.triplet'
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
    'classification': {'LinearSVC': LinearSVC,
                       'MLPClassifier': partial(MLPClassifier, hidden_layer_sizes=(256, 256))},
    'regression': {'SVR': SVR,
                   'MLPRegressor': partial(MLPRegressor, hidden_layer_sizes=(256, 256))},
    'recommendation': None  # wmfa (will be plugged in soon)
}

TASKMETRICS = {
    'classification': accuracy_score,
    'regression': r2_score,
    'recommendation': None # ap@40 (will be plugged in soon)
}


def load_data(fn, task):
    """"""
    assert task in TASKTYPE
    
    # loading feature
    X = np.load(fn)
    
    # loading targets
    if TASKTYPE[task] == 'recommendation':
        # triplet
        Y = pd.read_csv(TASK2LABEL_FN[task],
                        header=None, index_col=None)
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
        return StratifiedKFold(n_splits=n_cv, shuffle=True,
                               random_seed=random_seed).split(X, label)
    else:
        train_ix = Y[Y[2].isin({'train','valid'})][0].values
        test_ix = Y[Y[2].isin({'test'})][0].values
        
        for _ in range(n_cv):
            np.random.shuffle(train_ix)
            yield train_ix, test_ix


def run(feature_fns, task, n_cv=5):
    """"""
    result = []
    
    for fn in tqdm(feature_fn, total=len(feature_fn), ncols=80):
        id = dirname(fn)
        X, Y = load_data(fn, task)
        
        for name, model in TASKMODELS[TASKTYPE[task]].items():
            if task != 'recommendation':
                y = Y[3].values
                for train_ix, test_ix in split_generator(X, Y, n_cv):
                    model.fit(X[train_ix], y[train_ix])
                    y_pred = model.predict(X[test_ix])
                    y_true = y[test_ix]
                    res = TASKMETRICS[TASKTYPE[task]](y_true, y_pred)
                    
                    # store
                    result.append({'id': id, 'task': task, 'model': name, 'value': res,})
            else:
                raise NotImplementedError
    
    return result