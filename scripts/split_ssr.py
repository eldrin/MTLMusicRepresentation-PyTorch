import os
from os.path import basename, dirname, join, exists
import sys
# add repo root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import glob
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm


# source map
src2name = {
    's1': 'self_',
    's2': 'bpm',
    's3': 'year',
    's4': 'taste',
    's5': 'tag',
    's6': 'lyrics',
    's7': 'cdr_tag',
    's8': 'artist'
}

parser = argparse.ArgumentParser()
parser.add_argument("feature_dir", help='path to the root of feature files')
parser.add_argument("ssr_split", help='filename contains the ssr split plan (.txt)')
args = parser.parse_args()


# loading the plan
plan = pd.read_csv(args.ssr_split, index_col=None)
d = 256
for i, row in plan.iterrows():

    # setup paths
    id = row['index']
    model_id = row['model_index']

    # retrieve actual name
    src_name = src2name[row['source']]

    out_dir = join(args.feature_dir, '{}{:d}'.format(src_name, id))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # picking up indices
    mean_i_start = (row['position'] - 1) * d
    mean_i_end = mean_i_start + d

    std_offset = row['n'] * d
    std_i_start = std_offset + mean_i_start
    std_i_end = std_offset + mean_i_end

    # run process
    for feature_fn in tqdm(glob.glob(
            join(join(args.feature_dir, str(model_id)), '*.npy')), ncols=80):

        # load feature
        X = np.load(feature_fn)

        # slice according to the source we want
        x = np.concatenate(
            [X[:, mean_i_start:mean_i_end],
             X[:, std_i_start:std_i_end]],
            axis=1
        )

        # save sliced feature
        out_fn = join(out_dir, basename(feature_fn))
        np.save(out_fn, x)
