import os
from os.path import basename, dirname, join, exists
import sys
# add repo root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import glob
import argparse
import pickle as pkl

import numpy as np
import pandas as pd
import torch
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
parser.add_argument("model_dir", help='path to the root of feature files')
parser.add_argument("exp_design", help='path to the experimental design file (.csv)')
parser.add_argument("--k", type=int, default=5,
                    help='number of SSRs sampled from MSSCRs')
args = parser.parse_args()

# 0. get the procesed files
feature_file_fns = glob.glob(join(args.feature_dir, '*/*.npy'))
model_file_fns = glob.glob(join(args.model_dir, '*/*199.pth.tar'))
model_ids_processed = [basename(fn).split('_')[0].split('.')[-1]
                       for fn in model_file_fns]
model_ids_processed_only_from_design = []
for j in model_ids_processed:
    try:
        model_ids_processed_only_from_design.append(int(j))
    except ValueError:
        pass
        # print('Not a int')

# 1. loading the plan
exp_plan = pd.read_csv(args.exp_design, index_col=None)

# 2. check the "null" cases
exp_plan_msscr = exp_plan[
    (exp_plan['arc'] == 1) &
    (exp_plan.index.isin(model_ids_processed_only_from_design))
]

# 2.1. sample each cases
K = 5
cands = {}
for src, name in src2name.items():
    cands[name] = set(map(str,
        exp_plan_msscr[exp_plan_msscr[src] == 1].sample(K).index))

# 2.2. organize and get the final plan
msscr_idx = set(map(str, exp_plan_msscr.index))

# 3. find the starting / ending idx for slicing
# 4. put the info into some containor to actual process
# prepare feature file fns
target_info = {}
for model_id, fn in zip(model_ids_processed, model_file_fns):
    if model_id in msscr_idx:
        # find corresponding feature file name
        feat_fn = list(filter(
            lambda f: model_id == basename(dirname(f)),
            feature_file_fns
        ))

        # # if it's not processed
        # if len(feat_fn) == 0:
        #     continue

        # load the model checkpoint to get the tasks involved
        checkpoint = torch.load(
            fn, map_location=lambda storage, loc: storage)
        tasks = checkpoint['tasks']
        target_info[model_id] = {
            'model_fn': fn,
            'feat_fns': feat_fn,
            'tasks': tasks
        }
# pkl.dump(target_info, open('test.info.pkl', 'wb'))

# loading the plan
d = 256
for src_name, model_ids in tqdm(cands.items(), ncols=80):

    for id, model_id in enumerate(model_ids):

        # setup paths
        out_dir = join(args.feature_dir, '{}{:d}'.format(src_name, id))
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        # picking up indices
        tasks = target_info[model_id]['tasks']
        src_loc = tasks.index(src_name)
        n = len(tasks)

        mean_i_start = src_loc * d
        mean_i_end = mean_i_start + d

        std_offset = n * d
        std_i_start = std_offset + mean_i_start
        std_i_end = std_offset + mean_i_end

        # run process
        for feature_fn in tqdm(target_info[model_id]['feat_fns'], ncols=80):

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
