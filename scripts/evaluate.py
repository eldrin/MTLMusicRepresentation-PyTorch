import os
from os.path import dirname, basename, join, exists
import sys
# add repo root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse

from musmtl.evaluation import run

# setup arg parser
parser = argparse.ArgumentParser()
parser.add_argument("feature_fn", help='path to the feature file')
parser.add_argument("task",
                    help="""name of the task
                    {'eBallroom', 'FMA', 'GTZAN', 'IRMAS',
                    'MusicEmoArousal', 'MusicEmoValence', 'Jam', 'Lastfm1k'}""")
parser.add_argument("out_root", help='path to the dir where to save the result (.csv)')
parser.add_argument("--n-cv", type=int, default=5, help='number of split for validation')
parser.add_argument('--standardize', dest='is_standardize', action='store_true')
parser.add_argument('--no-standardize', dest='is_standardize', action='store_false')
parser.add_argument('--grid-search', dest='is_grid_search', action='store_true')
parser.add_argument('--no-grid-search', dest='is_grid_search', action='store_false')
args = parser.parse_args()

# run!
run(args.feature_fn, args.task, args.out_root, args.n_cv,
    args.is_standardize, args.is_grid_search)
