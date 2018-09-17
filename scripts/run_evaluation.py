import os
from os.path import basename, dirname, join, exists
import sys
# add repo root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import glob
import argparse
from tempfile import NamedTemporaryFile
import subprocess

from musmtl.evaluation import TASKKEY

parser = argparse.ArgumentParser()
parser.add_argument("feature_dir", help='path to the root of feature files')
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

for i, fn in enumerate(filter(lambda fn: TASKKEY[args.task] == basename(fn),
                              glob.glob(join(args.feature_dir, '*/*.npy')))):
    id = basename(dirname(fn))
    if exists(join(args.out_root, '{}_{}.csv'.format(id, args.task))):
        continue

    sclr = '--standardize' if args.is_standardize else '--no-standardize'
    grid = '--grid-search' if args.is_grid_search else '--no-grid-search'

    subprocess.call(['sbatch', './sbatchs/evaluate.sbatch',
                     fn, args.task, args.out_root, str(args.n_cv),
                     sclr, grid])
