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
args = parser.parse_args()

for i, fn in enumerate(filter(lambda fn: TASKKEY[args.task] == basename(fn),
                              glob.glob(join(args.feature_dir, '*/*.npy')))):
    if i > 0:
        break

    id = basename(dirname(fn))
    if exists(join(args.out_root, '{}_{}.csv'.format(id, args.task))):
        continue
    subprocess.call(['sbatch', './sbatchs/evaluate.sbatch',
                     fn, args.task, args.out_root, str(args.n_cv)])
