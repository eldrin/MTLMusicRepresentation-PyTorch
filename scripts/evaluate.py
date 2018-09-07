import os
from os.path import dirname, basename, join, exists
import sys
# add repo root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse

from musmtl.evaluation import TASKTYPE, run

    'eBallroom': 'classification',
    'FMA': 'classification',
    'GTZAN': 'classification',
    'IRMAS': 'classification',
    'MusicEmoArousal': 'regression',
    'MusicEmoValence': 'regression',
    'Jam': 'recommendation',
    'Lastfm1k': 'recommendation'


# setup arg parser
parser = argparse.ArgumentParser()
parser.add_argument("feature_dir", help='path to root dir where all the features dumped')
parser.add_argument("task",
                    help="""name of the task 
                    {'eBallroom', 'FMA', 'GTZAN', 'IRMAS', 
                    'MusicEmoArousal', 'MusicEmoValence', 'Jam', 'Lastfm1k'}""")
parser.add_argument("out_fn", help='path to save the result (.csv)')
parser.add_argument("--n-cv", type=int, default=5, help='number of split for validation')
args = parser.parse_args()

# get the fns
fns = list(filter(lambda fn: task == basename(fn),
                  glob.glob(join(feature_dir, '*/*.npy'))))
pd.DataFrame(run(fns, task, n_cv)).to_csv(out_fn, index=None)  
