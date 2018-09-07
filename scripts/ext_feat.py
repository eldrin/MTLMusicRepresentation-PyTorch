import os
from os.path import dirname, basename, join, exists
import sys
# add repo root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse

import numpy as np

from musmtl.tool import FeatureExtractor
from musmtl.utils import extract_mel, parmap

# setup arg parser
parser = argparse.ArgumentParser()
parser.add_argument("model_checkpoints", help='path to file listing model checkpoint files (.txt)')
parser.add_argument("scaler_fn", help='path to scaler model (.dat.gz)')
parser.add_argument("target_audios", help='path to file listing target audio files (.txt)')
parser.add_argument("out_root", help='filepath for the output')
parser.add_argument("--is-gpu", type=bool, default=False,
                    help='flag for gpu computation on feature extraction')
args = parser.parse_args()

# parse model paths
model_fns = []
with open(args.model_checkpoints) as f:
    for line in f:
        fn = line.replace('\n', '')

        # check input / output path is existing and valid
        if not exists(fn):
            raise IOError('[ERROR] model checkpoint not exists!')

        model_fns.append(fn)

# init worker and run!
print('Initiating worker...')
ext = FeatureExtractor(args.target_audios, args.is_gpu)

print('Processing...')
for feature, fn in zip(ext.run(model_fns, args.scaler_fn), model_fns):

    # get exp id
    ix = basename(fn).split('_')[0].split('.')[-1]
    out_fn = join(args.out_root, ix,
                  basename(args.target_audios).split('_')[0] + '.npy')
    if not exists(dirname(out_fn)):
        os.makedirs(dirname(out_fn))

    print('Saving {}...'.format(ix))
    np.save(out_fn, feature)
