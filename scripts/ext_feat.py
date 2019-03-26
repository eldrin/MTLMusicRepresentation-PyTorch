import os
from os.path import dirname, basename, join, exists
import sys
# add repo root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse

import numpy as np
import torch

from musmtl.tool import FeatureExtractor
from musmtl.utils import extract_mel, parmap

SCALER_FN = os.path.join(os.path.dirname(__file__), '..', 'models/sclr_dbmel.dat.gz')


# setup arg parser
parser = argparse.ArgumentParser()
parser.add_argument("model_checkpoints", help='path to file listing model checkpoint files (.txt)')
parser.add_argument("target_audios", help='path to file listing target audio files (.txt)')
parser.add_argument("out_root", help='filepath for the output')
parser.add_argument('--gpu', dest='is_gpu', action='store_true')
parser.add_argument('--no-gpu', dest='is_gpu', action='store_false')
args = parser.parse_args()

# gpu sanity check
if args.is_gpu and (torch.cuda.device_count() == 0):
    print('[Warning] No GPU found! the process will be done on CPU...')

# parse model paths
model_fns = []
with open(args.model_checkpoints) as f:
    for line in f:
        fn = line.replace('\n', '')

        # check input / output path is existing and valid
        if fn != 'mfcc' and fn != 'random' and not exists(fn):
            raise IOError('[ERROR] model checkpoint not exists!')

        model_fns.append(fn)

# init worker and run!
print('Initiating worker...')
ext = FeatureExtractor(args.target_audios, args.is_gpu)

print('Processing...')
for feature, fn in zip(ext.run(model_fns, SCALER_FN), model_fns):

    # get exp id
    ix = basename(fn).split('_')[0].split('.')[-1]
    out_fn = join(args.out_root, ix,
                  basename(args.target_audios).split('_')[0] + '.npy')
    if not exists(dirname(out_fn)):
        os.makedirs(dirname(out_fn))

    print('Saving {}...'.format(ix))
    np.save(out_fn, feature)
