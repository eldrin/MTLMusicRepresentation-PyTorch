#!/usr/bin/python
import os
from os.path import join
import sys
# add repo root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pickle as pkl
import argparse
import warnings
# TEMPORARY SOLUTION
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np

from musmtl.utils import extract_mel, parmap

from tqdm import tqdm

# NOTE: make sure export where MSD songs located
N_WORKERS = 2

# setup arg parser
parser = argparse.ArgumentParser()
parser.add_argument("songsroot", help='root where the all MSD .mp3 located')
parser.add_argument("tid2fn", help='map between MSD tid => fn (.pkl)')
parser.add_argument("candidates", help='text file containing target candidate tids')
parser.add_argument("outroot", help='output root where all the mel-spec saved')
parser.add_argument("--workers", type=int, default=N_WORKERS,
                    help='number of workers doing the job')
args = parser.parse_args()

# load MSD tid => fn map
print('1. Loading path map...')
tid2fn = pkl.load(open(args.tid2fn, 'rb'))

# read candidates
print('2. Loading candidates...')
with open(args.candidates) as f:
    tids = [l.replace('\n', '') for l in f]

# helper
def _ext_mel(tid, overwrite=True):
    """"""
    in_fn = join(args.songsroot, tid2fn[tid])
    out_fn = join(args.outroot, tid + '.npy')
    
    if os.path.exists(out_fn) and not overwrite:
        return

    Y = extract_mel(in_fn)  # (n_ch, steps, bins)
    Y = Y[None]  # (1, n_ch, steps, bins)
    np.save(out_fn, Y)

# process
print('3. Process & saving...')
parmap(_ext_mel, tids, n_workers=N_WORKERS, verbose=True)