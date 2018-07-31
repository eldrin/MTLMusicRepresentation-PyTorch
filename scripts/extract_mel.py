#!/usr/bin/python
import os
from os.path import join
import sys
# add repo root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pickle as pkl
import argparse
from multiprocessing import Pool
import warnings
# TEMPORARY SOLUTION
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import librosa
import numpy as np

from musmtl.utils import extract_mel

from tqdm import tqdm

# NOTE: make sure export where MSD songs located
MSDROOT = os.environ['MSDROOT']
N_WORKERS = 2

# setup arg parser
parser = argparse.ArgumentParser()
parser.add_argument("tid2fn", help='map between MSD tid => fn (.pkl)')
parser.add_argument("candidates", help='text file containing target candidate tids')
parser.add_argument("outroot", help='output root where all the mel-spec saved')
args = parser.parse_args()

# load MSD tid => fn map
print('1. Loading path map...')
tid2fn = pkl.load(open(args.tid2fn, 'rb'))

# read candidates
print('2. Loading candidates...')
with open(args.candidates) as f:
    tids = [l.replace('\n', '') for l in f.readlines()]

# helper
def _ext_mel(tid):
    """"""
    in_fn = join(MSDROOT, tid2fn[tid])
    out_fn = join(args.outroot, tid + '.npy')
    
    if os.path.exists(out_fn):
        return

    Y = extract_mel(in_fn)
    
    # (1, 2, steps, bins)
    np.save(out_fn, Y.transpose(0, 2, 1)[None])

# process
print('3. Process & saving...')
if N_WORKERS == 1:
    [_ext_mel(tid) for tid in tqdm(tids, total=len(tids), ncols=80)]
else:
    with Pool(processes=N_WORKERS) as p:
        with tqdm(total=len(tids), ncols=80) as pbar:
            for _ in p.imap_unordered(_ext_mel, tids):
                pbar.update()
