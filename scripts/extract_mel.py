#!/usr/bin/python
import os
from os.path import join
import pickle as pkl
import argparse
from multiprocessing import Pool
import warnings
# TEMPORARY SOLUTION
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import librosa
import numpy as np
from tqdm import tqdm

# NOTE: make sure export where MSD songs located
MSDROOT = os.environ['MSDROOT']
SR = 22050
N_FFT = 1024
HOP_LEN = 256
MONO = False
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
    
    y, sr = librosa.load(in_fn, sr=SR, mono=False)
    if y.ndim == 1:
        print('[Warning] "{}" only has 1 channel. making psuedo 2-channels...'
              .format(tid))
        y = np.vstack([y, y])

    Y = librosa.amplitude_to_db(np.array([
        librosa.feature.melspectrogram(
            ch, sr=sr, n_fft=N_FFT, hop_length=HOP_LEN)
        for ch in y
    ])).astype(np.float32)
    
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
