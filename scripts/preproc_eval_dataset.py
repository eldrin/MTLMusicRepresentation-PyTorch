#!/usr/bin/python
import os
from os.path import join, exists
import sys
# add repo root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

import argparse
import warnings
# TEMPORARY SOLUTION
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from musmtl.utils import extract_mel, parmap

# setup arg parser
parser = argparse.ArgumentParser()
parser.add_argument("info", help='a text file contains info on dataset (idx, path, label)')
parser.add_argument("outroot", help='output root where all the processed saved')
parser.add_argument("--workers", type=int, default=N_WORKERS,
                    help='number of workers doing the job')
args = parser.parse_args()

# prepare output root
if not exists(args.outroot):
    os.mkdir(args.outroot)

# helper
def _process(line):
    """"""
    # assuming it's tab separated (i.e. idx\tfn\tlabel)
    idx, fn, label = line.replace('\n', '').split('\t')
    out_fn = join(args.outroot, '{:06d}.npy'.format(idx))
    Y = extract_mel(fn)  # (2, steps, bins)
    np.save(out_fn, Y)

# process
with open(args.info) as f:
    parmap(_process, f.readlines(),
           n_workers=N_WORKERS, verbose=True)