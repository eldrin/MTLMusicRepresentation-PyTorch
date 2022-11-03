#!/usr/bin/python
import os
from pathlib import Path
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

# NOTE: make sure export where MSD songs located
N_WORKERS = 2


def parse_arguments() -> argparse.Namespace:
    """
    """
    # setup arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("songsroot", help='root where the all MSD .mp3 located')
    parser.add_argument("tid2fn", help='map between MSD tid => fn (.pkl)')
    parser.add_argument("candidates", help='text file containing target candidate tids')
    parser.add_argument("outroot", help='output root where all the mel-spec saved')
    parser.add_argument("--prefix", type=str, default='msd_melspec',
                        help=("name to be used to uniquely identify "
                              "the set of processed mel spectrum files."))
    parser.add_argument('--overwrite', default=True,
                        action=argparse.BooleanOptionalAction,
                        help="set if the melspectrum is overwritten if exists.")
    parser.add_argument('--verbose', default=True,
                        action=argparse.BooleanOptionalAction,
                        help="set verbosity")
    parser.add_argument("--workers", type=int, default=N_WORKERS,
                        help='number of workers doing the job')
    return parser.parse_args()

args = parse_arguments()


# prep some path related preproc
songsroot = Path(args.songsroot)
outroot = Path(args.outroot)

assert songsroot.exists()

# generate output directory if not exists
outroot.mkdir(parents=True, exist_ok=True)

# load MSD tid => fn map
print('1. Loading path map...')
tid2fn = pkl.load(open(args.tid2fn, 'rb'))

# read candidates
print('2. Loading candidates...')
with open(args.candidates) as f:
    tids = [l.replace('\n', '') for l in f]

def _ext_mel(
    tid: str,
) -> None:
    """
    """
    in_fn = songsroot / tid2fn[tid]

    # prep output path
    out_path = outroot / tid[2] / tid[3]
    out_path.mkdir(parents=True, exist_ok=True)

    # knit output name
    out_fn = out_path / f'{tid}.npy'

    if out_fn.exists() and not args.overwrite:
        return

    try:
        Y = extract_mel(in_fn.as_posix())  # (n_ch, steps, bins)
        # Y = Y[None]  # (1, n_ch, steps, bins)
        np.save(out_fn.as_posix(), Y)
    except Exception as e:
        print(tid, e)

# process
print('3. Process & saving...')
parmap(_ext_mel, tids, n_workers=args.workers, verbose=args.verbose)

print('4. write the filename list of the mels')
with (outroot / f'{args.prefix}_filenames.txt').open('w') as fp:
    for fn in outroot.glob('*/*/*.npy'):
        fp.write(f'{fn}\n')
