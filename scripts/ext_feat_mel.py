import os
from os.path import dirname, basename, join, exists
import sys
# add repo root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from musmtl.tool import FeatureExtractor
# from musmtl.utils import extract_mel, parmap


# setup arg parser
parser = argparse.ArgumentParser()
parser.add_argument("model_checkpoints", help='path to file listing model checkpoint files (.txt)')
parser.add_argument("target_mels", help='path to file listing target audio files (.txt)')
parser.add_argument("dataset", type=str, help='the name of the dataset (str)')
parser.add_argument("melfn2ids", help='file contains filename to songid map (.csv)')
parser.add_argument("out_root", help='filepath for the output')
parser.add_argument("--scaler-fn", type=str, default=None,
                    help='path to the mel-standard scaler')
parser.add_argument('--gpu', dest='is_gpu', action='store_true')
parser.add_argument('--no-gpu', dest='is_gpu', action='store_false')
args = parser.parse_args()

# gpu sanity check
if args.is_gpu and (torch.cuda.device_count() == 0):
    print('[Warning] No GPU found! the process will be done on CPU...')

if args.scaler_fn is None:
    # falling back to the default scaler
    SCALER_FN = os.path.join(os.path.dirname(__file__), '..', 'models/sclr_dbmel.dat.gz')
else:
    SCALER_FN = args.scaler_fn

device = 'cpu' if not args.is_gpu else 'cuda'

# get mel paths
mel_fns = []
with Path(args.target_mels).open('r') as fp:
    for line in fp:
        mel_fns.append(line.replace('\n', ''))

# load id map
with Path(args.melfn2ids).open('r') as fp:
    melfn2ids = {}
    for line in fp:
        mel_fn, id_ = line.replace('\n','').split(',')
        melfn2ids[mel_fn] = id_

# parse model paths
out_root = Path(args.out_root)
with open(args.model_checkpoints) as f:
    for line in f:
        model_fn = line.replace('\n', '')

        # load model
        scaler, model = FeatureExtractor._load_model(model_fn, SCALER_FN, args.is_gpu)

        # run
        Z = []
        ids = []
        with tqdm(total=len(mel_fns), ncols=80) as prog:
            for fn in mel_fns:
                fn = Path(fn)
                ids.append(melfn2ids[fn.name])
                x = torch.from_numpy(np.load(fn)).to(device)
                y = scaler(x)
                z = torch.cat(
                    [model.feature(y, task) for task in model.tasks],
                    dim=1
                )
                Z.append(z.detach().cpu().numpy())
                prog.update()
        Z = np.array(Z)[:, 0]
        ids = np.array(ids)

        fn = Path(fn)
        out_fn = out_root / f'kim_{args.dataset}_{Path(model_fn).stem}.npz'
        out_fn.parent.mkdir(exist_ok=True, parents=True)
        np.savez(
            out_fn, feature=Z, ids=ids,
            dataset=np.array(args.dataset),
            model_class='kim',
            model_filename=np.array(fn.as_posix())
        )
