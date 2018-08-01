from os.path import join, dirname, basename
import shutil
import pickle as pkl
from multiprocessing import Pool

# TEMPORARY SOLUTION
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
import torch
import librosa
from tqdm import tqdm

from .config import Config as cfg


def load_pickle(fn):
    """ python2 -> python3 loadable pickle loader
    
    Args:
        fn (str): path to pickle file

    ref: https://blog.modest-destiny.com/posts/python-2-and-3-compatible-pickle-save-and-load/
    """ 
    try:
        with open(fn, 'rb') as f:
            data = pkl.load(f)
    except UnicodeDecodeError as e:
        with open(fn, 'rb') as f:
            data = pkl.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data', fn, ':', e)
        raise
    return data


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """"""
    torch.save(state, filename)
    if is_best:
        new_basename = basename(filename)
        new_basename = new_basename.replace('checkpoint', 'model_best')
        new_fn = join(dirname(filename), new_basename)
        shutil.copyfile(filename, new_fn)

        
def extract_mel(fn, verbose=False):
    """"""
    y, sr = librosa.load(fn, sr=cfg.SR, mono=False)
    if y.ndim == 1:
        if verbose:
            print('[Warning] "{}" only has 1 channel. '.format(fn) +
                  'making psuedo 2-channels...')
        y = np.vstack([y, y])

    Y = librosa.power_to_db(np.array([
        librosa.feature.melspectrogram(
            ch, sr=sr, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LEN).T
        for ch in y
    ])).astype(np.float32)  # (2, t, 128)

    return Y  # (2, t, 128)


def parmap(func, iterable, n_workers=2, verbose=False):
    """ Simple Implementation for Parallel Map """
    
    if n_workers == 1:
        if verbose:
            iterable = tqdm(iterable, total=len(iterable), ncols=80)
        return map(func, iterable)
    else:
        with Pool(processes=n_workers) as p:
            if verbose:
                with tqdm(total=len(iterable), ncols=80) as pbar:
                    output = []
                    for o in p.imap_unordered(func, iterable):
                        output.append(o)
                        pbar.update()
                return output
            else:
                return p.imap_unordered(func, iterable)