import os
import json

# TEMPORARY SOLUTION
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
from sklearn.externals import joblib
import librosa

import torch
from torch.autograd import Variable
from tqdm import tqdm

from .model import VGGlikeMTL, SpecStandardScaler
from .utils import extract_mel
from .config import Config as cfg


def _generate_mels(fns):
    for fn in tqdm(fns, ncols=80):
        try:
            if os.path.splitext(fn)[-1] == '.npy':
                y = np.load(fn)[0]
            else:
                y = extract_mel(fn)
        except IOError as e:
            print(e)
        except Exception as e:
            print(e)
        else:
            yield fn, y


def mfcc_baseline(X):
    """"""
    # X => (1, 2, n_steps, n_bins)
    x = X[0].mean(0)  # => (n_steps, n_bins)
    m = librosa.feature.mfcc(S=x.T).T
    dm = m[1:] - m[:-1]
    ddm = dm[1:] - dm[:-1]
    return np.r_[
        m.mean(0), m.std(0),
        dm.mean(0),dm.std(0),
        ddm.mean(0), ddm.std(0)
    ]


class FeatureExtractor(object):
    def __init__(self, target_audios, is_gpu=False):
        """
        Args:
            target_audios (str): path to file listing target audio files (.txt)
        """
        # build & initialize model
        self.is_gpu = is_gpu

        # get the target melspec list
        with open(target_audios, 'r') as f:
            self.mel_fns = [ll.replace('\n', '') for ll in f.readlines()]

    @staticmethod
    def _load_model(model_fn, scaler_fn, is_gpu):
        """"""
        if model_fn is not None:
            # load checkpoint to the model
            checkpoint = torch.load(
                model_fn, map_location=lambda storage, loc: storage)
            model = VGGlikeMTL(checkpoint['tasks'],
                               checkpoint['branch_at'])
            model.load_state_dict(checkpoint['state_dict'])
        else:  # Random feature case
            model = VGGlikeMTL(['tag'], "null")
        model.eval()

        # initialize scaler
        sclr_ = joblib.load(scaler_fn)
        scaler = SpecStandardScaler(sclr_.mean_, sclr_.scale_)

        if is_gpu:
            scaler.cuda()
            model.cuda()

        return scaler, model

    @staticmethod
    def _preprocess_mel(y, is_gpu):
        """"""
        # zero padding
        if y.shape[1] % cfg.N_STEPS != 0:
            margin = cfg.N_STEPS - y.shape[1] % cfg.N_STEPS
            y = np.concatenate(
                [y, np.zeros((cfg.N_CH, margin, cfg.N_BINS))],
                axis=1
            ).astype(np.float32)  # (2, t, 128)

        # re-organize it to have (N, 2, 216, 128)
        Y = np.concatenate(
            [y[j].reshape(-1, cfg.N_STEPS, cfg.N_BINS)[:, None]
             for j in range(cfg.N_CH)],
            axis=1
        )
        Y = Variable(torch.from_numpy(Y).float())

        if is_gpu:
            Y = Y.cuda()

        return Y

    def _extract(self, scaler, model, Y):
        # scaling & extraction
        Y = scaler(Y)
        Y = torch.cat(
            [model.feature(Y, task) for task in model.tasks],
            dim=1
        )

        if self.is_gpu:
            Y = Y.data.cpu().numpy()
        else:
            Y = Y.data.numpy()

        # concat of mean / std
        z = np.r_[Y.mean(axis=0), Y.std(axis=0)]
        return z

    def run(self, model_fns, scaler_fn):
        """
        Args:
            model_fns (list of str): list of path to model checkpoint file (.pth.gz)
        """
        # load all melspecs
        # TODO: this approach is memory heavy, but fast. better way?
        X = list(_generate_mels(self.mel_fns))

        for model_fn in model_fns:
                # initiate output containor
                output = []

                # spawn model
                if model_fn == 'random':
                    scaler, model = self._load_model(None, scaler_fn, self.is_gpu)
                elif model_fn == 'mfcc':
                    pass
                else:
                    scaler, model = self._load_model(model_fn, scaler_fn, self.is_gpu)

                # process
                for fn, x in X:
                    if model_fn == 'mfcc':
                        # extract MFCC baseline
                        output.append(mfcc_baseline(x[None]))

                    else:
                        Y = self._preprocess_mel(x, self.is_gpu)
                        z = self._extract(scaler, model, Y)
                        output.append(z)

                yield np.array(output)  # (n x (d x m))
