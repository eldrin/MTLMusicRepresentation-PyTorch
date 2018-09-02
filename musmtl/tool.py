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
from prefetch_generator import BackgroundGenerator, background
from tqdm import tqdm

from .model import VGGlikeMTL, SpecStandardScaler
from .utils import extract_mel
from .config import Config as cfg


# @background(max_prefetch=10)
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
    def _load_model(fn, is_gpu):
        """"""
        # load checkpoint to the model
        checkpoint = torch.load(
            fn, map_location=lambda storage, loc: storage)
        model = VGGlikeMTL(checkpoint['tasks'],
                           checkpoint['branch_at'])
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        # initialize scaler
        sclr_ = joblib.load(checkpoint['scaler_fn'])
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

    def run(self, model_fns):
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
            scaler, model = self._load_model(model_fn, self.is_gpu)

            # process
            for fn, x in X:
                Y = self._preprocess_mel(x, self.is_gpu)
                z = self._extract(scaler, model, Y)
                output.append(z)

            yield np.array(output)  # (n x (d x m))

        # # OLD WAY
        # for fn, y in _generate_mels(self.fns):
        #     output.append(z)
        # return np.array(output)  # (n x (d x m))
