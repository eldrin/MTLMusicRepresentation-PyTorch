import os
import json

from functools import partial

# TEMPORARY SOLUTION
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
from sklearn.externals import joblib
import librosa

import torch
import torch.nn as nn
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

    @staticmethod
    def _extract(scaler, model, Y, is_gpu):
        # scaling & extraction
        Y = scaler(Y)
        Y = torch.cat(
            [model.feature(Y, task) for task in model.tasks],
            dim=1
        )

        if is_gpu:
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


class EasyFeatureExtractor(nn.Module):
    """ High-Level wrapper class for easier feature extraction """
    def __init__(self, model_fn, scaler_fn='./data/sclr_dbmel.dat.gz',
                 is_gpu=False):
        """
        Args:
            model_fn (str): path to the target VGGlikeMTL model parameter (.pth)
            scaler_fn (str): path to the mel-spectrum scaler (.dat.gz)
            is_gpu (bool): flag for the gpu computation
        """
        super().__init__()
        self.is_gpu = is_gpu
        self.scaler, self.model = FeatureExtractor._load_model(
            model_fn, scaler_fn, self.is_gpu)
        self.melspec = partial(librosa.feature.melspectrogram,
                               n_fft=cfg.N_FFT, hop_length=cfg.HOP_LEN)

    def forward(self, audio):
        """
        Args:
            audio (numpy.ndarray):
                audio tensor. only supports up to 2 channel (n_ch, sig_len)

        Outputs:
            output (numpy.ndarray):
                feature tensor. (512 * model.n_tasks)
        """
        # check audio validity
        audio = self._check_n_fix_audio(audio)

        # get melspec
        mel_ = []
        for channel in audio[:2]:
            mel_.append(self.melspec(channel))
        mel = np.array(mel_)

        # preprocess
        X = FeatureExtractor._preprocess_mel(mel, self.is_gpu)

        # extract
        z = FeatureExtractor._extract(self.scaler, self.model, X, self.is_gpu)

        # output
        return z 

    def _check_n_fix_audio(self, audio):
        """
        Args:
            audio (numpy.ndarray):
                audio tensor. only supports up to 2 channel (n_ch, sig_len)
        """
        if audio.ndim == 1:  # vector
            return auduio[None]
        elif audio.ndim == 2:  # multi-channel audio
            if audio.shape[0] == 1:
                return np.r_[audio, audio]  # stack to psuedo multi channel
            else:
                return audio
        else:
            raise ValueError('[ERROR] audio input must have either (sig_len,) \
                             or (n_ch, sig_len)!')


def load_model(model_fn, is_gpu=False):
    """"""
    # load the checkpoint
    # load the state_dict to the model
    # output loaded model
