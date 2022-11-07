import os
import json

from functools import partial
import argparse
import importlib.resources as importlib_resources
from pathlib import Path
import logging

# TEMPORARY SOLUTION
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
import joblib
import librosa

import torch
from torch.autograd import Variable
from tqdm import tqdm

from .model import VGGlikeMTL, SpecStandardScaler, DEFAULT_SCALER_REF
from .utils import extract_mel
from .config import Config as cfg


logging.basicConfig()
logger = logging.getLogger(__name__)


def _generate_mels(fns):
    for fn in tqdm(fns, ncols=80):
        try:
            if os.path.splitext(fn)[-1] == '.npy':
                y = np.load(fn, mmap_mode='r')
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
    def __init__(self, target_audios, device=False):
        """
        Args:
            target_audios (str): path to file listing target audio files (.txt)
        """
        # build & initialize model
        self.device = device

        # get the target melspec list
        with open(target_audios, 'r') as f:
            self.mel_fns = [ll.replace('\n', '') for ll in f.readlines()]

    @staticmethod
    def _load_model(model_fn, scaler_fn, device):
        """"""
        if model_fn is not None:
            # load checkpoint to the model
            checkpoint = torch.load(
                model_fn, map_location=lambda storage, loc: storage)
            n_ch_in = checkpoint['state_dict']['shared.0.conv.weight'].shape[1]
            model = VGGlikeMTL(checkpoint['tasks'],
                               checkpoint['branch_at'],
                               n_ch_in=n_ch_in)
            model.load_state_dict(checkpoint['state_dict'])
        else:  # Random feature case
            model = VGGlikeMTL(['tag'], "null")
        model.eval()

        # initialize scaler
        sclr_ = joblib.load(scaler_fn)
        scaler = SpecStandardScaler(sclr_.mean_, sclr_.scale_)

        # send the models to computing device
        scaler.to(device)
        model.to(device)

        return scaler, model

    @staticmethod
    def _preprocess_mel(y, device, config=None):
        """"""
        if config is None:
            config = dict(n_steps=cfg.N_STEPS,
                          n_ch=cfg.N_CH,
                          n_bins=cfg.N_BINS)
        else:
            config['n_steps'] = cfg.N_STEPS

        # zero padding
        if y.shape[1] % config['n_steps'] != 0:
            margin = config['n_steps'] - y.shape[1] % config['n_steps']
            y = np.concatenate(
                [y, np.zeros((config['n_ch'], margin, config['n_bins']))],
                axis=1
            ).astype(np.float32)  # (n_ch, t, n_bins)

        # re-organize it to have (M, n_ch, t, n_bins)
        Y = np.concatenate(
            [y[j].reshape(-1, config['n_steps'], config['n_bins'])[:, None]
             for j in range(config['n_ch'])],
            axis=1
        )
        Y = Variable(torch.from_numpy(Y).float()).to(device)

        return Y

    @staticmethod
    def _extract(scaler, model, Y, device):
        # scaling & extraction
        Y = scaler(Y)
        # Y = torch.cat(
        #     [model.feature(Y, task) for task in model.tasks],
        #     dim=1
        # )
        #
        # # send to CPU / numpy.ndarray
        # Y = Y.data.detach().cpu().numpy()
        #
        # # concat of mean / std
        # z = np.r_[Y.mean(axis=0), Y.std(axis=0)]

        Z = {}
        for task in model.tasks:
            y = model.feature(Y, task)
            z = torch.cat([y.mean(dim=0), y.std(dim=0)])
            Z[task] = z.detach().cpu().numpy()
        return Z

    def run(self, model_fns, scaler_fn):
        """
        Args:
            model_fns (list of str): list of path to model checkpoint file (.pth.gz)
        """
        # load all melspecs
        # TODO: this approach is memory heavy, but fast. better way?
        # X = list(_generate_mels(self.mel_fns))
        X = _generate_mels(self.mel_fns)

        for model_fn in model_fns:
            # spawn model
            if model_fn == 'random':
                scaler, model = self._load_model(None, scaler_fn, self.device)
            elif model_fn == 'mfcc':
                pass
            else:
                scaler, model = self._load_model(model_fn, scaler_fn, self.device)

            # initiate output containor
            output = {task:[] for task in model.tasks}
            processed_fns = []

            # process
            for fn, x in X:
                if model_fn == 'mfcc':
                    # extract MFCC baseline
                    output.append(mfcc_baseline(x[None]))

                else:
                    Y = self._preprocess_mel(x, self.device)
                    z = self._extract(scaler, model, Y, self.device)
                    for task in model.tasks:
                        output[task].append(z[task])

                # register processed filename into the container
                processed_fns.append(Path(fn).name)

            # post process
            output = {
                task: np.array(features)
                for task, features in output.items()
            }
            yield processed_fns, output  # (n x (d x m))


class EasyFeatureExtractor:
    """ High-Level wrapper class for easier feature extraction """
    def __init__(self, model_fn, config_fn, is_gpu=False):
        """
        Args:
            model_fn (str): path to the target VGGlikeMTL model parameter (.pth)
            config_fn (str): path to the model training config file (.json)
            is_gpu (bool): flag for the gpu computation
        """
        self.is_gpu = is_gpu
        self.config = config
        self.scaler, self.model = FeatureExtractor._load_model(
            model_fn, config['scaler_fn'], self.is_gpu)
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
        mel = np.array(mel_).transpose(0, 2, 1)

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
            return np.r_[audio[None], audio[None]]
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
    pass


def parse_arguments() -> argparse.Namespace:
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("model_checkpoints",
                        help=('path to file listing model checkpoint '
                              'files (.txt)'))
    parser.add_argument("target_audios",
                        help='path to file listing target audio files (.txt)')
    parser.add_argument("out_root",
                        help='filepath for the output')
    parser.add_argument("--scaler", type=str, default=None,
                        help=("filename of the melspectrogram scaler. "
                              "If not given, it uses the default scaler "
                              "which trained on about 200k music preview "
                              "of million song dataset."))
    parser.add_argument('--device', type=str, default='cpu',
                        help="select compute device for the extraction.")
    parser.add_argument('--verbose', default=True,
                        action=argparse.BooleanOptionalAction,
                        help="set verbosity")
    return parser.parse_args()


def main():
    """
    """
    args = parse_arguments()

    if args.verbose:
        logging.getLogger(__name__).setLevel(logging.INFO)

    # device sanity check
    if (args.device != 'cpu') and (not args.device.startswith('cuda')):
        raise ValueError('[ERROR] wrong device name is given!')

    if args.device.startswith('cuda') and (not torch.cuda.is_available()):
        raise ValueError('[ERROR] GPU is not available in this system! '
                         'the procedure will be computed on CPU...')

    # parse some path objects and names
    outroot = Path(args.out_root)
    if not outroot.exists():
        logger.info('Output path does not exist! making the directory...')
        outroot.mkdir(parents=True, exist_ok=True)
    in_name = Path(args.target_audios).name.split('_')[0]

    # parse model paths
    model_fns = []
    with open(args.model_checkpoints) as f:
        for line in f:
            fn = line.replace('\n', '')

            # check input / output path is existing and valid
            if fn != 'mfcc' and fn != 'random' and not os.path.exists(fn):
                raise IOError('[ERROR] model checkpoint not exists!')

            model_fns.append(fn)

    # init worker and run!
    logger.info('Initiating worker...')
    ext = FeatureExtractor(args.target_audios, args.device)

    # fetch the scaler file object
    logger.info('Processing...')
    with importlib_resources.as_file(DEFAULT_SCALER_REF) \
            if args.scaler is None else \
            Path(args.scaler) as fp:

        for (succssed, feature), fn in zip(ext.run(model_fns, fp), model_fns):

            # get exp id
            ix = Path(fn).stem.split('_')[0]
            out_fn = outroot / ix / f"{in_name}"
            out_fn.parent.mkdir(parents=True, exist_ok=True)

            logger.info('Saving {}...'.format(ix))
            np.savez(out_fn, fns=succssed, **feature)


if __name__ == "__main__":
    raise SystemExit(main())
