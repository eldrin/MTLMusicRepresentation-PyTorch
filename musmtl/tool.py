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


@background(max_prefetch=5)
def _generate_mels(fns):
    for fn in tqdm(fns, ncols=80):
        yield fn, extract_mel(fn)
        

class FeatureExtractor(object):
    def __init__(self, model_checkpoint, target_audios, is_gpu=False):
        """
        Args:
            model_checkpoint (str): path to model checkpoint file (.pth.gz)
            target_audios (str): path to file listing target audio files (.txt)
        """
        # build & initialize model
        self.is_gpu = is_gpu
        
        # load checkpoint to the model
        checkpoint = torch.load(model_checkpoint)
        self.model = VGGlikeMTL(checkpoint['tasks'],
                                checkpoint['branch_at'])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        
        # initialize scaler
        sclr_ = joblib.load(checkpoint['scaler_fn'])
        self.scaler = SpecStandardScaler(sclr_.mean_, sclr_.scale_)
        
        if self.is_gpu:
            self.scaler.cuda()
            self.model.cuda()
    
        # get the target list
        with open(target_audios, 'r') as f:
            self.fns = [ll.replace('\n', '') for ll in f.readlines()]
        
    def run(self):
        """"""
        output = []
        for fn, y in _generate_mels(self.fns):

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
            
            if self.is_gpu:
                Y = Y.cuda()
            
            # scaling & extraction
            Y = self.scaler(Y)
            Y = torch.cat(
                [self.model.feature(Y, task)
                 for task in self.model.tasks],
                dim=1
            )
            
            if self.is_gpu:
                Y = Y.data.cpu().numpy()
            else:
                Y = Y.data.numpy()
            
            # concat of mean / std
            z = np.r_[Y.mean(axis=0), Y.std(axis=0)]
            output.append(z)
            
        return np.array(output)  # (n x (d x m))