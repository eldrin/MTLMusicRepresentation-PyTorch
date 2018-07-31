import os
import sys
# add repo root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
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

from musmtl.model import VGGlikeMTL, SpecStandardScaler
from musmtl.utils import extract_mel
from musmtl.config import Config as cfg


@background(max_prefetch=5)
def _generate_mels(fns):
    for fn in tqdm(fns, ncols=80):
        yield fn, extract_mel(fn)
        

class FeatureExtractor(object):
    def __init__(self, exp_config, exp_idx, model_checkpoint,
                 target_audios, is_gpu=False):
        """
        Args:
            exp_config (str): path to experiment configuration file (.json)
            exp_idx (int): index of corresponding experiment
            model_checkpoint (str): path to model checkpoint file (.pth.gz)
            target_audios (str): path to file listing target audio files (.txt)
        """
        # build & initialize model
        self.is_gpu = is_gpu
        self.conf = json.load(open(exp_config))
        self.model = VGGlikeMTL(
            self.conf['experiments'][exp_idx]['tasks'],
            self.conf['experiments'][exp_idx]['branch_at']
        )

        # load checkpoint to the model
        checkpoint = torch.load(model_checkpoint)
        self.model.load_state_dict(checkpoint['state_dict'])
        
        # initialize scaler
        sclr_ = joblib.load(self.conf['scaler_fn'])
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

            y = y.transpose(0, 2, 1)[None]  # (1, 2, t, 128)
            
            # re-organize it to have (N, 2, 216, 128)
            Y = y.reshape(-1, cfg.N_CH, cfg.N_STEPS, cfg.N_BINS)
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


if __name__ == "__main__":
    # setup arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_config", help='path to experiment configuration file (.json)')
    parser.add_argument("exp_idx", type=int,
                        help='index of corresponding experiment')
    parser.add_argument("model_checkpoint", help='path to model checkpoint file (.pth.gz)')
    parser.add_argument("target_audios", help='path to file listing target audio files (.txt)')
    parser.add_argument("out_fn", help='filename for the output')
    parser.add_argument("--is-gpu", type=bool, default=False,
                        help='flag for gpu computation on feature extraction')
    args = parser.parse_args()
    
    # init worker and run!
    print('Initiating worker...')
    ext = FeatureExtractor(
        args.exp_config, args.exp_idx,
        args.model_checkpoint, args.target_audios,
        args.is_gpu
    )
    feature = ext.run()
    
    print('Saving...')
    np.save(args.out_fn, feature)
    