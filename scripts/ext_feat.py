import os
import sys
# add repo root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.externals import joblib
import librosa

import torch
from torch.autograd import Variable
from tqdm import tqdm
import fire

from musmtl.model import VGGlikeMTL, SpecStandardScaler
from musmtl.utils import log_amplitude

N_CH = 2
N_STEPS = 216
N_BINS = 128


class FeatureExtractor(object):
    def __init__(self, model_config, model_checkpoint, target_audios, is_gpu=False):
        """
        Args:
            model_config (str): path to model configuration file (.json)
            model_checkpoint (str): path to model checkpoint file (.pth.gz)
            target_audios (str): path to file listing target audio files (.txt)
        """
        # build & initialize model
        self.model_confg = json.load(open(config_fn))
        self.model = VGGlikeMTL(
            self.model_config['tasks'],
            self.model_config['branch_at']
        )

        # load checkpoint to the model
        checkpoint = torch.load(model_checkpoint)
        self.model.load_state_dict(checkpoint['state_dict'])
        
        # initialize scaler
        sclr_ = joblib.load(self.model_config['scaler_fn'])
        self.scaler = SpecStandardScaler(sclr.mean_, sclr.scale_)
        
        if is_gpu:
            self.scaler.cuda()
            self.model.cuda()
    
        # get the target list
        with open(target_audios, 'rb') as f:
            self.fns = [ll.replace('\n', '') for ll in f.readlines()]
        
    def run(self):
        """"""
        output = {}
        for fn in tqdm(self.fns, ncols=80):
            y = extract_mel(fn)
            Y = y.reshape(-1, N_CH, N_STEPS, N_BINS)
            Y = Variable(torch.from_numpy(Y).float())
            
            # re-organize it to have (N, 2, 216, 128)
            Y = self.scaler(Y)
            Y = self.model(Y)
            
            if self.is_gpu:
                Y = Y.data.cpu().numpy()
            else:
                Y = Y.data.numpy()
            
            # concat of mean / std
            z = np.r_[Y.mean(axis=0), Y.std(axis=0)]
            output[fn] = z
            
        return output

    
def extract_mel(fn):
    """"""
    y, sr = librosa.load(fn, mono=False)   
    Y = log_amplitude(librosa.feature.melspectrogram(
        y, sr=sr, n_fft=1024, hop_length=256, power=1.))[None]
    
    # zero padding
    if Y.shape[2] % N_STEPS == 0:
        Y = np.concatenate(
            [Y, np.zeros((1, N_CH, N_STEPS - Y.shape[2] % N_STEPS, N_BINS))],
            axis=2
        ).astype(np.float32)
        
    return Y
    

if __name__ == "__main__":
    fire.Fire(FeatureExtractor)