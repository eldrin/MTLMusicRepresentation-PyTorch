import os
from os.path import join, basename
import glob
import six

# TEMPORARY SOLUTION
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from tqdm import tqdm

from .utils import load_pickle


class MSDMelDataset(Dataset):
    """ mel-spectrogram dataset of the subset of Million Song Dataset """
    def __init__(self, melpaths_fn, label_fn, crop_len=216, 
                 on_mem=True, transform=None):
        """
        Args:
            melpaths_fn (str): path to the text file containing 
                                candidate mel spectrograms (.npy)
            label_fn (str): path to label file (a dictionary contains all labels)
            transform (callable, optional): optional transform to be applied
        """
        # load mel data
        # : melpaths is 
        # : dictionary of tid: melspec (1, 2, t, 128)
        self.on_mem = on_mem
        with open(melpaths_fn) as f:
            melpaths = [line.replace('\n', '') for line in f.readlines()]
        
        tids = []
        if self.on_mem:
            # pre-load all melspec on memory
            # (suitable for low-mem / high speed storage & cpu machine)
            self.mels = {}
            for fn in tqdm(melpaths, ncols=80):
                tid = basename(fn).split('.npy')[0]
                tids.append(tid)
                self.mels[tid] = np.load(fn)
        else:
            # only keeps paths for lazy loading
            # (suitable for hi-mem / low speed storage machine)
            self.melpaths = {}
            for fn in melpaths:
                tid = basename(fn).split('.npy')[0]
                tids.append(tid)
                self.melpaths[tid] = fn

        # load label data
        # : dictionary of dictionary (~700mb)
        # : each sub-dict contains tid: item_factor
        # : no self (should be generated dynamically)
        self.labels = load_pickle(label_fn) 
        
        # get the intersection of tids between mel and labels
        self.tids = list(set(tids).intersection(
            set(six.next(six.itervalues(self.labels)).keys())
        ))
        
        # make hash for tids
        self.i_tids = dict(enumerate(self.tids))
        self.tids_i = {v:k for k, v in self.i_tids.items()}
        
        self.transform = transform
        self.crop_len = crop_len

    def __len__(self):
        """"""
        return len(self.i_tids)
    
    def __getitem__(self, idx_task):
        """"""
        # unpack
        idx, task = idx_task
        
        # retrieve mel spec
        x = self._load_mel(idx)
        # random cropping
        x_ = self._crop_mel(x)

        if task == 'self_':
            # generate ((X_l, X_r), y_self)
            X_l = x_
            y_self_ = np.random.randint(2)
            if y_self_ == 1:
                # select 2 chunks from same song
                X_r = self._crop_mel(x)
                y_self = np.array([0., 1.])
            else:
                # select random negative sample
                neg_idx = self._negative_sampling(self.i_tids[idx])
                X_r = self._load_mel(neg_idx)
                X_r = self._crop_mel(X_r)
                y_self = np.array([1., 0.])

            sample = {'mel': (X_l, X_r), 'label': y_self}
        else:    
            # retrive (X_t, z_t)            
            sample = {'mel': x_, 'label': self.labels[task][self.i_tids[idx]]}
    
        if self.transform:
            sample = self.transform(sample)
    
        return sample

    def _load_mel(self, idx):
        """"""
        tid = self.i_tids[idx]
        if self.on_mem:
            return self.mels[tid]
        else:
            return np.load(self.melpaths[tid], mmap_mode='r')
            
    def _crop_mel(self, mel):
        """"""
        if mel.shape[2] >= self.crop_len:
            st = np.random.randint(mel.shape[2] - self.crop_len)
            return mel[:, :, st: st+self.crop_len]
        else:
            # zero-pad first
            mel_ = np.zeros((1, 2, 216, 128), dtype=np.float32)
            st_ = np.random.randint(216 - mel.shape[2])
            mel_[:, :, st_:st_+mel.shape[2]] = mel
            return mel_

    def _negative_sampling(self, tid):
        """""" 
        neg_idx = np.random.choice(self.tids)
        while neg_idx == tid:
            neg_idx = np.random.choice(self.tids)
        return self.tids_i[neg_idx]
    
    
class ToVariable(object):
    """ Convert ndarrays in sample in Variables. """
    def __call__(self, sample):
        """"""
        mel, label = sample['mel'], sample['label']
        if isinstance(mel, tuple):
            mel = (
                Variable(torch.from_numpy(mel[0]).float()),
                Variable(torch.from_numpy(mel[1]).float())
            )
        else:
            mel = Variable(torch.from_numpy(mel).float())

        return {'mel': mel,
                'label': Variable(torch.from_numpy(label).float())}