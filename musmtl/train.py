import os
from os.path import join
from collections import OrderedDict
from itertools import chain
import uuid

# TEMPORARY SOLUTION
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
from sklearn.externals import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

from .model import VGGlikeMTL, SpecStandardScaler
from .data import MSDMelDataset, ToVariable
from .optimizer import MultipleOptimizerDict
from .utils import save_checkpoint

from tqdm import trange, tqdm
import fire


class Trainer(object):
    """ Model trainer """
    def __init__(self, train_dataset, tasks, branch_at, scaler_fn,
                 out_root, in_fn=None, learn_rate=0.0001, n_epoches=100,
                 batch_sz=48, l2=1e-6, valid_dataset=None, save_every=None,
                 report_every=100, is_gpu=True, name=None, **kwargs):
        """
        Args:
            train_dataset (MSDMelDataset): training dataset instance
            tasks (list of str): involved tasks
                    ({'self', 'year', 'bpm', 'tag', 'taste', 'cdr', 'artist'})
            branch_at (str): point where the network branches
                             ({'2', '4', '6', 'fc'})
            scaler_fn (str): path to saved sklearn.preprocessing.StandardScaler
                            containing mean / scale of mel spectrums
            out_root (str): output path for checkpoints
            in_fn (str, optional): checkpoint fn (not implemented yet)
            learn_rate (float): learning rate for Adam optimizer
            n_epochs (int): number of epoches for training
            batch_sz (int): number of samplers per batch
            l2 (float): coefficient for L2 regularization on weight
            valid_dataset (MSDMelDataset, optional): validation dataset instance
            on_mem (bool): flag whether the data loaded on memory or not
            report_every (int): frequency for evaluation
            is_gpu (bool): flag whether gpu is used or not
            name (str): name for the experimental run. randomly generaged if not given
        """
        self.tasks = tasks
        self.branch_at = branch_at

        self.learn_rate = learn_rate
        self.batch_sz = batch_sz
        self.n_epoches = n_epoches
        self.l2 = l2
        
        self.report_every = report_every
        self.save_every = save_every
        
        self.scaler_fn = scaler_fn
        
        self.is_gpu = is_gpu
        self.in_fn = in_fn
        self.out_root = out_root
        # prepare dir
        if not os.path.exists(self.out_root):
            os.mkdir(self.out_root)

        # initialize tb-logger
        if name is None:
            self.logger = SummaryWriter()
            self.name = ''
        else:
            self.logger = SummaryWriter('runs/{}'.format(name))
            self.name = name

        # initialize the dataset
        self.train_dataset = train_dataset
        if valid_dataset is not None:
            self.valid_dataset = valid_dataset

        self.n_batches = int(len(self.train_dataset.tids) / self.batch_sz) + 1
        self.n_batches *= len(tasks)

        # initialize the models
        sclr = joblib.load(scaler_fn)
        self.scaler = SpecStandardScaler(sclr.mean_, sclr.scale_)
        self.model = VGGlikeMTL(tasks, branch_at)

        # initialize optimizer
        ops = OrderedDict([
            ('shared', optim.Adam(
                self.model.shared.parameters(),
                lr = self.learn_rate / len(self.tasks),
                weight_decay = self.l2))
        ])
        for task in tasks:
            feat_net = self.model.branches_feature[self.model._task2idx[task]]
            inf_net = self.model.branches_infer[self.model._task2idx[task]]
            ops.update(
                {
                    task: optim.Adam(
                        chain(feat_net.parameters(), inf_net.parameters()),
                        lr = self.learn_rate,
                        weight_decay = self.l2
                    )
                }   
            )
        self.opt = MultipleOptimizerDict(**ops)

        # setup loss
        self.criterion = F.kl_div
        # self.criterion = categorical_crossentropy
        
        if self.is_gpu:
            self.model.cuda()
            self.scaler.cuda()

    def fit(self):
        """"""
        self.iters = 0
        self.epoch = 0
        try:
            for n in trange(self.n_epoches, ncols=80):
                self.epoch = n
                self.model.train()
                
                # per-epoch checkpoint
                if ((self.save_every is not None) and 
                    (self.epoch % self.save_every == 0)):
                    self._validate()

                for _ in trange(self.n_batches, ncols=80):
                    # pick task & indices
                    task = np.random.choice(self.tasks)
                    idx = np.random.choice(
                        len(self.train_dataset.tids), self.batch_sz, False)

                    # draw data
                    X, y = self._draw_samples(idx, task, 'train')

                    # update the model
                    self.opt.zero_grad()
                    X = self.scaler(X)
                    y_pred = self.model(X, task)
                    l = self.criterion(
                        F.log_softmax(y_pred, dim=1), y)
                    l.backward()
                    self.opt.step(['shared', task])
                    self.iters += 1

                    # runtime checkpoint
                    if (self.iters % self.report_every == 0 and 
                            hasattr(self, 'valid_dataset')):

                        # training log
                        self.logger.add_scalar(
                            'tloss/{}'.format(task), l[0], self.iters)

                        # validation
                        if self.save_every is not None:
                            self._validate(save=False)
                        else:
                            self._validate()
                        
        except KeyboardInterrupt:
            print('[Warning] User stopped the training!')
        
        finally:
            # for the case where the training accidentally break
            self._validate()
            
    def _draw_samples(self, idx, task, dset='train'):
        """"""
        if dset == 'train':
            dataset = self.train_dataset
        else:
            dataset = self.valid_dataset
        X, y = [], []
        for i in idx:
            sample = self.train_dataset[i, task]
            X.append(sample['mel'])
            y.append(sample['label'][None, :])
        X = torch.cat(X, dim=0)
        y = torch.cat(y, dim=0)
        
        if self.is_gpu:
            X, y = X.cuda(), y.cuda()
            
        return X, y
    
    def _validate(self, save=True):
        """"""
        self.model.eval()  # toggle to evaluation mode
        
        # validation
        for task in self.tasks:
            idx = np.random.choice(
                len(self.valid_dataset.tids), self.batch_sz, False)
            
            # draw data
            X, y = self._draw_samples(idx, task, 'valid')
            
            # forward
            X = self.scaler(X)
            y_pred = self.model(X, task)
            l = self.criterion(F.log_softmax(y_pred, dim=1), y)
            
            # logging
            self.logger.add_scalar('vloss/{}'.format(task), 
                                   l[0], self.iters)       
        if save:
            if self.save_every is not None:
                out_fn = ('{}_checkpoint_it{:d}.pth.tar'
                          .format(self.name, self.epoch))
            else:
                out_fn = ('{}_checkpoint.pth.tar'
                          .format(self.name))

            save_checkpoint(
                {'iters': self.iters,
                 'tasks': self.tasks,
                 'branch_at': self.branch_at,
                 'scaler_fn': self.scaler_fn,
                 'state_dict': self.model.state_dict(),
                 'optimizer': self.opt.state_dict()},
                False,  # we don't care best model
                join(self.out_root , out_fn)
            )

        # toggle to training mode
        self.model.train()   
        
        
def categorical_crossentropy(input, target):
    """ Categorical Crossentropy Op for probability input
    
    Args:
        input (torch.tensor): log probability over classes (N x C)
        target (torch.tensor): linear probability over classes (N x C)
    """
    return -(input * target).sum(1).mean(0)