import os
from os.path import join, basename, dirname
import subprocess
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
from prefetch_generator import BackgroundGenerator, background

from .model import VGGlikeMTL, SpecStandardScaler
from .data import MSDMelDataset, ToVariable
from .optimizer import MultipleOptimizerDict
from .utils import save_checkpoint

from tqdm import trange, tqdm
import fire


class Trainer(object):
    """ Model trainer """
    def __init__(self, train_dataset, tasks, branch_at, scaler_fn,
                 out_root, n_outs=50, in_fn=None, learn_rate=0.0001, n_epoches=100,
                 batch_sz=48, l2=1e-6, valid_dataset=None,
                 save_every=None, save_location='localhost',
                 report_every=100, is_gpu=True, name=None, **kwargs):
        """
        Args:
            train_dataset (MSDMelDataset): training dataset instance
            tasks (list of str): involved tasks
                    ({'self_', 'year', 'bpm', 'tag', 'taste', 'cdr', 'artist'})
            branch_at (str): point where the network branches
                             ({'2', '4', '6', 'fc'})
            scaler_fn (str): path to saved sklearn.preprocessing.StandardScaler
                            containing mean / scale of mel spectrums
            out_root (str): output path for checkpoints
            n_outs (int): number of output dimension at terminal layer
            in_fn (str, optional): checkpoint fn (not implemented yet)
            learn_rate (float): learning rate for Adam optimizer
            n_epochs (int): number of epoches for training
            batch_sz (int): number of samplers per batch
            l2 (float): coefficient for L2 regularization on weight
            valid_dataset (MSDMelDataset, optional): validation dataset instance
            on_mem (bool): flag whether the data loaded on memory or not
            report_every (int): frequency for evaluation
            save_every (int): frequency for checkpoint dumping
            save_location (str): indicates whether root directory for the `out_root`
                                 is local or network server
                                 ({'localhost', '192.168.X.XXX', etc.})
            is_gpu (bool): flag whether gpu is used or not
            name (str): name for the experimental run. randomly generaged if not given
        """
        self.tasks = tasks
        self.branch_at = branch_at
        self.n_outs = n_outs

        self.learn_rate = learn_rate
        self.batch_sz = batch_sz
        self.n_epoches = n_epoches
        self.l2 = l2

        self.report_every = report_every
        self.save_every = save_every
        self.save_location = save_location

        self.scaler_fn = scaler_fn

        self.is_gpu = is_gpu
        self.in_fn = in_fn

        # initialize tb-logger
        if name is None:
            self.logger = SummaryWriter()
            self.name = ''
        else:
            self.logger = SummaryWriter('runs/{}'.format(name))
            self.name = name

        # setup dump locations
        self.out_root = out_root
        self.run_out_root = join(self.out_root, self.name)

        # prepare dirs
        if self.save_location == 'localhost':
            if not os.path.exists(self.out_root):
                os.mkdir(self.out_root)
            if not os.path.exists(self.run_out_root):
                os.mkdir(self.run_out_root)

        # initialize the dataset
        self.train_dataset = train_dataset
        if valid_dataset is not None:
            self.valid_dataset = valid_dataset

        self.n_batches = int(len(self.train_dataset.tids) / self.batch_sz) + 1
        self.n_batches *= len(tasks)

        # initialize the models
        sclr = joblib.load(scaler_fn)
        self.scaler = SpecStandardScaler(sclr.mean_, sclr.scale_)
        self.model = VGGlikeMTL(tasks, branch_at, n_outs=n_outs)

        # multi-gpu
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            self.multi_gpu = True
            print('[Info] Using {:d} gpus!'.format(
                torch.cuda.device_count()))
        else:
            self.multi_gpu = False

        # initialize optimizer
        ops = OrderedDict()
        if branch_at != 'null':
            ops.update(
                {
                    'shared': optim.Adam(
                        (self.model.module.shared.parameters()
                         if self.multi_gpu
                         else self.model.shared.parameters()),
                        lr = self.learn_rate,
                        weight_decay = self.l2)
                }
            )

        for task in tasks:
            if self.multi_gpu:
                task_i = self.model.module._task2idx[task]
                feat_net = self.model.module.branches_feature[task_i]
                inf_net = self.model.module.branches_infer[task_i]
            else:
                task_i = self.model._task2idx[task]
                feat_net = self.model.branches_feature[task_i]
                inf_net = self.model.branches_infer[task_i]

            ops.update(
                {
                    task: optim.Adam(
                        chain(feat_net.parameters(), inf_net.parameters()),
                        lr = self.learn_rate,
                        weight_decay = self.l2)
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

                for X, y, task in self._iter_batches():
                    # pre-processing
                    X, y = self._preprocess(X, y, task)

                    # update the model
                    self.opt.zero_grad()
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
                            'tloss/{}'.format(task), l.item(), self.iters)

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
            sample = dataset[i, task]
            X.append(sample['mel'])
            y.append(sample['label'][None, :])

        y = torch.cat(y, dim=0)

        if task == 'self_':
            X_l = torch.cat([x[0] for x in X], dim=0)
            X_r = torch.cat([x[1] for x in X], dim=0)
            return (X_l, X_r), y

        else:
            X = torch.cat(X, dim=0)
            return X, y

    def _preprocess(self, X, y, task):
        """"""
        if task == 'self_':
            X_l, X_r = X[0], X[1]
            if self.is_gpu:
                X_l, X_r, y = X_l.cuda(), X_r.cuda(), y.cuda()
            # scaling
            X_l, X_r = self.scaler(X_l), self.scaler(X_r)
            return (X_l, X_r), y
        else:
            if self.is_gpu:
                X, y = X.cuda(), y.cuda()
            # scaling
            X = self.scaler(X)
            return X, y

    @background(max_prefetch=5)
    def _iter_batches(self):
        """"""
        for fn in trange(self.n_batches, ncols=80):
            # pick task & indices
            task = np.random.choice(self.tasks)
            idx = np.random.choice(
                len(self.train_dataset.tids), self.batch_sz, False)
            # draw data
            X, y = self._draw_samples(idx, task, 'train')
            yield X, y, task

    def _validate(self, save=True):
        """"""
        self.model.eval()  # toggle to evaluation mode

        # validation
        for task in self.tasks:
            idx = np.random.choice(
                len(self.valid_dataset.tids), self.batch_sz, False)

            # draw data
            X, y = self._draw_samples(idx, task, 'valid')
            X, y = self._preprocess(X, y, task)

            # forward
            y_pred = self.model(X, task)
            l = self.criterion(F.log_softmax(y_pred, dim=1), y)

            # logging
            self.logger.add_scalar('vloss/{}'.format(task),
                                   l.item(), self.iters)
        if save:
            if self.save_every is not None:
                out_fn = ('{}_checkpoint_it{:d}.pth.tar'
                          .format(self.name, self.epoch))
            else:
                out_fn = ('{}_checkpoint.pth.tar'
                          .format(self.name))

            if self.multi_gpu:
                model_state = self.model.module.state_dict()
            else:
                model_state = self.model.state_dict()

            if self.save_location == 'localhost':
                save_checkpoint(
                    {'iters': self.iters,
                     'tasks': self.tasks,
                     'n_outs': self.n_outs,
                     'branch_at': self.branch_at,
                     'scaler_fn': self.scaler_fn,
                     'state_dict': model_state,
                     'optimizer': self.opt.state_dict()},
                    False,  # we don't care best model
                    join(self.run_out_root , out_fn)
                )
            else:
                # TODO: check if the desitnation is accessible and valid?
                # (currently assuming it's always accessible and valid)
                    save_checkpoint(
                        {'iters': self.iters,
                         'tasks': self.tasks,
                         'n_outs': self.n_outs,
                         'branch_at': self.branch_at,
                         'scaler_fn': self.scaler_fn,
                         'state_dict': model_state,
                         'optimizer': self.opt.state_dict()},
                        False,  # we don't care best model
                        out_fn
                    )
                    # send the file to the destination using scp
                    dest = join(self.save_location, self.name) + '/'
                    subprocess.call(['rsync', '-r', out_fn, dest])
                    os.remove(out_fn)

        # toggle to training mode
        self.model.train()


def categorical_crossentropy(input, target):
    """ Categorical Crossentropy Op for probability input

    Args:
        input (torch.tensor): log probability over classes (N x C)
        target (torch.tensor): linear probability over classes (N x C)
    """
    return -(input * target).sum(1).mean(0)
