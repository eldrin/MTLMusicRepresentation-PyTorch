import os
from os.path import exists, abspath
from functools import partial

# TEMPORARY SOLUTION
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from .data import MSDMelDataset, ToVariable
from .train import Trainer
from .config import Config as cfg


class Experiment:
    """ Experiment helper """
    def __init__(self, config):
        """
        Args:
            config_fn (str): path to experiment configuration file
        """
        self.config = config
        if self.config['save_location'] == 'localhost':
            if not exists(abspath(self.config['out_root'])):
                os.mkdir(abspath(self.config['out_root']))

        # load datasets
        print('Load training dataset')
        self.train_dataset = MSDMelDataset(
            config['mel_root'], config['train_melpaths_fn'],
            config['label_fn'], on_mem=config['on_mem'],
            transform=ToVariable())

        if ('valid_melpaths_fn' in config and
                os.path.exists(config['valid_melpaths_fn'])):

            print('Load validation dataset')
            self.valid_dataset = MSDMelDataset(
                config['mel_root'], config['valid_melpaths_fn'],
                config['label_fn'], on_mem=config['on_mem'],
                transform=ToVariable())
        else:
            self.valid_dataset = None

        # override some setup if provided
        lr = config['learn_rate'] if 'learn_rate' in config else cfg.LEARN_RATE
        nepo = config['n_epoches'] if 'n_epoches' in config else cfg.N_EPOCHES
        bsz = config['batch_sz'] if 'batch_sz' in config else cfg.BATCH_SZ
        l2 = config['l2'] if 'l2' in config else cfg.L2
        save_loc = config['save_location'] if 'save_location' in config else cfg.SAVE_LOC
        n_outs = config['n_outs'] if 'n_outs' in config else cfg.N_OUTS

        # setup custom trainer class with global setup
        self.Trainer = partial(
            Trainer,
            train_dataset = self.train_dataset,
            valid_dataset = self.valid_dataset,
            learn_rate = lr, n_epoches = nepo,
            batch_sz = bsz, l2 = l2,
            save_location = save_loc,
            n_outs = n_outs,
            save_every = config['save_every'],
            scaler_fn = config['scaler_fn'],
            out_root = config['out_root'],
            is_gpu = config['is_gpu'],
        )

    def run(self):
        """"""
        for i, setup in enumerate(self.config['experiments']):
            print('Starting experiment: {} ({:d}/{:d})'.format(
                setup['name'], i, len(self.config['experiments'])))
            setup['name'] = '.'.join([self.config['name'], setup['name']])

            # instantiate trainer & run
            trainer = self.Trainer(**setup)
            trainer.fit()
