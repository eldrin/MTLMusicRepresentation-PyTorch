import os
from os.path import exists, abspath
from functools import partial

# TEMPORARY SOLUTION
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from .data import MSDMelDataset, ToVariable
from .train import Trainer


class Experiment:
    """ Experiment helper """
    def __init__(self, config):
        """
        Args:
            config_fn (str): path to experiment configuration file
        """
        self.config = config
        if not exists(abspath(self.config['out_root'])):
            os.mkdir(abspath(self.config['out_root']))
        
        # load datasets
        print('Load training dataset')
        self.train_dataset = MSDMelDataset(
            self.config['train_melpaths_fn'], self.config['label_fn'],
            on_mem=self.config['on_mem'], transform=ToVariable())
        
        if ('valid_melpaths_fn' in self.config and 
                os.path.exists(self.config['valid_melpaths_fn'])):
            
            print('Load validation dataset')
            self.valid_dataset = MSDMelDataset(
                self.config['valid_melpaths_fn'], self.config['label_fn'],
                on_mem=self.config['on_mem'], transform=ToVariable())
        else:
            self.valid_dataset = None
        
        # setup custom trainer class with global setup
        self.Trainer = partial(
            Trainer,
            train_dataset = self.train_dataset,
            valid_dataset = self.valid_dataset,
            save_every = self.config['save_every'],
            scaler_fn = self.config['scaler_fn'],
            out_root = self.config['out_root'],
            is_gpu = self.config['is_gpu'],
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