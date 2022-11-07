import os
from os.path import exists, abspath
from pathlib import Path
from functools import partial
import json
import argparse

# TEMPORARY SOLUTION
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from .data import MSDMelDataset, MSDAudioDataset, MSDMelHDFDataset, ToVariable
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
        self.train_dataset = self._init_dataset(config['dataset'],
                                                split='train')
        if ('valid_tids_fn' in config['dataset'] and
                os.path.exists(config['dataset']['valid_tids_fn'])):
            self.valid_dataset = self._init_dataset(config['dataset'],
                                                    split='valid')

        # override some setup if provided
        lr = config['learn_rate'] if 'learn_rate' in config else cfg.LEARN_RATE
        nepo = config['n_epoches'] if 'n_epoches' in config else cfg.N_EPOCHES
        bsz = config['batch_sz'] if 'batch_sz' in config else cfg.BATCH_SZ
        l2 = config['l2'] if 'l2' in config else cfg.L2
        save_loc = config['save_location'] if 'save_location' in config else cfg.SAVE_LOC
        n_outs = config['n_outs'] if 'n_outs' in config else cfg.N_OUTS
        n_workers = config['num_workers'] if 'num_workers' in config else cfg.N_WORKERS

        if 'n_ch' in config:
            if config['n_ch'] == 'infer':
                n_ch_in = self._infer_n_input_ch()
            elif isinstance(config['n_ch'], int):
                n_ch_in = config['n_ch']
            else:
                raise ValueError('[ERROR] `n_ch` should be an integer or string `infer`')
        else:
            n_ch_in = cfg.N_CH  # default

        if self.config['save_location'] == 'localhost':
            if not exists(abspath(self.config['out_root'])):
                os.mkdir(abspath(self.config['out_root']))

        # setup custom trainer class with global setup
        self.Trainer = partial(
            Trainer,
            train_dataset = self.train_dataset,
            valid_dataset = self.valid_dataset,
            learn_rate = lr, n_epoches = nepo,
            batch_sz = bsz, l2 = l2,
            save_location = save_loc,
            n_outs = n_outs,
            n_ch_in = n_ch_in,
            save_every = config['save_every'],
            scaler_fn = config['scaler_fn'],
            out_root = config['out_root'],
            is_gpu = config['is_gpu'],
            num_workers = n_workers
        )

    def _init_dataset(self, data_config, split='train'):
        """
        split \in {'train', 'valid'}
        """
        assert split in {'train', 'valid'}

        # load datasets
        print(f'Load {split} dataset')
        if data_config['type'] == 'npy':
            dataset = MSDMelDataset(
                data_config['mel_root'], data_config[f'{split}_tids_fn'],
                data_config['label_fn'], on_mem=data_config['on_mem'],
                ignore_intersection=data_config['ignore_label_intersection'],
                transform=ToVariable())

        elif data_config['type'] == 'hdf':
            dataset = MSDMelHDFDataset(
                data_config['hdf_fn'], data_config[f'{split}_tids_fn'],
                data_config['label_fn'],
                ignore_intersection=data_config['ignore_label_intersection'],
                transform=ToVariable())

        elif data_config['type'] == 'audio':
            dataset = MSDAudioDataset(
                data_config['audio_root'], data_config[f'{split}_tids_fn'],
                data_config['tid2path_fn'], data_config['label_fn'],
                ignore_intersection=data_config['ignore_label_intersection'],
                device='cpu',
                transform=ToVariable())

        return dataset

    def _infer_n_input_ch(self):
        """
        """
        sample_ = self.train_dataset[(0, 'self_')]
        x_ = sample_['mel'][0]
        n_ch_in = x_.shape[1]
        return n_ch_in

    def run(self):
        """"""
        for i, setup in enumerate(self.config['experiments']):
            print('Starting experiment: {} ({:d}/{:d})'.format(
                setup['name'], i, len(self.config['experiments'])))
            setup['name'] = '.'.join([self.config['name'], setup['name']])

            # instantiate trainer & run
            trainer = self.Trainer(**setup)
            trainer.fit()


def parse_arguments():
    """
    """
    # setup arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help='configuration (.json) path')
    return parser.parse_args()


def main():
    """
    """
    args = parse_arguments()

    # load config file
    with Path(args.config).open('r') as fp:
        conf = json.load(open(args.config))

    # instantiate experiment & run
    Experiment(conf).run()


if __name__ == "__main__":
    raise SystemExit(main())
