from os.path import join, basename
import pickle as pkl
import six
import random

# TEMPORARY SOLUTION
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
import h5py

import torch
import torchaudio
from torch.utils.data import Dataset
from torch.autograd import Variable
from tqdm import tqdm

from .utils import load_pickle


BINARY_CLASS_LABEL = np.array([[1., 0.], [0., 1.]])


def _negative_sampling(pos_i, n_items):
    """
    """
    neg_idx = random.randint(0, n_items - 1)
    while neg_idx == pos_i:
        neg_idx = random.randint(0, n_items - 1)
    return neg_idx


class MSDMelDataset(Dataset):
    """ mel-spectrogram dataset of the subset of Million Song Dataset """
    def __init__(self, mel_root, tids_fn, label_fn, crop_len=216,
                 on_mem=True, transform=None, ignore_intersection=False):
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
        with open(tids_fn) as f:
            melpaths = [join(mel_root, line.replace('\n', '') + '.npy')
                        for line in f.readlines()]

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
        self.ignore_intersection = ignore_intersection
        if self.ignore_intersection:
            self.tids = set(tids)
        else:
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
                # y_self = np.array([0., 1.])
                y_self = BINARY_CLASS_LABEL[1]
            else:
                # select random negative sample
                neg_idx = self._negative_sampling(idx)
                X_r = self._load_mel(neg_idx)
                X_r = self._crop_mel(X_r)
                # y_self = np.array([1., 0.])
                y_self = BINARY_CLASS_LABEL[0]

            sample = {'mel': (X_l, X_r), 'label': y_self, 'task': task}
        else:
            # retrive (X_t, z_t)
            sample = {'mel': x_,
                      'label': self.labels[task][self.i_tids[idx]],
                      'task': task}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _load_mel(self, idx):
        """
        output feature (mel spectrogram) has (n_ch, n_seq, dim)
        """
        tid = self.i_tids[idx]
        if self.on_mem:
            return self.mels[tid]
        else:
            return np.load(self.melpaths[tid], mmap_mode='r')

    def _crop_mel(self, mel):
        """
        expects the input has the shape of (n_ch, n_seq, dim)
        """
        if mel.shape[1] > self.crop_len:
            st = np.random.randint(mel.shape[1] - self.crop_len)
            out = mel[:, st: st+self.crop_len]
        elif mel.shape[1] == self.crop_len:
            out = mel
        else:  # mel length < crop_len
            # zero-pad first
            n_ch = mel.shape[0]
            dim = mel.shape[-1]
            mel_ = np.zeros((n_ch, self.crop_len, dim), dtype=np.float32)
            st_ = np.random.randint(self.crop_len - mel.shape[1])
            mel_[:, st_:st_ + mel.shape[1]] = mel
            out = mel_
        return np.array(out)

    def _negative_sampling(self, tid_i):
        """"""
        return _negative_sampling(tid_i, len(self.i_tids))


class MSDAudioDataset(Dataset):
    """ mel-spectrogram dataset of the subset of Million Song Dataset """
    def __init__(self, audio_root, tids_fn, tid2path_fn, label_fn,
                 sample_rate=22050, n_fft=2048, hop_sz=512, crop_len=216,
                 transform=None, device='cpu', ignore_intersection=False):
        """
        Args:
            audio_root (str): path to the text file containing
                              candidate mel spectrograms (.npy)
            tids_fn (str): filename that contains the list of MSD tids to be loaded
            tid2path_fn (str): filename contains the map between the tid and path
            label_fn (str): path to label file (a dictionary contains all labels)
            transform (callable, optional): optional transform to be applied
        """
        self.audio_root = audio_root
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_sz = hop_sz
        self.crop_len = crop_len
        self.transform = transform
        self.crop_len = crop_len
        self.device = device

        # initiating melspec object
        # some parameters are fixed to match the librosa
        # (for experimental consistency reason)
        self._melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate = self.sample_rate,
            n_fft = self.n_fft,
            hop_length = self.hop_sz,
            pad_mode = 'reflect',
            norm = 'slaney',
            mel_scale = 'slaney'
        ).to(device)
        self._power2db = torchaudio.transforms.AmplitudeToDB(
            'power', top_db = 80.0
        ).to(device)

        ##################
        # load data
        ##################
        with open(tid2path_fn, 'rb') as fp:
            tid2path = pkl.load(fp)

        with open(tids_fn) as f:
            tids = []
            self.audiopaths = {}
            for line in f.readlines():
                tid = line.replace('\n', '')
                tids.append(tid)
                self.audiopaths[tid] = join(self.audio_root, tid2path[tid])

        # load label data
        # : dictionary of dictionary (~700mb)
        # : each sub-dict contains tid: item_factor
        # : no self (should be generated dynamically)
        self.labels = load_pickle(label_fn)

        # get the intersection of tids between mel and labels
        self.ignore_intersection = ignore_intersection
        if self.ignore_intersection:
            self.i_tids = list(set(tids))
        else:
            self.i_tids = list(set(tids).intersection(
                set(six.next(six.itervalues(self.labels)).keys())
            ))

        # make hash for tids
        self.tids_i = {v:k for k, v in enumerate(self.i_tids)}

    def __len__(self):
        """"""
        return len(self.i_tids)

    def __getitem__(self, idx_task):
        """"""
        # unpack
        idx, task = idx_task

        # retrieve mel spec
        x = self._load_audio(idx)
        x = self._compute_feature(x)

        # random cropping
        x_ = self._crop_mel(x)

        if task == 'self_':
            # generate ((X_l, X_r), y_self)
            X_l = x_
            y_self_ = np.random.randint(2)
            if y_self_ == 1:
                # select 2 chunks from same song
                X_r = self._crop_mel(x)
                # y_self = np.array([0., 1.])
                y_self = BINARY_CLASS_LABEL[1]
            else:
                # select random negative sample
                neg_idx = self._negative_sampling(idx)
                x = self._load_audio(neg_idx)
                X_r = self._compute_feature(x)
                X_r = self._crop_mel(X_r)
                # y_self = np.array([1., 0.])
                y_self = BINARY_CLASS_LABEL[0]

            sample = {'mel': (X_l, X_r), 'label': y_self, 'task': task}
        else:
            # retrive (X_t, z_t)
            sample = {'mel': x_,
                      'label': self.labels[task][self.i_tids[idx]],
                      'task': task}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _load_audio(self, idx):
        """
        outputs:
            torch.Tensor: signal with shape of (n_ch, n_len_sig)
        """
        tid = self.i_tids[idx]
        y, sr = torchaudio.load(self.audiopaths[tid])
        if sr != self.sample_rate:
            y = torchaudio.functional.resample(y, sr, self.sample_rate)
        return y

    def _compute_feature(self, x):
        """
        outputs (n_ch, n_mels, time)
        """
        return (
            self._power2db(self._melspec(x.to(self.device)))
            .permute(0, 2, 1)
            .detach()
            .cpu()
            .numpy()
        )

    def _crop_mel(self, mel):
        """
        expects the input has the shape of (n_ch, n_seq, dim)
        """
        if mel.shape[1] > self.crop_len:
            st = np.random.randint(mel.shape[1] - self.crop_len)
            out = mel[:, st: st+self.crop_len]
        elif mel.shape[1] == self.crop_len:
            out = mel
        else:  # mel length < crop_len
            # zero-pad first
            n_ch = mel.shape[0]
            dim = mel.shape[-1]
            mel_ = np.zeros((n_ch, self.crop_len, dim), dtype=np.float32)
            st_ = np.random.randint(self.crop_len - mel.shape[1])
            mel_[:, st_:st_ + mel.shape[1]] = mel
            out = mel_
        return np.array(out)

    def _negative_sampling(self, tid_i):
        """"""
        neg_idx = np.random.choice(len(self.i_tids))
        while neg_idx == tid_i:
            neg_idx = np.random.choice(len(self.i_tids))
        return neg_idx


class MSDMelHDFDataset(Dataset):
    """ mel-spectrogram dataset of the subset of Million Song Dataset """
    def __init__(self, h5_fn, tids_fn, label_fn, crop_len=216, transform=None,
                 ignore_intersection=False, on_mem=False):
        """
        Args:
            h5_py (str): HDF file that contains mel spectrogram data (.h5)
            label_fn (str): path to label file (a dictionary contains all labels)
            transform (callable, optional): optional transform to be applied
        """
        self.h5_fn = h5_fn
        with open(tids_fn) as f:
            self.tids = [line.replace('\n', '') for line in f]

        # load label data
        # : dictionary of dictionary (~700mb)
        # : each sub-dict contains tid: item_factor
        # : no self (should be generated dynamically)
        self.labels = load_pickle(label_fn)

        # get the intersection of tids between mel and labels
        self.ignore_intersection = ignore_intersection
        if self.ignore_intersection:
            self.tids = set(self.tids)
        else:
            self.tids = list(set(self.tids).intersection(
                set(six.next(six.itervalues(self.labels)).keys())
            ))

        # make hash for tids
        self.i_tids = dict(enumerate(self.tids))
        self.tids_i = {tid:i for i, tid in self.i_tids.items()}

        # we just open it when any operation is happening
        with h5py.File(self.h5_fn, 'r') as hf:
            self.tid_to_hdf_idx = {
                tid: i for i, tid in enumerate(hf['ids'][:].astype('U'))
            }

        self.transform = transform
        self.crop_len = crop_len

        # TODO: in case on-the-core option, it loads everything
        self.on_mem = on_mem

    def __len__(self):
        """"""
        return len(self.i_tids)

    def __getitem__(self, idx_task):
        """"""
        # unpack
        idx, task = idx_task

        # retrieve mel spec and crop
        x_ = self._load_and_crop_mel(idx)

        if task == 'self_':
            # generate ((X_l, X_r), y_self)
            X_l = x_
            y_self_ = np.random.randint(2)
            if y_self_ == 1:
                # select 2 chunks from same song
                X_r = self._load_and_crop_mel(idx)
                y_self = BINARY_CLASS_LABEL[1]
            else:
                # select random negative sample
                neg_idx = self._negative_sampling(idx)
                X_r = self._load_and_crop_mel(neg_idx)
                y_self = BINARY_CLASS_LABEL[0]

            sample = {'mel': (X_l, X_r), 'label': y_self, 'task': task}
        else:
            # retrive (X_t, z_t)
            sample = {'mel': x_,
                      'label': self.labels[task][self.i_tids[idx]],
                      'task': task}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _load_and_crop_mel(self, idx):
        """
        """
        tid = self.i_tids[idx]
        i = self.tid_to_hdf_idx[tid]

        with h5py.File(self.h5_fn, 'r') as hf:
            i0, i1 = hf['indptr'][i], hf['indptr'][i + 1]

            length = i1 - i0
            if length > self.crop_len:
                st = i0 + np.random.randint(length - self.crop_len)
                out = hf['data'][:, st:st + self.crop_len]
            elif length == self.crop_len:
                out = hf['data'][:, i0:i1]
            else:  # mel length < crop_len
                # zero-pad first
                n_ch, _, dim = hf['data'].shape
                mel_ = np.zeros((n_ch, self.crop_len, dim), dtype=np.float32)
                st_ = np.random.randint(self.crop_len - length)
                mel_[:, st_:st_ + length] = hf['data'][:, i0:i1]
                out = mel_

        return np.array(out)

    def _negative_sampling(self, tid_i):
        """"""
        return _negative_sampling(tid_i, len(self.i_tids))


class MTLBatchSampler:
    def __init__(self, n_samples, tasks,
                 task_weight=None,
                 batch_size=48,
                 drop_last=False,
                 shuffle=True):
        """
        """
        self.n_samples = n_samples
        self.tasks = tasks
        self.task_weight = task_weight
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        # compute number of batches given sample size and `drop_last` indicator
        self.n_batches = n_samples // batch_size
        if not self.drop_last:
            self.n_batches += int(n_samples % batch_size != 0)

        # TODO: check validity of task weight (should be a iterable of float)

    def __len__(self):
        """
        """
        return self.n_batches

    def __iter__(self):
        """
        """
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        for i in range(self.n_batches):
            # choose task
            if self.task_weight is None:
                t = self.tasks[np.random.randint(len(self.tasks))]
            else:
                t = np.random.choice(self.tasks, p=self.task_weight)
            idx = indices[i * self.batch_size: (i+1) * self.batch_size]
            idx = [(i, t) for i in idx]
            yield idx


class ToVariable(object):
    """ Convert ndarrays in sample in Variables. """
    def __call__(self, sample):
        """"""
        mel, label, task = sample['mel'], sample['label'], sample['task']
        if isinstance(mel, tuple):
            mel = (
                Variable(torch.from_numpy(mel[0]).float()),
                Variable(torch.from_numpy(mel[1]).float())
            )
        else:
            mel = Variable(torch.from_numpy(mel).float())

        return {'mel': mel,
                'label': Variable(torch.from_numpy(label).float()),
                'task': task}

