from typing import Optional
from pathlib import Path
from os.path import join, dirname, basename
import shutil
import logging
import pickle as pkl
from multiprocessing import Pool
import argparse

# TEMPORARY SOLUTION
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
import torch
import librosa
import h5py
from tqdm import tqdm

from .config import Config as cfg


H5PY_STR = h5py.special_dtype(vlen=str)
DEFAULT_RAND_SEED = 2022

logging.basicConfig()
logger = logging.getLogger(__name__)


def load_pickle(fn):
    """ python2 -> python3 loadable pickle loader

    Args:
        fn (str): path to pickle file

    ref: https://blog.modest-destiny.com/posts/python-2-and-3-compatible-pickle-save-and-load/
    """
    try:
        with open(fn, 'rb') as f:
            data = pkl.load(f)
    except UnicodeDecodeError as e:
        with open(fn, 'rb') as f:
            data = pkl.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data', fn, ':', e)
        raise
    return data


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """"""
    torch.save(state, filename)
    if is_best:
        new_basename = basename(filename)
        new_basename = new_basename.replace('checkpoint', 'model_best')
        new_fn = join(dirname(filename), new_basename)
        shutil.copyfile(filename, new_fn)


def extract_mel(fn, mono=False, verbose=False):
    """"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = librosa.load(fn, sr=cfg.SR, mono=mono)

    if not mono:
        if y.ndim == 1:
            if verbose:
                print('[Warning] "{}" only has 1 channel. '.format(fn) +
                      'making psuedo 2-channels...')
            y = np.vstack([y, y])
    else:
        y = y[None]

    Y = librosa.power_to_db(np.array([
        librosa.feature.melspectrogram(
            y=ch, sr=sr, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LEN).T
        for ch in y
    ])).astype(np.float32)  # (2, t, 128) or (1, t, 128) if mono

    return Y  # (2, t, 128) or (1, t, 128) if mono


def parmap(func, iterable, n_workers=2, verbose=False):
    """ Simple Implementation for Parallel Map """

    if n_workers == 1:
        if verbose:
            iterable = tqdm(iterable, total=len(iterable), ncols=80)
        return list(map(func, iterable))
    else:
        with Pool(processes=n_workers) as p:
            if verbose:
                with tqdm(total=len(iterable), ncols=80) as pbar:
                    output = []
                    for o in p.imap_unordered(func, iterable):
                        output.append(o)
                        pbar.update()
                return output
            else:
                return p.imap_unordered(func, iterable)


def __ext_mel(
    input_args: tuple[str, str, bool, bool],
    file_id: Optional[str] = None,
) -> None:
    """ simple helper function to compute the melspec parallelly.

    Args:
        input_args: tuple of filenamd, output path, overwrite flag, and mono
                    floag. it'll be parsed into four input arguments.
        file_id: if given, it is used as the stem of outputing filename

    Raises:
        ValueError: if the stem of name of the input file is shorter than
                    2 characters.
    """
    # parse input argument
    fn, out_path, overwrite, mono = input_args

    stem = file_id if file_id is not None else Path(fn).stem

    if len(stem) < 2:
        ValueError('[ERROR] the length of the stem of input filename '
                   ',or `file_id`, if given, '
                   'must be longer than 2 characters')

    # knit output filename and directory name
    out_dir = Path(out_path) / stem[0] / stem[1]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fn = out_dir / f"{stem}.npy"

    if out_fn.exists() and not overwrite:
        return

    try:
        Y = extract_mel(fn, mono=mono)
        np.save(out_fn.as_posix(), Y)
    except Exception as e:
        print(stem, e)


def _get_split_bounds(
    cands: list[str],
    train_ratio: float = 0.8,
    test_ratio: float = 0.5
) -> tuple[int, int]:
    """ compute split bounds

    Args:
        cands: list of file ids used as the total pool to be split
        train_ratio: determines the ratio of training samples.
        test_ratio: ratio of test samples "WITHIN THE NON-TRAINING"
                    "SAMPLES". For instance if the `train-ratio` is
                    set as 0.5 and `test-ratio` is set as 0.5, the
                    ratio of test samples for the entire dataset
                    becomes 0.25. Similarly, it will be 0.1 if
                    the `train-ratio` is set as 0.8 and `test-ratio`
                    is set as same (0.5).
    Returns:
        split bounds
    """
    n_samples = len(cands)

    # check healthiness of the input
    # TODO: in theory, up to 3 samples we can do the operation. check more thoroughly for future.
    if n_samples < 100:
        raise ValueError('[ERROR] the total number of samples should be larger '
                         'than 100!')

    if test_ratio <= 0 or test_ratio >= 1:
        raise ValueError('[ERROR] the test ratio should be larger than 0 '
                         'and smaller than 1. In other words, it should '
                         'in (0, 1)!')

    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError('[ERROR] the train ratio should be larger than 0 '
                         'and smaller than 1. In other words, it should '
                         'in (0, 1)!')

    train_bound = int(train_ratio * n_samples)
    valid_bound = (
        int((n_samples - train_bound) * (1. - test_ratio)) + train_bound
    )
    return train_bound, valid_bound


def make_hdf(
    candidates: str,
    out_path: str,
    out_prefix: str = 'mel_hdf',
    n_total_frames: Optional[int] = None,
    gen_split: bool = True,
    random_seed: int = 2022,
    train_ratio: float = 0.8,
    test_ratio: float = 0.5,
    drop_chan_dim: bool = False,
    verbose: bool = False
) -> None:
    """ convert individual mel spec files (.npy) into single HDF db file.

    Args:
        candidates: text file containing target candidate filenames
        out_path: output root where all the mel-spec saved
        out_prefix: prefix to be used in front of the output filename
        n_total_frames: the number of total frames. If not given, it is computed
                        from the files given
        gen_split: generate the split file contains ids for {train, valid, test}
                   sets. The split files are stored in text file in the out_path
        random_seed: the default random seed to be used in generating the random
                     split of train/valid/test. If not given the default seed
                     (i.e., 2022) is used.
        train_ratio: determines the ratio of training samples.
        test_ratio: ratio of test samples "WITHIN THE NON-TRAINING SAMPLES".
                    For instance if the `train_ratio` is set as 0.5 and
                    `test_ratio` is set as 0.5, the ratio of test samples
                    for the entire dataset becomes 0.25. Similarly, it will
                    be 0.1 if the `train_ratio` is set as 0.8 and `test_ratio`
                    is set as same (0.5).
        drop_chan_dim: if set True, the procedure remove the preceding channel
                       axis by: 1) drop if the audio originally mono, or 2) take
                       the average on the axis if there are more than two
                       channels.
        verbose: set verbosity
    """
    # set random seed
    rng = np.random.default_rng(random_seed)

    # set paths
    in_fn = Path(candidates)
    outroot = Path(out_path)
    outroot.mkdir(parents=True, exist_ok=True)

    # knit output filename
    out_fn = outroot / f'{out_prefix}.h5'

    # read candidates
    cands = []
    tids = []
    with in_fn.open('r') as fp:
        for line in fp:
            cands.append(line.replace('\n', ''))
            tids.append(Path(line).stem)

    # check if we also make the split files
    if gen_split:

        # split path
        split_outroot = outroot / 'splits'
        split_outroot.mkdir(parents=True, exist_ok=True)

        # then we generate split files
        split_names = ['train', 'valid', 'test']
        split_fns = {
            split: split_outroot / f'{split}.txt'
            for split in split_names
        }
        split_bounds = _get_split_bounds(cands,
                                         train_ratio,
                                         test_ratio)
        rnd_idx = rng.permutation(len(cands))

        splits = np.split(rnd_idx, split_bounds)
        for split, split_name in zip(splits, split_names):
            with Path(split_fns[split_name]).open('w') as fp:
                for i in split:
                    tid = Path(cands[i]).stem
                    fp.write(f'{tid}\n')

    # build HDF data
    n_ch, _, dim = np.load(Path(cands[0])).shape
    with h5py.File(out_fn, 'w') as hf:

        indptr = [0]
        if n_total_frames is None:
            with tqdm(total=len(cands), ncols=80, disable=not verbose) as prog:
                total_len = 0
                for fn in cands:
                    x = np.load(fn, mmap_mode='r')
                    n_ch, length, dim = x.shape

                    indptr.append(indptr[-1] + length)
                    total_len += length
                    prog.update()
        else:
            total_len = n_total_frames

        if drop_chan_dim:
            data = hf.create_dataset('data', shape=(total_len, dim), dtype=np.float32)
        else:
            data = hf.create_dataset('data', shape=(n_ch, total_len, dim), dtype=np.float32)

        with tqdm(total=len(cands), ncols=80, disable=not verbose) as prog:
            for i, fn in enumerate(cands):
                x = np.load(fn)

                if n_total_frames is None:
                    i0, i1 = indptr[i], indptr[i+1]
                else:
                    n_ch, length, dim = x.shape
                    i0, i1 = indptr[-1], indptr[-1] + length
                    indptr.append(i1)

                if drop_chan_dim:
                    data[i0:i1] = x.mean(0) if x.shape[0] > 1 else x[0]
                else:
                    data[:, i0:i1] = x

                prog.update()

        # write data
        hf.create_dataset('indptr', data=np.array(indptr), dtype='i4')
        hf.create_dataset('ids', data=np.array(tids, dtype=H5PY_STR))


def parse_args() -> argparse.Namespace:
    """
    """
    parser = argparse.ArgumentParser(
        prog='mtlutils',
        description=(
            "It contains a few helpful utilities used for preparing "
            "data for MTL music representation procedure."
        )
    )
    subparsers = parser.add_subparsers(title="command",
                                       dest="command",
                                       help="sub-command help")
    subparsers.required = True

    base_subparser = argparse.ArgumentParser(add_help=False)
    base_subparser.add_argument("-p", "--path", type=str, default="./",
                                help="path where the output stored")
    base_subparser.add_argument('--verbose', default=True,
                                action=argparse.BooleanOptionalAction,
                                help="set verbosity")

    # `melspec` sub command ====================================================
    melspec = subparsers.add_parser(
        'melspec', parents=[base_subparser],
        help='compute melspectrogram from audio files'
    )

    melspec.add_argument("filelist",
                         help=("a text file containing a list of audio files "
                               "per each line from which we compute the "
                               "melspectrograms per song."))
    melspec.add_argument("--workers", type=int, default=1,
                         help='number of workers doing the job')
    melspec.add_argument('--overwrite', default=True,
                         action=argparse.BooleanOptionalAction,
                         help="set if the melspectrum is overwritten if exists.")
    melspec.add_argument('--mono', default=False,
                         action=argparse.BooleanOptionalAction,
                         help=("set the procedure extract mel spectrum from "
                               "mono channel audio. if the original audio is "
                               "stereo it's forcefully converted into mono "
                               "mono channel audio."))

    # `makehdf` sub command ====================================================
    mkhdf = subparsers.add_parser(
        'mkhdf', parents=[base_subparser],
        help='convert mel spectrum files into a single HDF dataset file'
    )

    mkhdf.add_argument("filelist",
                       help=("a text file containing a list of numpy files "
                             "per each line from which we compute the "
                             "melspectrograms per song."))
    mkhdf.add_argument("--out-prefix", type=str, default='mel_hdf',
                       help='prefix to be used in front of the output filename')
    mkhdf.add_argument("--n-total-frames", type=int, default=None,
                       help=('the number of total frames. If not given, it is computed '
                               'from the files given'))
    mkhdf.add_argument('--gen-split', default=True,
                       action=argparse.BooleanOptionalAction,
                       help=("generate the split file contains ids for "
                             "{train, valid, test} sets. The split files "
                             "are stored in text file in the `outroot`"))
    mkhdf.add_argument("--random-seed", type=int, default=DEFAULT_RAND_SEED,
                       help=("the default random seed to be used in generating "
                             "the random split of train/valid/test. If not "
                             "given the default seed (i.e., 2022) is used."))
    mkhdf.add_argument('--train-ratio', type=float, default=0.8,
                       help='determines the ratio of training samples.')
    mkhdf.add_argument('--test-ratio', type=float, default=0.5,
                       help=('ratio of test samples "WITHIN THE NON-TRAINING '
                             'SAMPLES". For instance if the `train-ratio` is '
                             'set as 0.5 and `test-ratio` is set as 0.5, the '
                             'ratio of test samples for the entire dataset '
                             'becomes 0.25. Similarly, it will be 0.1 if '
                             'the `train-ratio` is set as 0.8 and `test-ratio` '
                             'is set as same (0.5).'))
    mkhdf.add_argument('--drop-chan-dim', default=False,
                       action=argparse.BooleanOptionalAction,
                       help=("if set True, the procedure remove the preceding "
                             "channel axis by: 1) drop if the audio originally "
                             "mono, or 2) take the average on the axis if "
                             "there are more than two channels."))

    return parser.parse_args()


def main():
    """
    """
    args = parse_args()

    if args.verbose:
        logging.getLogger(__name__).setLevel(logging.INFO)

    # generate output directory if not exists
    outroot = Path(args.path)
    outroot.mkdir(parents=True, exist_ok=True)

    if args.command == 'melspec':

        # build input iterable
        with Path(args.filelist).open('r') as fp:
            cands = [
                (line.replace('\n', ''),
                 args.path,
                 args.overwrite,
                 args.mono)
                for line in fp
            ]

        # process
        parmap(__ext_mel, cands, n_workers=args.workers, verbose=args.verbose)

    elif args.command == 'mkhdf':

        make_hdf(
            args.filelist,
            args.path,
            args.out_prefix,
            args.n_total_frames,
            args.gen_split,
            args.random_seed,
            args.train_ratio,
            args.test_ratio,
            args.drop_chan_dim,
            args.verbose
        )

    else:
        ValueError('[ERROR] only `melspec` subcommand is '
                   'available!')


if __name__ == "__main__":
    raise SystemExit(main())
