from pathlib import Path
import argparse

import h5py
import numpy as np
from tqdm import tqdm

H5PY_STR = h5py.special_dtype(vlen=str)
DEFAULT_RAND_SEED = 2022


def parse_arguments() -> argparse.Namespace:
    """
    """
    # setup arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates",
                        help='text file containing target candidate filenames')
    parser.add_argument("outroot",
                        help='output root where all the mel-spec saved')
    parser.add_argument("--out-prefix", type=str, default='mel_hdf',
                        help='prefix to be used in front of the output filename')
    parser.add_argument("--n-total-frames", type=int, default=None,
                        help=('the number of total frames. If not given, it is computed '
                              'from the files given'))
    parser.add_argument('--gen-split', default=True,
                        action=argparse.BooleanOptionalAction,
                        help=("generate the split file contains ids for "
                              "{train, valid, test} sets. The split files "
                              "are stored in text file in the `outroot`"))
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RAND_SEED,
                        help=("the default random seed to be used in generating "
                              "the random split of train/valid/test. If not "
                              "given the default seed (i.e., 2022) is used."))
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='determines the ratio of training samples.')
    parser.add_argument('--test-ratio', type=float, default=0.5,
                        help=('ratio of test samples "WITHIN THE NON-TRAINING"'
                              '"SAMPLES". For instance if the `train-ratio` is '
                              'set as 0.5 and `test-ratio` is set as 0.5, the '
                              'ratio of test samples for the entire dataset '
                              'becomes 0.25. Similarly, it will be 0.1 if '
                              'the `train-ratio` is set as 0.8 and `test-ratio` '
                              'is set as same (0.5).'))
    parser.add_argument('--verbose', default=True,
                        action=argparse.BooleanOptionalAction,
                        help="set verbosity")
    return parser.parse_args()


def _get_split_bounds(
    cands: list[str],
    train_ratio: float = 0.8,
    test_ratio: float = 0.5
) -> tuple[int, int]:
    """
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


def main():
    """
    """
    args = parse_arguments()

    # set random seed
    rng = np.random.default_rng(args.random_seed)

    # set paths
    in_fn = Path(args.candidates)
    outroot = Path(args.outroot)
    outroot.mkdir(parents=True, exist_ok=True)

    # knit output filename
    out_fn = outroot / f'{args.out_prefix}.h5'

    # read candidates
    cands = []
    tids = []
    with in_fn.open('r') as fp:
        for line in fp:
            cands.append(line.replace('\n', ''))
            tids.append(Path(line).stem)

    # check if we also make the split files
    if args.gen_split:

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
                                         args.train_ratio,
                                         args.test_ratio)
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
        if args.n_total_frames is None:
            with tqdm(total=len(cands), ncols=80, disable=not args.verbose) as prog:
                total_len = 0
                for fn in cands:
                    x = np.load(fn, mmap_mode='r')
                    n_ch, length, dim = x.shape

                    indptr.append(indptr[-1] + length)
                    total_len += length
                    prog.update()
        else:
            total_len = args.n_total_frames

        data = hf.create_dataset('data', shape=(n_ch, total_len, dim), dtype=np.float32)
        with tqdm(total=len(cands), ncols=80, disable=not args.verbose) as prog:
            for i, fn in enumerate(cands):
                x = np.load(fn)

                if args.n_total_frames is None:
                    i0, i1 = indptr[i], indptr[i+1]
                else:
                    n_ch, length, dim = x.shape
                    i0, i1 = indptr[-1], indptr[-1] + length
                    indptr.append(i1)

                data[:, i0:i1] = x
                prog.update()

        # write data
        hf.create_dataset('indptr', data=np.array(indptr), dtype='i4')
        hf.create_dataset('ids', data=np.array(tids, dtype=H5PY_STR))


if __name__ == '__main__':
    main()
