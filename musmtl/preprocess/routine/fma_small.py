from pathlib import Path
import argparse
import logging
import json
import zipfile
import pickle as pkl

import numpy as np
from scipy import sparse as sp
import pandas as pd

from . import (InteractionData,
               NumericData,
               ProcessedData,
               fit_factors)
from ...utils import extract_mel

# TODO: this is certainly a hack. find better way
from . import __name__ as parent_name
from .utils import fetch_urls


PROG_NAME = 'fmasmtlprep'


logging.basicConfig()
logger = logging.getLogger(PROG_NAME)


FMA_ROOT_URL = "https://os.unil.cloud.switch.ch/fma"
URLS = {
    'fma_small': f'{FMA_ROOT_URL}/fma_small.zip',
    'fma_metadata': f'{FMA_ROOT_URL}/fma_metadata.zip'
}
CHECKSUMS = {
    'fma_small': 'ade154f733639d52e35e32f5593efe5be76c6d70',
    'fma_metadata': 'f0df49ffe5f2a6008d7dc83c6915b31835dfe733'
}


def load_metadata(
    fn: str
) -> dict[str, pd.DataFrame]:
    """ load the FMA metadata

    Args:
        fn: filename of the uncompressed metadata file (.zip)

    Returns:
        :obj:`~pandas.Dataframe` containing the metadata.
    """
    # load the metadata file
    metadata = {}

    with zipfile.ZipFile(fn) as zfp:
        with zfp.open('fma_metadata/tracks.csv') as fp:
            metadata['tracks'] = pd.read_csv(fp, header=[0, 1], index_col=0)

        with zfp.open('fma_metadata/genres.csv') as fp:
            metadata['genres'] = pd.read_csv(fp)

    return metadata


def load_tag_data(
    fn: str
) -> InteractionData:
    """ get the tag (genres_all) in form of :obj:`~musmtl.preprocess.routine.InteractionData`

    It process the dataframe and then returns the song-tag assignment in
    :obj:`~musmtl.preprocess.routine.InteractionData` format.

    Args:
        fn: filename of the uncompressed metadata file (.zip)

    Returns:
        FMA song - tag (genres_all) assignment interaction data
    """
    # load metadata
    metadata = load_metadata(fn)

    # get genreid -> idx map
    genremap = {
        v:k for k, v
        in metadata['genres']['genre_id'].items()
    }
    genres_list = metadata['genres']['title'].values.tolist()

    # get the song - tag (genres_all) assignment
    genres_all = (
        metadata['tracks'][('track', 'genres_all')]
        .apply(eval)
        .apply(lambda genres: [genremap[g] for g in genres])
    )

    indices = np.hstack(genres_all.values).astype(int)
    data = np.ones_like(indices, dtype=np.float32)
    indptr = np.r_[0, np.cumsum(genres_all.apply(len))]

    # build matrix
    mat = sp.csr_matrix(
        (data, indices, indptr),
        shape=(genres_all.shape[0], len(genres_list))
    )

    return InteractionData(
        mat.tocoo(),
        metadata['tracks'].index.astype(str).str.zfill(6).tolist(),
        genres_list
    )


LOAD_FUNCS = {
    'tag': load_tag_data
}

REQUIRED_DATA = {
    'tag': ('fma_metadata',)
}

POST_PROCESSING_HOOK = {}


def parse_args() -> argparse.Namespace:
    """
    """
    parser = argparse.ArgumentParser(
        prog=PROG_NAME,
        description=(
            "It helps to preprocess the learning target factors "
            "of the `MTLMusicRepresentation`."
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

    # `fetch` sub command =====================================================
    fetch = subparsers.add_parser('fetch',
                                  parents=[base_subparser],
                                  help='fetch relevant data files to the disk')

    # `fitfactor` sub command =================================================
    fitfactor = subparsers.add_parser('fitfactor',
                                      parents=[base_subparser],
                                      help='fit factors for each aspect')
    fitfactor.add_argument("config",
                           help=("configuration file (.json) specifies the "
                                 "path to the each data file fetched from "
                                 "`fetch` sub routine"))
    fitfactor.add_argument("--n-components", type=int, default=50,
                           help=("the number of components of latent "
                                 "component model."))
    fitfactor.add_argument("--n-iters", type=int, default=100,
                           help="the number of iterations for the model fit")

    return parser.parse_args()


def main():
    """
    """
    args = parse_args()

    if args.verbose:
        logging.getLogger(parent_name).setLevel(logging.INFO)

    if args.command == 'fetch':

        # fetch the files
        out_fns = fetch_urls(args.path, URLS, CHECKSUMS,
                             hash_type='sha1',
                             verbose=args.verbose)

        # at the end of the sub routine, we save the json file
        # containing the file keyword and path
        with (Path(args.path) / 'file_paths.json').open('w') as fp:
            json.dump(out_fns, fp)

    elif args.command == 'fitfactor':

        factor_output = fit_factors(
            args.config,
            LOAD_FUNCS,
            REQUIRED_DATA,
            POST_PROCESSING_HOOK,
        )

        # save the output
        with (Path(args.path) / 'target_factors.pkl').open('wb') as fp:
            pkl.dump(factor_output, fp)

    else:
        ValueError('[ERROR] only `fetch` and `fitfactor` subcommands are '
                   'available!')


if __name__ == "__main__":
    main()
