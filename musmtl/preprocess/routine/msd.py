from pathlib import Path
import argparse
import logging
import sqlite3
import zipfile
import json
import pickle as pkl

import numpy as np
from scipy import sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer

from . import (InteractionData,
               NumericData,
               ProcessedData,
               fit_factors)

# TODO: this is certainly a hack. find better way
from . import __name__ as parent_name
from .utils import fetch_urls


PROG_NAME = "msdmtlprep"


logging.basicConfig()
logger = logging.getLogger(PROG_NAME)


# Currently 'bpm' and 'cdr' is not supported by this automatic routine
# so unfortunately it's not possible to run this to get the all the
# learning targets. But we provide the pre-computed learning target (factors)
# for all the aspects as part of this project.
#
# in theory, with files corresponding to 'bpm' and 'cdr' hosted somewhere
# the routine works for all the aspects automatically.
MSD_ROOT_URL = "http://millionsongdataset.com/sites/default/files"
URLS = {
    'echonest': f"{MSD_ROOT_URL}/challenge/train_triplets.txt.zip",
    'lastfm': f"{MSD_ROOT_URL}/lastfm/lastfm_tags.db",
    'mxm': f"{MSD_ROOT_URL}/AdditionalFiles/mxm_dataset.db",
    'metadata': f"{MSD_ROOT_URL}/AdditionalFiles/track_metadata.db",
    'year': f"{MSD_ROOT_URL}/AdditionalFiles/tracks_per_year.txt",
    'bpm': None,
    'cdr': None,
    'unique_tracks': f"{MSD_ROOT_URL}/AdditionalFiles/unique_tracks.txt"
}
CHECKSUMS = {
    'echonest': 'fadbc813c11881b3147af3c26006335f',
    'lastfm': 'ed3ce4acaa8fc1935267c7cb98601804',
    'mxm': '45bb0c46ffe8c892e19af90515db473e',
    'metadata': '89705ce338a360ddcf7e527cd91f8328',
    'year': 'a5793654ba6c13eaa99ca80a2caf2a9d',
    'bpm': None,
    'cdr': None,
    'unique_tracks': 'a02eb8275b10286d678309a19e229bd0'
}


def load_lastfm(
    db_fn: str
) -> InteractionData:
    """ preprocess lastfm data to output :obj:`~musmtl.preprocess.routine.InteractionData`.

    Args:
        db_fn: filename of lastfm db data

    Returns:
        preprocessed interaction data

    Raises:
        ValueError: if db file doesn't exist
    """
    # wrap it to the path
    db_path = Path(db_fn)
    if not db_path.exists():
        raise ValueError('[ERROR] db file does not exist!')

    # open the connection
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()

        # fetch entity names
        tids = [r[0] for r in cur.execute("SELECT * FROM tids").fetchall()]
        tags = [r[0] for r in cur.execute("SELECT * FROM tags").fetchall()]

        # prep the containers for sparse mat
        rows, cols, data = [], [], []
        tid_tag = cur.execute("SELECT * FROM tid_tag").fetchall()
        for tid, tag, intensity in tid_tag:
            rows.append(tid - 1)
            cols.append(tag - 1)
            data.append(max(intensity, 1.))

        # wrap it to the sparse matrix
        mat = sp.coo_matrix(
            (data, (rows, cols)),
            shape=(len(tids), len(tags))
        )

    # output the result
    return InteractionData(mat, tids, tags)


def load_echonest(
    db_fn: str,
    unique_tracks_fn: str
) -> InteractionData:
    """ preprocess echonest data to output :obj:`~musmtl.preprocess.routine.InteractionData`.

    Args:
        db_fn: filename of echonest db data
        unique_tracks_fn: filename of track list metadata

    Returns:
        preprocessed interaction data

    Raises:
        ValueError: if db file doesn't exist
    """
    # wrap it to the path
    paths = {'echonest': Path(db_fn),
             'unique_tracks': Path(unique_tracks_fn)}
    for key, path in paths.items():
        if not path.exists():
            raise ValueError(f'[ERROR] {key} file does not exist!')

    # load unique tracks and build songid -> trackid map
    sid2tid = {}
    with paths['unique_tracks'].open('r') as fp:
        for line in fp:
            tid, sid, _, _ = line.replace('\n', '').split('<SEP>')
            sid2tid[sid] = tid

    # prep the containers
    rows, cols, data = [], [], []
    users = {}
    users_list = []
    items = {}
    items_list = []

    # load the zipped file and process them
    with zipfile.ZipFile(paths['echonest']) as zfp:
        with zfp.open('train_triplets.txt') as fp:
            for line in fp:

                # parse
                u, i, intensity = (
                    line.decode('utf8').replace('\n', '').split('\t')
                )
                i = sid2tid[i]  # convert to tid

                # register to the containors
                if u not in users:
                    users[u] = len(users)
                    users_list.append(u)
                if i not in items:
                    items[i] = len(items)
                    items_list.append(i)

                rows.append(items[i])
                cols.append(users[u])
                data.append(float(intensity))

    # wrap to the matrix
    mat = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(len(items), len(users))
    )

    return InteractionData(mat, items_list, users_list)


def load_mxm(
    db_fn: str,
    apply_tfidf: bool = True
) -> InteractionData:
    """ preprocess musixmatch data to output :obj:`~musmtl.preprocess.routine.InteractionData`.

    Args:
        db_fn: filename of musixmatch db data
        apply_tfidf: determines whether we apply TF-IDF on the raw data.

    Returns:
        preprocessed interaction data

    Raises:
        ValueError: if db file doesn't exist
    """
    # wrap it to the path
    db_path = Path(db_fn)
    if not db_path.exists():
        raise ValueError('[ERROR] db file does not exist!')

    # open the connection
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()

        # prep the containers for sparse mat
        rows, cols, data = [], [], []
        tids = {}
        tids_list = []
        words = {}
        words_list = []

        triplets = cur.execute("SELECT track_id, word, count FROM lyrics").fetchall()
        for tid, word, count in triplets:

            # register to the containors
            if tid not in tids:
                tids[tid] = len(tids)
                tids_list.append(tid)
            if word not in words:
                words[word] = len(words)
                words_list.append(word)

            rows.append(tids[tid])
            cols.append(words[word])
            data.append(float(count))

        # wrap it to the sparse matrix
        mat = sp.coo_matrix(
            (data, (rows, cols)),
            shape=(len(tids), len(words))
        )

        if apply_tfidf:
            mat = TfidfTransformer().fit_transform(mat)

    return InteractionData(mat, tids_list, words_list)


def load_cdr(
    db_fn: str
) -> InteractionData:
    """ preprocess CDR-tag data to output :obj:`~musmtl.preprocess.routine.InteractionData`.

    Args:
        db_fn: filename of lastfm db data

    Returns:
        preprocessed interaction data

    Raises:
        ValueError: if db file doesn't exist
    """
    # wrap it to the path
    db_path = Path(db_fn)
    if not db_path.exists():
        raise ValueError('[ERROR] db file does not exist!')

    # prep the containers
    rows, cols, data = [], [], []
    tids = {}
    tids_list = []
    tags = {}
    tags_list = []

    # load the zipped file and process them
    with zipfile.ZipFile(db_path) as zfp:
        with zfp.open('CDR_genre.txt') as fp:
            for line in fp:

                # parse
                tid, tag = (
                    line.decode('utf8').replace('\n', '').split('<SEP>')
                )

                # register to the containors
                if tid not in tids:
                    tids[tid] = len(tids)
                    tids_list.append(tid)
                if tag not in tags:
                    tags[tag] = len(tags)
                    tags_list.append(tag)

                rows.append(tids[tid])
                cols.append(tags[tag])
                data.append(1.)

    # wrap to the matrix
    mat = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(len(tids), len(tags))
    )

    return InteractionData(mat, tids_list, tags_list)


def __fetch_track_artist_map(
    metadata_db_fn: str
) -> list[tuple[str, str]]:
    """ fetch track - artist map from the metadata db

    it is a sub-routine that fetches track - artist map from the metadata db

    Args:
        metadata_db_fn: filename of msd track metadata db

    Returns:
        track - artist map

    Raises:
        ValueError: if db file doesn't exist
    """
    metadata_db_path = Path(metadata_db_fn)
    if not metadata_db_path.exists():
        raise ValueError(f'[ERROR] metadata db file does not exist!')

    # fetch the msd track - artist map from metadata db
    with sqlite3.connect(metadata_db_path) as conn:
        cur = conn.cursor()

        # prep the containers for sparse mat
        track_artist = cur.execute(
            "SELECT track_id, artist_id FROM songs"
        ).fetchall()
    return track_artist


def load_artist(
    lastfm_db_fn: str,
    metadata_db_fn: str
) -> InteractionData:
    """ preprocess MSD-Artist tag data to output :obj:`~musmtl.preprocess.routine.InteractionData`.

    This procedure combines the track-artist information and track-tag
    information to generate track-artist-tag factor model, in rather
    simple way. The procedure involves

    Args:
        lastfm_db_fn: filename of msd-lastfm db data
        metadata_db_fn: filename of msd track metadata db

    Returns:
        preprocessed interaction data

    Raises:
        ValueError: if db file doesn't exist
    """
    # load lastfm data first
    lastfm = load_lastfm(paths['lastfm'].as_posix())
    tid_inv = {tid: i for i, tid in enumerate(lastfm.row_entities)}

    # fetch the msd track - artist map from metadata db
    track_artist = __fetch_track_artist_map(metadata_db_fn)

    # prep the containers for sparse mat
    rows, cols, data = [], [], []
    aids = {}
    aids_list = []
    for tid, aid in track_artist:

        # we don't care if it's not part of the LastFM
        if tid not in tid_inv:
            continue

        # register to the containors
        if aid not in aids:
            aids[aid] = len(aids)
            aids_list.append(aid)

        rows.append(tid_inv[tid])
        cols.append(aids[aid])
        data.append(1.)

    # wrap it to the sparse matrix
    mat = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(len(tid_inv), len(aids))
    ).T.tocoo()

    # get artist - tag matrix (num_artist, num_tags)
    final_mat = mat @ lastfm.mat

    return InteractionData(final_mat, aids_list, lastfm.col_entities)


def post_process_artist(
    artist_tag_factors: ProcessedData,
    lastfm_db_fn: str,
    metadata_db_fn: str
) -> ProcessedData:
    """ additional post-hoc routine to assign artist-tag factors to each tracks

    it is a simple post-hoc routine for assigning artist-tag factors to their
    corresponding tracks accordingly, using the track - artist map.

    Args:
        artist_tag_factors: artist-tag factor data computed from artist-tag data
        lastfm_db_fn: filename of msd-lastfm db data
        metadata_db_fn: filename of msd track metadata db.

    Returns:
        post-processed [track - [artist - tag]] factor data.

    Raises:
        ValueError: if db file doesn't exist
    """
    # get inverse map from processed artist - tag factor model
    aid_inv = {aid: i for i, aid in enumerate(artist_tag_factors.row_entities)}

    # load lastfm data first
    lastfm = load_lastfm(paths['lastfm'].as_posix())
    tid_inv = {tid: i for i, tid in enumerate(lastfm.row_entities)}

    # fetch the msd track - artist map from metadata db
    tid2aid = dict(__fetch_track_artist_map(metadata_db_fn))
    tid2aid_idx = [aid_inv[tid2aid[tid]] for tid in lastfm.row_entities]

    # index to get the re-assigned track - [artist - tag] factors
    new_mat = artist_tag_factors.row_factors[tid2aid_idx]

    # wrap and return
    return ProcessedData(
        row_factors=new_mat,
        col_factors=artist_tag_factors.col_factors,
        row_entities=lastfm_row_entities,
        col_entities=artist_tag_factors.col_entities
    )


def load_bpm(
    db_fn: str
) -> NumericData:
    """ preprocess MSD-BPM (estimation) data to output :obj:`~musmtl.preprocess.routine.NumericData`.

    Args:
        db_fn: filename of msd-bpm (estimation) data

    Returns:
        preprocessed numeric data

    Raises:
        ValueError: if db file doesn't exist
    """
    # wrap it to the path
    db_path = Path(db_fn)
    if not db_path.exists():
        raise ValueError('[ERROR] db file does not exist!')

    # prep the containers
    values, entities = [], []

    # load the zipped file and process them
    with zipfile.ZipFile(db_path) as zfp:
        with zfp.open('msd_tempo_est.txt') as fp:
            for line in fp:

                # parse
                tid, bpm = (
                    line.decode('utf8').replace('\n', '').split('\t')
                )
                entities.append(tid)
                values.append(float(bpm))

    return NumericData(np.array(values), entities)


def load_year(
    db_fn: str
) -> NumericData:
    """ preprocess MSD-release year data to output :obj:`~musmtl.preprocess.routine.NumericData`.

    Args:
        db_fn: filename of msd-release year (estimation) data

    Returns:
        preprocessed numeric data

    Raises:
        ValueError: if db file doesn't exist
    """
    # wrap it to the path
    db_path = Path(db_fn)
    if not db_path.exists():
        raise ValueError('[ERROR] db file does not exist!')

    # prep the containers
    values, entities = [], []

    # load the zipped file and process them
    with db_path.open('r') as fp:
        for line in fp:
            # parse
            year, tid, _, _ = line.replace('\n', '').split('<SEP>')
            entities.append(tid)
            values.append(float(year))
    return NumericData(np.array(values), entities)


LOAD_FUNCS = {
    'echonest': load_echonest,
    'lastfm': load_lastfm,
    'mxm': load_mxm,
    'cdr': load_cdr,
    'artist': load_artist,
    'bpm': load_bpm,
    'year': load_year,
}

REQUIRED_DATA = {
    'echonest': ('echonest', 'unique_tracks'),
    'lastfm': ('lastfm',),
    'mxm': ('mxm',),
    'cdr': ('cdr',),
    'artist': ('lastfm', 'track_metadata'),
    'bpm': ('bpm',),
    'year': ('year',)
}

POST_PROCESSING_HOOK = {
    'artist': (post_process_artist, ('lastfm', 'track_metadata'))
}


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
                             hash_type='md5',
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
            POST_PROCESSING_HOOK
        )

        # save the output
        with (Path(args.path) / 'target_factors.pkl').open('wb') as fp:
            pkl.dump(factor_output, fp)

    else:
        ValueError('[ERROR] only `fetch` and `fitfactor` subcommands are '
                   'available!')


if __name__ == "__main__":
    main()
