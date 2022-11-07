from typing import Optional, Union
from pathlib import Path
import hashlib
import requests
import logging

from tqdm import tqdm


logging.basicConfig()
logger = logging.getLogger(__name__)


def _fetch(
    url: str,
    out_dir: str,
    block_size: int = 512,
    verbose: bool = False
) -> Path:
    """ simple utility function to fetch files from internet

    Args:
        url: the URL to the file to be fetched
        out_dir: path to dump the received file

    Returns:
        output filename

    Raises:
        :obj:`~requests.exceptions.HTTPError`:
            if there's something wrong in the file fetching.
    """
    # knit output filename
    out_fn = (Path(out_dir) / Path(url).name)

    # try to fetch the file
    try:
        response = requests.get(url, stream=True)
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        with tqdm(total=total_size_in_bytes,
                  ncols=80, unit='iB',
                  disable=not verbose,
                  unit_scale=True) as prog, \
             out_fn.open('wb') as file:

            for data in response.iter_content(block_size):
                prog.update(len(data))
                file.write(data)

        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)

    return out_fn


HASH_FUNC = {
    'md5': hashlib.md5,
    'sha1': hashlib.sha1
}


def _checksum(
    filename: str,
    original_hash: Optional[str] = None,
    hash_type: str = 'md5',
    chunksize: int = 8192
) -> str:
    """ computes md5 checksum of given file

    it computes md5 checksum of the file and compares to the original if given.

    Args:
        filename: input filename to compute hash
        original_md5: original hash checksum to be compared. If not given,
                      no comparison is made
        hash_type: type of hash function to be computed {'md5', 'sha1'}
        chunksize: size of chunk of file to compute the hash

    Returns:
        string of hex md5sum

    Raises:
        ValueErroe: 1. if computed hash does not match to the given original
                       (if given)
                    2. if hash type is not supported

    """
    if hash_type not in HASH_FUNC:
        raise ValueError('[ERROR] unsupported `hash_type`!')

    # compute hash
    with Path(filename).open('rb') as fp:
        file_hash = HASH_FUNC[hash_type]()
        while chunk := fp.read(chunksize):
            file_hash.update(chunk)
        hash_ = file_hash.hexdigest()

    # compare them if original is given
    if (original_hash is not None) and (original_hash != hash_):
        raise ValueError(f'[ERROR] {hash_type} does not match!')

    return hash_


def fetch_urls(
    out_path: Union[str, Path],
    urls: dict[str, str],
    original_hashes: Optional[dict[str, str]] = None,
    hash_type: str = 'md5',
    verbose: bool = False
) -> dict[str, str]:
    """ fetch files specified from the url dictionary

    it is a main routine function to fetch files. it checks whether
    the file is already exists and also check their md5sum for file integrity.

    Args:
        urls: target URLs to be downloaded
        original_md5s: if given, check each files' md5sum

    Returns:
        dictionary contains outputed filenames per each key given from urls

    Raises:
        ValueError: if urls and original_md5 does not have the identical keys
    """
    if (
        (original_hashes is not None) and
        (set(urls.keys()) != set(original_hashes.keys()))
    ):
        raise ValueError(
            '[ERROR] if `original_md5s` is given, their keys must be '
            'identical to the keys of `urls`'
        )

    # wrap the output path as Path object
    out_root = (
        out_path
        if isinstance(out_path, Path) else
        Path(out_path)
    )

    # first, fetch all files to the disk
    out_fns = {}
    for k, url in urls.items():

        # if there's no URL, we pass
        if url is None:
            out_fns[k] = None
            continue

        # check if the file is already there
        out_fn = out_root / Path(url).name
        if not out_fn.exists():
            logger.info(f'Fetching {k}...')
            out_fn = _fetch(url, out_root.as_posix(), verbose=verbose)

        # check hash
        _checksum(out_fn,
                  original_hashes[k],
                  hash_type)

        # put it into the containor
        out_fns[k] = out_fn.as_posix()

    return out_fns
