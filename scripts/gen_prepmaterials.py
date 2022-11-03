#!/usr/bin/python
import argparse
from pathlib import Path
import pickle as pkl


def parse_arguments() -> argparse.Namespace:
    """
    """
    # setup arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("songsroot", type=str,
                        help=('root directory where the all audio '
                              'files are stored'))
    parser.add_argument("outroot", type=str,
                        help='output root where all the mel-spec saved')
    parser.add_argument("--extension", type=str, default='mp3',
                        help='file extension of the target (audio) files')
    parser.add_argument("--dir-levels", type=int, default=0,
                        help=("indicating how many directory levels from `songsroot` "
                              "to the leaf audio file"))
    parser.add_argument('--verbose', default=True,
                        action=argparse.BooleanOptionalAction,
                        help="set verbosity")
    return parser.parse_args()


def _get_parent(
    audiofile_path: Path,
    dir_levels: int
) -> Path:
    """
    """
    # index all the parents up to the level we want to merge
    parents_to_file = [
        audiofile_path.parents[i].name for i in range(dir_levels)
    ][::-1]

    # build path and output
    return Path('/'.join(parents_to_file))


def main():
    """
    """
    args = parse_arguments()

    # setup some paths
    outroot = Path(args.outroot)
    outroot.mkdir(exist_ok=True, parents=True)

    # get the filenames
    songsroot = Path(args.songsroot)
    songs_fns = list(songsroot.glob(f'**/*.{args.extension}'))

    # build & save the tid -> filename map
    tid2fn = {
        p.stem: (_get_parent(p, args.dir_levels) / p.name).as_posix()
        for p in songs_fns
    }
    with (outroot / 'tid2fn.pkl').open('wb') as fp:
        pkl.dump(tid2fn, fp)

    # build & save the candidate lists (indexed by tid)
    with (outroot / 'candidates.txt').open('w') as fp:
        for fn in songs_fns:
            fp.write(f'{fn.stem}\n')


if __name__ == "__main__":
    main()
