import os
from os.path import join
import glob
from random import shuffle
import argparse

# from paper
N_TRAIN = 41841
N_VALID = 4649

# setup parser
parser = argparse.ArgumentParser()
parser.add_argument("melroot",
                    help="root directory where melspec ('.npy') saved")

parser.add_argument("trainfn",
                    help="filename for output txt file contain training candidate paths")

parser.add_argument("validfn",
                    help="filename for output txt file contain training candidate paths")

parser.add_argument("--num-train", dest='n_train', type=int,
                    default=N_TRAIN, help="number of training samples")

parser.add_argument("--num-valid", dest='n_valid', type=int,
                    default=N_VALID, help="number of training samples")

args = parser.parse_args()

# load all fns
fns = glob.glob(join(args.melroot, '*.npy'))
assert len(fns) >= args.n_train  # sanity check

# split it into train / valid set
shuffle(fns)
train_fns = fns[:args.n_train]
valid_fns = fns[args.n_train:args.n_train + args.n_valid]

# write paths
with open(args.trainfn, 'w') as f:
    for fn in train_fns:
        f.write('{}\n'.format(fn))
    
with open(args.validfn, 'w') as f:
    for fn in valid_fns:
        f.write('{}\n'.format(fn))