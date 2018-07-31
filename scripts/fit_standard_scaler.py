import os, sys, glob, argparse
from os.path import join
# add repo root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# TEMPORARY SOLUTION
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

# setup arg parser
parser = argparse.ArgumentParser()
parser.add_argument("melroot", help='root where all the mel (.npy) files located')
parser.add_argument("outfn", help='fn for output state_dict (.pth.gz) file')
args = parser.parse_args()

print('1. Get the candidates...')
fns = glob.glob(join(args.melroot, '*.npy'))

print('2. Training a scaler...')
scaler = StandardScaler()
for fn in tqdm(fns, ncols=80):
    X = np.load(fn)  # (1, 2, t, 128)
    scaler.partial_fit(X.reshape(-1, 128))
    
print('3. Saving...')
joblib.dump(scaler, args.outfn)