import os
import sys
# add repo root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse

import numpy as np

from musmtl.tool import FeatureExtractor
from musmtl.utils import extract_mel, parmap


# setup arg parser
parser = argparse.ArgumentParser()
parser.add_argument("model_checkpoint", help='path to model checkpoint file (.pth.gz)')
parser.add_argument("target_audios", help='path to file listing target audio files (.txt)')
parser.add_argument("out_fn", help='filename for the output')
parser.add_argument("--is-gpu", type=bool, default=False,
                    help='flag for gpu computation on feature extraction')
args = parser.parse_args()

# init worker and run!
print('Initiating worker...')
ext = FeatureExtractor(
    args.model_checkpoint, args.target_audios,
    args.is_gpu
)
feature = ext.run()

print('Saving...')
np.save(args.out_fn, feature)