#!/usr/bin/python
import time
import json
import os
from os.path import join, abspath
import sys
# add repo root to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
from musmtl.experiment import Experiment

# setup arg parser
parser = argparse.ArgumentParser()
parser.add_argument("config", help='configuration (.json) path')
args = parser.parse_args()

# load config file
conf = json.load(open(args.config))

# instantiate experiment & run
Experiment(conf).run()