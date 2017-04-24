#!/usr/bin/env python

"""
"""
import time
#start = time.time()

import matplotlib 
matplotlib.use('agg')
import argparse
from main import *

path = os.getcwd()

parser = argparse.ArgumentParser()

parser.add_argument('--sig_high', type=float)
parser.add_argument('--sig_low', type=float)
parser.add_argument('--n_sigs', type=int)
parser.add_argument('--model')
parser.add_argument('--mass', type=float)
parser.add_argument('--fnfp', type=float)
parser.add_argument('--element')
parser.add_argument('--exposure', type=float)
parser.add_argument('--e_th', type=float)
parser.add_argument('--delta', type=float)
parser.add_argument('--time_info')
parser.add_argument('--GF')
parser.add_argument('--file_tag')
parser.add_argument('--n_runs', type=int)

args = parser.parse_args()

if args.time_info == 'T':
    time = True
elif args.time_info == 'F':
    time = False
if args.GF == 'T':
    GF = True
elif args.GF == 'F':
    GF = False



nu_floor(args.sig_low, args.sig_high, n_sigs=args.n_sigs, model=args.model,
         mass=args.mass, fnfp=args.fnfp, element=args.element, exposure=args.exposure,
         delta=args.delta, GF=False, time_info=time, file_tag=args.file_tag, n_runs=args.n_runs,
         Eth=args.e_th)

