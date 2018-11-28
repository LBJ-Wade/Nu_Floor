#!/usr/bin/env python

"""
"""
import time
#start = time.time()
import matplotlib
matplotlib.use('agg')
import argparse
from main import *
from identify_neutrinos import *
from constants import *
from experiments import *
import numpy as np
import glob
from construct_floors import *


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
parser.add_argument('--DARK',default='T')

args = parser.parse_args()

if args.time_info == 'T':
    timeT = True
else:
    timeT = False
if args.GF == 'T':
    GF = True
elif args.GF == 'F':
    GF = False
if args.DARK == 'T':
    DARK = True
else:
    DARK = False

# normally set to false except for Carlos Blanco project
BOUND_DERIVE = False

if DARK:
    if not BOUND_DERIVE:
        nu_floor(args.sig_low, args.sig_high, n_sigs=args.n_sigs, model=args.model,
                 mass=args.mass, fnfp=args.fnfp, element=args.element, exposure=args.exposure,
                 delta=args.delta, GF=False, time_info=timeT, file_tag=args.file_tag, n_runs=args.n_runs,
                 Eth=args.e_th)
    else:
        nu_floor_Bound(args.sig_low, args.sig_high, n_sigs=args.n_sigs,
                       model=args.model, mass=args.mass, fnfp=args.fnfp,
                       element=args.element, exposure=args.exposure,
                       delta=args.delta, GF=False, time_info=timeT, file_tag=args.file_tag,
                       n_runs=args.n_runs, Eth=args.e_th)


else:
    identify = np.array(['geoU', 'geoTh', 'geoK'])
    maxE = 0.
    uncert = 0.5
    for i in identify:
        elem, Qmax, Qmin = Element_Info(args.element)
        maxER = Nu_spec(lab='Snolab').max_er_from_nu(NEUTRINO_EMAX[i], elem[0,0])
        if maxER > maxE:
            maxE = maxER
    print 'Maximum Energy:', maxE
    print 'Uncertainty Multiplier', uncert
    identify_nu(exposure_low=0.1, exposure_high=50., expose_num=13, element=args.element,
                file_tag=args.file_tag, n_runs=500, Eth=args.e_th, Ehigh=maxE,
                identify=identify, red_uncer=uncert, shotnoise=True)

