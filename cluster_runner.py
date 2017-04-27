#!/usr/bin/env python

import subprocess as sp
import os,sys,fnmatch
import argparse
import pickle
import numpy as np


path = os.getcwd()

mass_arr = np.concatenate((np.linspace(1., 8.5, 13), np.linspace(9., 25., 8),
                           np.logspace(np.log10(30.), 3., 20)))

parser = argparse.ArgumentParser()

parser.add_argument('--sig_high', type=float, default=10.**-43.)  # x-sec range
parser.add_argument('--sig_low', type=float, default=10.**-48.)  # x-sec range
parser.add_argument('--n_sigs', type=int, default=20)  # number of cross-section tests in x-sec range
parser.add_argument('--model', default="sigma_si") 
parser.add_argument('--masses', nargs='+', default=np.array([9.]), type=float,)
parser.add_argument('--fnfp', type=float, default=1.)
parser.add_argument('--element', nargs='+', default=['Xenon'])
parser.add_argument('--exposure', type=float, default=1.0)  # Ton-yr
parser.add_argument('--ethresh', type=float, default=-1.)
parser.add_argument('--delta', type=float, default=0.)  # FIX for now
parser.add_argument('--time_info',default='F')  # FIX for now
parser.add_argument('--GF', default='F')  # FIX for now
parser.add_argument('--file_tag',default='_')
parser.add_argument('--n_runs', type=int, default=5)  # number of realizations of data
parser.add_argument('--tag',default='')

args = parser.parse_args()
sig_h = args.sig_high
sig_l = args.sig_low
nsig = args.n_sigs
model = args.model
fnfp = args.fnfp
exposure = args.exposure
delta = args.delta
time_info = args.time_info
GF = args.GF
file_tag = args.file_tag
n_runs = args.n_runs
TAG = args.tag
ethresh = args.ethresh

MASSES = args.masses
EXPERIMENTS = args.element
SINGLE_EXPERIMENTS = []
for i, experiment in enumerate(EXPERIMENTS):
    labels = experiment.split()
    for lab in labels:
        if lab not in SINGLE_EXPERIMENTS:
            SINGLE_EXPERIMENTS.append(lab)


cmds = []
count = 0
for experiment in SINGLE_EXPERIMENTS:
    for mass in MASSES:
        cmd = 'cd '+ path + '\n' + 'python Nu_runner.py ' +\
              '--sig_high {} --sig_low {} --n_sigs {} '.format(sig_h, sig_l, nsig) +\
              '--model {} --mass {} --fnfp {} --element {} '.format(model, mass, fnfp, experiment) +\
              '--exposure {} --delta {} --time_info {} --GF {} '.format(exposure, delta, time_info, GF) +\
              '--file_tag {} --n_runs {} --e_th {} '.format(file_tag, n_runs, ethresh)

        cmds.append(cmd)
        count += 1

print '\n There will be {} Runs.\n'.format(count)

for i in range(count):
    fout=open('current_runs/nu_floor_runner_{}_{}.sh'.format(TAG, i+1), 'w')
    for cmd in cmds[i::count]:
        fout.write('{}\n'.format(cmd))
    fout.close()

fout = open('current_runs/commandrunner_{}.sh'.format(TAG), 'w')
fout.write('#! /bin/bash\n')
if exposure > 30.:
    data = 2
else:
    data = 1
fout.write('#$ -l h_rt=24:00:00,h_data='+str(data)+'G\n')
fout.write('#$ -cwd\n')
fout.write('#$ -t 1-{}\n'.format(count))
fout.write('#$ -V\n')
fout.write('bash nu_floor_runner_{}_$SGE_TASK_ID.sh\n'.format(TAG))
fout.close()

