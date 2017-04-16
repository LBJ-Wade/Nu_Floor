~~~~~~~~~~~~~~~~~~~~~
Code calculates discovery limits for generic DM interactions
~~~~~~~~~~~~~~~~~~~~~


How to run:

1.) Edit file cluster_runner.py 
This controls experiment that will be run, dark matter model, mass and cross section range to scan, etc.

2.) Run ‘python cluster_runner.py’

3.) ‘cd current_runs’

4.) If on server, run ‘qsub commandrunner_.sh’, if on local computer, individually run ‘bash nu_floor_runner__{}.sh’ where {} should be replaced by one of the file numbers

5.) Output files will be saved in folder ‘Saved_Files’


Summary of files:

Nu_runner.py — sole purpose is to call nu_floor, located in main.py

constants.py — stores constants (duh)

experiments.py — holds experimental info, will need to be edited to allow for more generic experiments

formUV.py — has DM differential cross sections for all interactions

globals.py — has some generic global definitions (I actually don’t think its used for this program, I may have removed its purpose for this program)

helpers.py — has some generic functions like \eta(\vmin)

likelihood.py — contains neutrino flux calculations and likelihood functions

main.py — contains nu_floor program which is how one obtains and outputs test statistics as a function of mass and cross section

rate_UV.py — calculates the differential rate and the integrated rate for DM

test_plots.py — creates test plots we can use as checks. Currently there are two functions which can be called by writing the following in python:

from test_plots import *
neutrino_specturm()
neutrino_recoils()



OTHER INFO:

—See main.py for possible experimental and model inputs.

—Don’t use inelastic scattering yet. Can almost be used for SI/SD, but not for interactions which have v_\perp dependence

—Don’t use GF or time_info. These are not implemented.

—Check that minimization is not failing.

