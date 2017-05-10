import numpy as np
import os
from likelihood import *
from helpers import *
from rate_UV import *
import numpy.random as random
import matplotlib
matplotlib.use('Agg')
import pylab as pl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import matplotlib.patheffects as PathEffects
import matplotlib.gridspec as gridspec
import glob
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times','Palatino']})
rc('text', usetex=True)


mpl.rcParams['xtick.major.size']=8
mpl.rcParams['ytick.major.size']=8
mpl.rcParams['xtick.labelsize']=18
mpl.rcParams['ytick.labelsize']=18

path = os.getcwd()
Sv_dir = path + '/NeutrinoSims/'
test_plots = os.getcwd() + '/Test_Plots/'


def simulate_neutrino_recoils(number=100000, element='Germanium', file_tag='_', xenLAB='LZ',
                              Emax=None, Emin=None):

    experiment_info, Qmin, Qmax = Element_Info(element)
    labor = laboratory(element, xen=xenLAB)
    if Emax:
        Qmax = Emax
    if Emin:
        Qmin = Emin

    Ner = 10000
    nu_comp = ['b8', 'b7l1', 'b7l2', 'pepl1', 'hep', 'pp', 'o15', 'n13', 'f17', 'atm',
               'dsnb3mev', 'dsnb5mev', 'dsnb8mev', 'reactor', 'geoU', 'geoTh', 'geoK']

    keep_nus = []
    max_es = np.zeros(len(nu_comp))
    for i in range(len(nu_comp)):
        max_es[i] = Nu_spec(labor).max_er_from_nu(NEUTRINO_EMAX[nu_comp[i]], experiment_info[0][0])
        if max_es[i] > Qmin:
            keep_nus.append(i)
            if max_es[i] > Qmax:
                max_es[i] = Qmax
    nu_comp = [x for i, x in enumerate(nu_comp) if i in keep_nus]
    max_es = max_es[keep_nus]
    nu_contrib = len(nu_comp)
    print 'Neutrinos Considered: ', nu_comp

    for i,nu_name in enumerate(nu_comp):
        file_info = Sv_dir + 'Simulate_' + nu_name + '_' + element
        file_info += '_Eth_{:.2f}_Emax_{:.2f}_'.format(Qmin,Qmax) + labor + '_'
        file_info += file_tag + '.dat'
        print 'Output File: ', file_info

        recoils = np.zeros(number)
        er_list = np.logspace(np.log10(Qmin), np.log10(max_es[i]), Ner)
        nuspectrum = np.zeros(Ner)
        for iso in experiment_info:
            nuspectrum += Nu_spec(labor).nu_rate(nu_name, er_list, iso)

        nu_pdf = nuspectrum
        cdf_nu = np.zeros_like(nu_pdf)
        for j in range(len(nu_pdf)):
            cdf_nu[j] = np.trapz(nu_pdf[:j],er_list[:j])
        cdf_nu /= cdf_nu.max()

        u = random.rand(number)
        for j in range(number):
            recoils[j] = er_list[np.absolute(cdf_nu - u[j]).argmin()]
        np.savetxt(file_info, recoils)

    return

def compare_simulated_spectrum(nu='b8', element='Germanium', file_tag='_', xenLAB='LZ', fs=20):

    experiment_info, Qmin, Qmax = Element_Info(element)
    labor = laboratory(element, xen=xenLAB)

    file_info = Sv_dir + 'Simulate_' + nu + '_' + element
    file_info += '_Eth_{:.2f}_Emax_{:.2f}_'.format(Qmin, Qmax) + labor + '_'
    file_info += file_tag + '.dat'
    simulated_events = np.loadtxt(file_info)

    pl.figure()
    ax = pl.gca()

    er_list = np.logspace(np.log10(Qmin), np.log10(Qmax), 1000)
    nuspectrum = np.zeros_like(er_list)
    for iso in experiment_info:
        nuspectrum += Nu_spec(labor).nu_rate(nu, er_list, iso)
    nuspectrum /= np.trapz(nuspectrum, er_list)

    pl.plot(er_list, nuspectrum, 'green', lw=1, label=nu)
    n, bins, patches = plt.hist(simulated_events, bins='auto',
                                range=(np.min(simulated_events), np.max(simulated_events)),
                                normed=1, log=True, facecolor='blue', alpha=0.4)

    ax.set_xlabel(r'$E_R$  [keV]', fontsize=fs)
    ax.set_ylabel(r'Recoil Spectrum  [$keV^{-1}$]', fontsize=fs)

    plt.xlim(xmin=Qmin, xmax=np.max(simulated_events))
    plt.ylim(ymin=10. ** -3., ymax=np.max(n))
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.tight_layout()
    filename = test_plots + '/Check_Simulated_Spectrum_' + nu + '_' + element
    filename += '_Eth_{:.2f}_Emax_{:.2f}_'.format(Qmin, Qmax) + labor + '_.pdf'
    plt.savefig(filename)
    return


def simulate_DM_recoils(number=100000, element='Germanium', file_tag='_', xenLAB='LZ',
                        model="sigma_si", fnfp=1., delta=0., GF=False, time_info=False,
                        mass_arr=np.concatenate((np.linspace(1., 8.5, 13), np.linspace(9., 25., 8),
                                                 np.logspace(np.log10(30.), 3., 20)))):

    experiment_info, Qmin, Qmax = Element_Info(element)
    labor = laboratory(element, xen=xenLAB)

    er_list = np.logspace(np.log10(Qmin), np.log10(Qmax), 10000)
    time_list = np.zeros_like(er_list)

    coupling = "fnfp" + model[5:]
    mindm = np.zeros(len(experiment_info[:, 0]))
    for i, iso in enumerate(experiment_info):
        mindm[i] = MinDMMass(iso[0], delta, Qmin, vesc=533. + 232.)
    MinDM = np.min(mindm)
    print 'Minimum DM mass: ', MinDM

    drdq_params = default_rate_parameters.copy()
    drdq_params['element'] = element
    drdq_params[model] = 1e-40
    drdq_params[coupling] = fnfp
    drdq_params['delta'] = delta
    drdq_params['GF'] = GF
    drdq_params['time_info'] = time_info



    for mass in mass_arr:
        if mass < MinDM + 0.5:
            continue

        file_info = path + '/DarkMatterSims/Simulate_DarkMatter_' + element
        file_info += '_' + model + '_' + coupling + '_{:.2f}_DM_Mass_{:.2f}_GeV'.format(fnfp,mass)
        file_info += '_Eth_{:.2f}_Emax_{:.2f}_'.format(Qmin,Qmax) + labor + '_'
        file_info += file_tag + '.dat'
        print 'Output File: ', file_info

        drdq_params['mass'] = mass

        dm_spec = dRdQ(er_list, time_list, **drdq_params)

        recoils = np.zeros(number)

        pdf = dm_spec
        cdf = np.zeros_like(pdf)
        for j in range(len(pdf)):
            cdf[j] = np.trapz(pdf[:j],er_list[:j])
        cdf /= cdf.max()

        u = random.rand(number)
        for j in range(number):
            recoils[j] = er_list[np.absolute(cdf - u[j]).argmin()]

        np.savetxt(file_info, recoils)

    return