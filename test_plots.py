import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as pl
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from experiments import *
from likelihood import *
from helpers import *
from rate_UV import *

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
test_plots = os.getcwd() + '/Test_Plots/'

#solar nu flux taken from SSM 1104.1639


def neutrino_spectrum(lab='Snolab', Emin=0.1, Emax=1000., fs=18, save=True):
    filename = test_plots + 'NeutrinoFlux_' + lab + '.pdf'
    ylims = [10**-4., 10**13.]

    nu_comp = ['b8', 'b7l1', 'b7l2', 'pepl1', 'hep', 'pp', 'o15', 'n13', 'f17', 'atmnue',
               'atmnuebar', 'atmnumu', 'atmnumubar', 'dsnb3mev', 'dsnb5mev', 'dsnb8mev',
               'reactor', 'geoU', 'geoTh','geoK']
    color_list = ['#800080', '#000080', '#000080', '#8A2BE2', '#A52A2A', '#A0522D', '#DC143C', '#B8860B',
                  '#8B008B', '#556B2F', '#FF8C00', '#9932CC', '#E9967A', '#FF1493', '#696969', '#228B22',
                  '#40E0D0', '#CD5C5C', '#90EE90', '#90EE90']
    nu_labels = ['8B', '7B [384.3 keV]', '7B [861.3 keV]', 'pep', 'hep', 'pp', '15O', '13N', '17F',
                 r'atm $\nu_e$', r'atm $\nu_{\bar e}$', r'atm $\nu_\mu$',
                 r'atm $\nu_{\bar \mu}$', 'DSN 3 MeV',
                 'DSN 5 MeV', 'DSN 8 MeV', 'Reactor', 'Geo U', 'Geo Th','Geo K']

    pl.figure()
    ax = pl.gca()

    for i,nu in enumerate(nu_comp):
        er, spec = Nu_spec(lab).nu_spectrum_enu(nu)
        pl.plot(er, spec, color_list[i], lw=1, label=nu_labels[i])

    ax.set_xlabel(r'$E_\nu$  [MeV]', fontsize=fs)
    ax.set_ylabel(r'Neutrino Flux  [$cm^{-2} s^{-1} MeV^{-1}$]', fontsize=fs)


    plt.tight_layout()

    plt.xlim(xmin=Emin, xmax=Emax)
    plt.ylim(ymin=ylims[0],ymax=ylims[1])
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.legend(loc=1, frameon=True, framealpha=0.5, fontsize=9, ncol=1, fancybox=True)


    if save:
        plt.savefig(filename)
    return


def neutrino_recoils(Emin=0.001, Emax=100., element='Germanium', fs=18, save=True,
                     mass=6., sigmap=4.*10**-45., model='sigma_si', fnfp=1.,
                     delta=0., GF=False, time_info=False, xenlab='LZ'):
    coupling = "fnfp" + model[5:]

    filename = test_plots + 'Recoils_in_' + element + '_'
    filename += model + '_' + coupling + '_{:.2f}'.format(fnfp)
    filename += '_DM_mass_{:.2f}_CSec_{:.2e}_Delta_{:.2f}'.format(mass, sigmap, delta)
    if element == 'xenon':
        filename += xenlab + '_'
    filename += '.pdf'

    er_list = np.logspace(np.log10(Emin), np.log10(Emax), 500)

    experiment_info, Qmin, Qmax = Element_Info(element)
    lab = laboratory(element, xen=xenlab)

    nu_comp = ['b8', 'b7l1', 'b7l2', 'pepl1', 'hep', 'pp', 'o15', 'n13', 'f17', 'atmnue',
               'atmnuebar', 'atmnumu', 'atmnumubar', 'dsnb3mev', 'dsnb5mev', 'dsnb8mev',
               'reactor', 'geoU', 'geoTh','geoK']

    nu_labels = ['8B', '7B [384.3 keV]', '7B [861.3 keV]', 'pep', 'hep', 'pp', '15O', '13N', '17F',
                 r'atm $\nu_e$', r'atm $\nu_{\bar e}$', r'atm $\nu_\mu$',
                 r'atm $\nu_{\bar \mu}$', 'DSN 3 MeV',
                 'DSN 5 MeV', 'DSN 8 MeV', 'Reactor', 'Geo U', 'Geo Th','Geo K']

    nu_lines = ['b7l1', 'b7l2', 'pepl1']
    line_flux = [(0.1) * 5.00 * 10. ** 9., (0.9) * 5.00 * 10. ** 9., 1.44 * 10. ** 8.]
    e_lines = [0.380, 0.860, 1.440]

    color_list = ['#800080', '#000080', '#000080', '#8A2BE2', '#A52A2A', '#A0522D', '#DC143C', '#B8860B',
                  '#8B008B', '#556B2F', '#FF8C00', '#9932CC', '#E9967A', '#FF1493', '#696969', '#228B22',
                  '#40E0D0', '#CD5C5C', '#90EE90','#90EE90']
    line_list = ['-', '--', '--', '-', '-','-', '-','-', '-','-', '-','-', '-','-', '-','-', '-','-', '-','-']

    nu_contrib = len(nu_comp)

    nuspec = np.zeros(nu_contrib, dtype=object)

    for i in range(nu_contrib):
        nuspec[i] = np.zeros_like(er_list)

    for iso in experiment_info:
        for i in range(nu_contrib):
            nuspec[i] += Nu_spec(lab).nu_rate(nu_comp[i], er_list, iso)

    coupling = "fnfp" + model[5:]

    drdq_params = default_rate_parameters.copy()
    drdq_params['element'] = element
    drdq_params['mass'] = mass
    drdq_params[model] = sigmap
    drdq_params[coupling] = fnfp
    drdq_params['delta'] = delta
    drdq_params['GF'] = GF
    drdq_params['time_info'] = time_info

    time_list = np.zeros_like(er_list)
    dm_spec = dRdQ(er_list, time_list, **drdq_params) * 10. ** 3. * s_to_yr

    pl.figure()
    ax = pl.gca()

    ax.set_xlabel(r'Recoil Energy  [keV]', fontsize=fs)
    ax.set_ylabel(r'Event Rate  [${\rm ton}^{-1} {\rm yr}^{-1} {\rm keV}^{-1}$]', fontsize=fs)

    for i in range(nu_contrib):
        pl.plot(er_list, nuspec[i], color_list[i], ls=line_list[i], lw=1, label=nu_labels[i])

    pl.plot(er_list, dm_spec, 'b', lw=1, label='Dark Matter')
    print 'Number of dark matter events: ', np.trapz(dm_spec, er_list)
    plt.tight_layout()

    plt.xlim(xmin=Emin, xmax=Emax)
    plt.ylim(ymin=10.**-5., ymax=10.**8.)
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.legend(loc=1, frameon=True, framealpha=0.5, fontsize=9, ncol=1, fancybox=True)

    if save:
        plt.savefig(filename)
    return


def test_sims(esim, e_list, dm, nu):
    pl.figure()
    ax = pl.gca()
    ax.set_xlabel(r'Recoil Energy  [keV]', fontsize=18)
    ax.set_ylabel(r'Event Rate  [${\rm ton}^{-1} {\rm yr}^{-1} {\rm keV}^{-1}$]', fontsize=18)

    pl.plot(e_list, dm, 'r', lw=1, label='Dark Matter')
    pl.plot(e_list, nu, 'blue', lw=1, label='Neutrino')


    bin = np.linspace(np.min(e_list), 2., 100)
    ax.hist(esim, bins=bin, normed=True, histtype='step', fc=None, ec='Black', lw=2)

    plt.tight_layout()

    plt.xlim(xmin=np.min(e_list), xmax=2.)
    ax.set_xscale("log")

    plt.savefig(test_plots + 'TEST.pdf')
    return


def number_dark_matter(element='Xenon', model='sigma_si', exposure=1., mass=10., sigma=1e-40,
                       fnfp=1., delta=0., GF=False, time_info=False):
    experiment_info, Qmin, Qmax = Element_Info(element)
    coupling = "fnfp" + model[5:]

    er_list = np.logspace(np.log10(Qmin), np.log10(Qmax), 100)

    drdq_params = default_rate_parameters.copy()
    drdq_params['element'] = element
    drdq_params[model] = sigma
    drdq_params[coupling] = fnfp
    drdq_params['delta'] = delta
    drdq_params['GF'] = GF
    drdq_params['time_info'] = time_info
    drdq_params['mass'] = mass
    spectrum = dRdQ(er_list, np.zeros_like(er_list), **drdq_params)
    rate = np.trapz(spectrum, er_list) * exposure * 3.154*10.**7. * 1e3

    print 'Element: ', element
    print 'Model: ', model, ' Mass {:.2f} Sigma: {:.2e}'.format(mass, sigma)
    print 'Dark Matter Events: {:.2f}'.format(rate)
    return

