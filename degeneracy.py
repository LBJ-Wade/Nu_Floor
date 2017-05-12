import matplotlib
matplotlib.use('Agg')
import pylab as pl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

import numpy as np
from likelihood import *
from experiments import *
import os
from scipy.interpolate import interp1d
from helpers import *
from rate_UV import *
import numpy.random as random
from scipy.stats import poisson
from scipy.optimize import minimize, curve_fit
from math import factorial

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

test_plots = os.getcwd() + '/Test_Plots/'


def find_degeneracy2(nu_cand='b8', Emin=0.1, Emax=5., bins=20,
                    Mmin=5., Mmax=10., Mnum=50, element='Germanium',
                    model='sigma_si', fnfp=1.,
                    delta=0., GF=False, time_info=False, xenlab='LZ'):

    mass_list = np.logspace(np.log10(Mmin), np.log10(Mmax), Mnum)

    coupling = "fnfp" + model[5:]
    er_list = np.logspace(np.log10(Emin), np.log10(Emax), 1000)

    experiment_info, Qmin, Qmax = Element_Info(element)
    labor = laboratory(element, xen=xenlab)

    nuspec = np.zeros_like(er_list)
    for iso in experiment_info:
        nuspec += Nu_spec(labor).nu_rate(nu_cand, er_list, iso)

    er_list = er_list[nuspec > 0.]
    nuspec = nuspec[nuspec > 0.]


    dm_class = DM_fitting_funcs(element, model, coupling, fnfp=fnfp,
                                delta=delta, GF=GF, time_info=time_info)

    if nu_cand == 'b8':
        mx_guess = 6.
    elif 'dsnb' in nu_cand:
        mx_guess = 20.
    elif 'atm' in nu_cand:
        if 'massless' in model and 'LS' not in model:
            mx_guess = 1000.
        else:
            mx_guess = 100.
    else:
        mx_guess = 30.
    s_norm = dm_class.dm_spectrum(np.array([er_list[0]]), 0., mx_guess)
    if s_norm > 0:
        sig_guess = nuspec[0] / s_norm
    else:
        sig_guess = np.array([10.**-44.])

    guess = np.array([np.log10(sig_guess[0]), mx_guess])
    print 'Guess: ', guess
    try:
        popt, pcov = curve_fit(dm_class.dm_spectrum, er_list, nuspec, p0=guess,
                               bounds=([-60., -30.],[1., 1000.]),
                               sigma=nuspec, absolute_sigma=False)
    except RuntimeError:
        popt, pcov = curve_fit(lambda x,y: dm_class.dm_spectrum(x,y, 1000.), er_list, nuspec, p0=guess[0],
                               sigma=nuspec, absolute_sigma=False)

    bfmass = popt[-1]
    bfcs = popt[0]
    dm = dm_class.dm_spectrum(er_list, bfcs, bfmass)

    chisq = gauss_reg2(0., dm, nuspec)

    print 'Element: ', element
    print 'Model: ', model
    print 'Coupling: ', fnfp
    print 'Delta: ', delta
    print 'DM Mass: ', bfmass
    print 'Best Fit Cross Section: ', bfcs

    return bfmass, bfcs, chisq


class DM_fitting_funcs(object):

    def __init__(self, element, model, coupling, fnfp=1., delta=0., GF=False, time_info=False):
        self.drdq_params = default_rate_parameters.copy()
        self.drdq_params['element'] = element
        self.drdq_params[model] = 1.
        self.drdq_params[coupling] = fnfp
        self.drdq_params['delta'] = delta
        self.drdq_params['GF'] = GF
        self.drdq_params['time_info'] = time_info

    def dm_spectrum(self, eng, norm, mx):
        self.drdq_params['mass'] = mx
        return 10.**norm * dRdQ(eng, np.zeros_like(eng), **self.drdq_params)*10.**3.*s_to_yr


def find_degeneracy(nu_cand='b8', Emin=0.1, Emax=5., bins=20,
                    Mmin=5., Mmax=10., Mnum=50, element='Germanium',
                    model='sigma_si', fnfp=1.,
                    delta=0., GF=False, time_info=False, xenlab='LZ'):

    mass_list = np.logspace(np.log10(Mmin), np.log10(Mmax), Mnum)

    coupling = "fnfp" + model[5:]
    er_list = np.logspace(np.log10(Emin), np.log10(Emax), 500)

    experiment_info, Qmin, Qmax = Element_Info(element)
    labor = laboratory(element, xen=xenlab)

    nuspec = np.zeros_like(er_list)
    bin_edge = np.logspace(np.log10(Emin), np.log10(Emax), bins + 1)
    nu_bins = np.zeros(bins)
    for i in range(bins):
        nuspec = 0.
        erlist2 = np.logspace(np.log10(bin_edge[i]), np.log10(bin_edge[i+1]), 100)
        for iso in experiment_info:
            nuspec += Nu_spec(labor).nu_rate(nu_cand, erlist2, iso)
        nu_bins[i] = np.trapz(nuspec, erlist2)

    store_arr = np.zeros(Mnum * 3).reshape((Mnum, 3))
    minMassarr = np.zeros_like(mass_list)
    for j,mass in  enumerate(mass_list):
        hold = np.zeros(len(experiment_info))
        if delta == 0:
            for i,iso in enumerate(experiment_info):
                hold[i] = MinDMMass(iso[0], delta, Emin, vesc=533.+232.)
            minMassarr[j] = np.min(hold)

    for i,mass in enumerate(mass_list):
        if mass < minMassarr[i]:
            continue
        drdq_params = default_rate_parameters.copy()
        drdq_params['element'] = element
        drdq_params['mass'] = mass
        drdq_params[model] = 1.
        drdq_params[coupling] = fnfp
        drdq_params['delta'] = delta
        drdq_params['GF'] = GF
        drdq_params['time_info'] = time_info

        dm_rates = np.zeros(bins)
        for j in range(bins):
            dm_rates[j] = R(Qmin=bin_edge[j], Qmax=bin_edge[j+1], **drdq_params)*10.**3.*s_to_yr

        if delta == 0.:
            sigp_list = np.linspace(-50., -30., 300)
        else:
            sigp_list = np.linspace(-60., -30., 300)

        chi_tab = np.zeros_like(sigp_list)
        for j,sig in enumerate(sigp_list):
            if not np.any(10.**sig * dm_rates < 1e-10):
                chi_tab[j] = gauss_reg(sig, dm_rates, nu_bins)
        chi_tab[chi_tab == np.nan] = 1e50
        chi_tab[chi_tab == 0.] = 1e50
        #print chi_tab
        sig_fit = sigp_list[np.argmin(chi_tab)]
        like_val = gauss_reg(sig_fit, dm_rates, nu_bins)
        store_arr[i] = [mass, np.abs(like_val), sig_fit]

    store_arr = store_arr[store_arr[:,0] > 0.]
    #print store_arr
    bf_index = np.argmin(store_arr[:, 1])
    bfmass = store_arr[bf_index, 0]
    bfcs = store_arr[bf_index, 2]

    print 'Element: ', element
    print 'Model: ', model
    print 'Coupling: ', fnfp
    print 'Delta: ', delta
    print 'DM Mass: ', bfmass
    print 'Best Fit Cross Section: ', bfcs

    return bfmass, bfcs, store_arr[bf_index, 1]


def plt_model_degeneracy(nu_cand='b8', Emin=0.1, Emax=7., bins=10,
                         Mmin=5., Mmax=14., Mnum=100, element='Germanium',
                         fnfp=1., delta=0., GF=False, time_info=False, xenlab='LZ',
                         models=np.array(['sigma_si','sigma_sd','sigma_anapole','sigma_elecdip',
                                          'sigma_magdip','sigma_LS',
                                          'sigma_f1','sigma_f2','sigma_f3',
                                          'sigma_si_massless', 'sigma_sd_massless',
                                          'sigma_anapole_massless', 'sigma_magdip_massless',
                                          'sigma_elecdip_massless', 'sigma_LS_massless',
                                          'sigma_f1_massless','sigma_f2_massless','sigma_f3_massless']),
                         fs=18,
                         c_list=np.array(['blue', 'green','red','violet','aqua','magenta','orange',
                                          'brown', 'goldenrod', 'salmon', 'grey', 'indianred']),
                         tag='_'):

    svfile = 'Model_Degeneracy_' + nu_cand + '_' + element
    svfile += '_Emin_{:.2f}_Emax_{:.2f}_delta_{:.2f}'.format(Emin, Emax, delta)
    filename = test_plots + svfile
    filename += tag + '.pdf'

    ergs = np.logspace(np.log10(Emin), np.log10(Emax), 200)
    experiment_info, Qmin, Qmax = Element_Info(element)
    labor = laboratory(element, xen=xenlab)

    nuspec = np.zeros_like(ergs)
    for iso in experiment_info:
        nuspec += Nu_spec(labor).nu_rate(nu_cand, ergs, iso)

    nu_events = np.trapz(nuspec, ergs)
    pl.figure()
    ax = pl.gca()

    ax.set_xlabel(r'Recoil Energy  [keV]', fontsize=fs)
    ax.set_ylabel(r'Event Rate  [${\rm ton}^{-1} {\rm yr}^{-1} {\rm keV}^{-1}$]', fontsize=fs)

    pl.plot(ergs, nuspec, 'k', lw=1, label=nu_cand)
    bfits = np.zeros(len(models) * 3).reshape((len(models), 3))
    dm_spec = np.zeros(len(models) * len(ergs)).reshape((len(models), len(ergs)))
    for i,mm in enumerate(models):
        label = mm[6:]
        look_low = label.find('_')
        if look_low > 0:
            label = label[:look_low] + ' ' + label[look_low+1:]
        coupling = "fnfp" + mm[5:]
        if (mm == 'sigma_si_massless' or mm == 'sigma_sd_massless' or
                    mm == 'sigma_anapole_massless'):
            massmin = 2000.
            massmax = 4000.
        elif (mm == 'sigma_magdip_massless' or mm == 'sigma_elecdip_massless') and \
            ('atm' in nu_cand):
            massmin = 2000.
            massmax = 4000.
        elif (mm == 'sigma_magdip' or mm == 'sigma_elecdip' or mm == 'sigma_LS' or
              mm == 'sigma_LS_massless') and ('atm' in nu_cand):
            massmin = 10.
            massmax = 80.
        else:
            massmin = Mmin
            massmax = Mmax
        # mass, cs, like = find_degeneracy(nu_cand=nu_cand, Emin=Emin, Emax=Emax, bins=bins,
        #                            Mmin=massmin, Mmax=massmax, Mnum=Mnum, element=element,
        #                            model=mm, fnfp=fnfp,
        #                            delta=delta, GF=GF, time_info=time_info, xenlab=xenlab)

        mass, cs, like = find_degeneracy2(nu_cand=nu_cand, Emin=Emin, Emax=Emax, bins=bins,
                                         Mmin=massmin, Mmax=massmax, Mnum=Mnum, element=element,
                                         model=mm, fnfp=fnfp,
                                         delta=delta, GF=GF, time_info=time_info, xenlab=xenlab)

        bfits[i] = [mass, cs, like]
        drdq_params = default_rate_parameters.copy()
        drdq_params['element'] = element
        drdq_params['mass'] = mass
        drdq_params[mm] = 10.**cs
        drdq_params[coupling] = fnfp
        drdq_params['delta'] = delta
        drdq_params['GF'] = GF
        drdq_params['time_info'] = time_info

        dm_spec[i] = dRdQ(ergs, np.zeros_like(ergs), **drdq_params) * 10. ** 3. * s_to_yr
        dmevts = np.trapz(dm_spec[i], ergs)
        dm_spec[i] *= nu_events/dmevts
        if 'massless' in mm:
            ls = '--'
            jj = i - len(c_list)
        else:
            ls = '-'
            jj = i
        pl.plot(ergs, dm_spec[i], c_list[jj], lw=1, ls=ls, label=label)
    bfits = bfits.T

    print 'Saving File to: ', svfile
    models_str = ''
    for i in range(len(models)):
        models_str += models[i] + '  '
    fmt_str = '%.4e ' * len(models)
    np.savetxt(test_plots + 'BF_' + svfile + '.dat', bfits, fmt=fmt_str, header=models_str)
    fmt_str = '%.4e ' * (len(models) + 2)
    np.savetxt(test_plots + 'Spectrum_' + svfile + '.dat', np.column_stack((ergs, nuspec, dm_spec.T)), fmt=fmt_str,
               header='Energy  NuSpectrum  ' + models_str)
    plt.xlim(xmin=Emin, xmax=Emax)
    if nu_cand == 'b8':
        plt.ylim(ymin=10. ** 0., ymax=3*10. ** 3.)
    elif nu_cand == 'atmnumu':
        plt.ylim(ymin=10. ** -5., ymax=5.*10.**-3.)
    elif nu_cand == 'dsnb8mev':
        plt.ylim(ymin=10. ** -5., ymax=5. * 10. ** -3.)
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.legend(loc=1, frameon=True, framealpha=0.5, fontsize=9, ncol=1, fancybox=True)
    plt.tight_layout()
    print filename
    plt.savefig(filename)
    return


def plt_inelastic_degeneracy(nu_cand='b8', Emin=1., Emax=7., bins=15,
                            Mnum=100, element='Germanium',
                            fnfp=1., delta=np.array([-40., -20., -10., 10., 20., 40., 100.]),
                            GF=False, time_info=False, xenlab='LZ',
                            model='sigma_si', fs=18,
                            c_list=np.array(['blue', 'green','red','violet','aqua','magenta','orange',
                                             'brown', 'goldenrod', 'salmon', 'grey', 'indianred']),
                            tag='_'):

    filename = test_plots + 'Model_Degeneracy_Inelastic_' + model + '_' + nu_cand + '_' + element
    filename += '_Emin_{:.2f}_Emax_{:.2f}'.format(Emin, Emax, delta)
    filename += tag + '.pdf'

    ergs = np.logspace(np.log10(Emin), np.log10(Emax), 200)
    experiment_info, Qmin, Qmax = Element_Info(element)
    labor = laboratory(element, xen=xenlab)


    nuspec = np.zeros_like(ergs)
    for iso in experiment_info:
        nuspec += Nu_spec(labor).nu_rate(nu_cand, ergs, iso)

    nu_events = np.trapz(nuspec, ergs)
    pl.figure()
    ax = pl.gca()

    ax.set_xlabel(r'Recoil Energy  [keV]', fontsize=fs)
    ax.set_ylabel(r'Event Rate  [${\rm ton}^{-1} {\rm yr}^{-1} {\rm keV}^{-1}$]', fontsize=fs)

    pl.plot(ergs, nuspec, 'k', lw=1, label=nu_cand)
    bfits = np.zeros(len(delta) * 3).reshape((len(delta), 3))
    dm_spec = np.zeros(len(delta) * len(ergs)).reshape((len(delta), len(ergs)))
    for i,dd in enumerate(delta):
        # minmass = v_leq_vesc(dd, experiment_info[0,0], vesc=533.+232.)
        # if minmass < 0:
        #     minmass = 0.1
        # width = brentq(lambda x: ERplus(x, experiment_info[0,0], 533.+232., dd) -
        #                  ERminus(x, experiment_info[-1,0], 533.+232., dd) - (Emax - Emin), minmass, 2000.)
        # minEmin = ERminus(width, experiment_info[-1,0], 533.+232., dd)
        # maxEmin = ERminus(2000., experiment_info[-1, 0], 533. + 232., dd)
        #
        # if minEmin > Emin:
        #     print 'Delta not compatible.'
        #     print 'Min mass {:.2f}, MinEmin {:.2f}'.format(width, minEmin)
        #     continue
        #
        # if maxEmin > Emin:
        #     maxmass = brentq(lambda x: ERminus(x, experiment_info[-1,0], 533.+232., dd) - Emin, width, 2000.)
        # else:
        #     maxmass = 100.

        # print 'For delta: {:.1f}, Mass range: [{:.2f}, {:.2f}]'.format(dd, minmass, maxmass)
        label = str(dd) + ' keV'
        coupling = "fnfp" + model[5:]

        # mass, cs, like = find_degeneracy(nu_cand='b8', Emin=Emin, Emax=Emax, bins=bins,
        #                            Mmin=minmass, Mmax=maxmass, Mnum=Mnum, element=element,
        #                            model=model, fnfp=fnfp,
        #                            delta=dd, GF=GF, time_info=time_info, xenlab=xenlab)
        mass, cs, like = find_degeneracy2(nu_cand=nu_cand, Emin=Emin, Emax=Emax, bins=bins,
                                         Mmin=minmass, Mmax=maxmass, Mnum=Mnum, element=element,
                                         model=model, fnfp=fnfp,
                                         delta=dd, GF=GF, time_info=time_info, xenlab=xenlab)
        bfits[i] = [mass, cs, like]
        drdq_params = default_rate_parameters.copy()
        drdq_params['element'] = element
        drdq_params['mass'] = mass
        drdq_params[model] = 10.**cs
        drdq_params[coupling] = fnfp
        drdq_params['delta'] = dd
        drdq_params['GF'] = GF
        drdq_params['time_info'] = time_info

        dm_spec[i] = dRdQ(ergs, np.zeros_like(ergs), **drdq_params) * 10. ** 3. * s_to_yr
        dmevts = np.trapz(dm_spec[i], ergs)
        dm_spec[i] *= nu_events/dmevts
        pl.plot(ergs, dm_spec[i], c_list[i], lw=1, ls='--', label=label)


    plt.xlim(xmin=Emin, xmax=Emax)
    plt.ylim(ymin=10. ** 0., ymax=3*10. ** 3.)
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.legend(loc=1, frameon=True, framealpha=0.5, fontsize=9, ncol=1, fancybox=True)
    plt.tight_layout()
    plt.savefig(filename)
    return


def gauss_reg(sig, dm, nu):
    rates = 10.**sig * dm
    ret = 0.
    for i in range(len(nu)):
        if rates[i] == 0 and nu[i] > 0:
            return 1e50
        elif rates[i] == 0 and nu[i] == 0:
            pass
        elif rates[i] > 0. and nu[i] == 0.:
            ret += rates[i]
        else:
            ret += ((rates[i] - nu[i])**2. / rates[i])

    return ret

def gauss_reg2(sig, dm, nu):
    rates = 10.**sig * dm
    ret = 0.
    for i in range(len(nu)):
        if nu[i] > 0:
            ret += ((rates[i] - nu[i])**2. / nu[i])
    return ret
